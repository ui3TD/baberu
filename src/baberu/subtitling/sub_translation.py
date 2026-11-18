import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
import logging

from baberu.providers.llm.base import COMPLETE, TRUNCATED, LLMProvider
from pysubs2 import SSAEvent, SSAFile

from baberu import constants
from baberu.errors import BaberuError
from baberu.settings import TranslationSettings
from baberu.subtitling import translation_prompts, translation_response
from baberu.subtitling.frames import Frame, FrameSource
from baberu.subtitling.types import LineRange
from baberu.tools.file_utils import atomic_write_text

logger = logging.getLogger(__name__)

ESTIMATED_SECONDS_PER_BATCH: float = 81
PLACEHOLDER = "[Translation missing]"
REPORT_VERSION = 1


@dataclass
class _Batch:
    """What every request for one batch shares, and the record of what happened to it."""
    context_prompt: str
    llms: list[LLMProvider]
    lang_from: str
    lang_to: str
    options: TranslationSettings
    frames: FrameSource | None = None
    attempts: list[dict] = field(default_factory=list)
    # Events whose translation came back the same as the source twice. They no longer count as problems.
    confirmed: set[int] = field(default_factory=set)

    @property
    def same_language(self) -> bool:
        return self.lang_from.split("-")[0].lower() == self.lang_to.split("-")[0].lower()


def translate(sub_file: SSAFile,
              output_file: Path,
              context_prompt: str,
              llm: LLMProvider,
              fallback_llm: LLMProvider | None,
              lang_from: str,
              lang_to: str,
              options: TranslationSettings,
              line_range: LineRange | None = None,
              report_file: Path | None = None,
              frames: FrameSource | None = None) -> tuple[list[str], dict]:
    """Translates a subtitle file, saving progress after every batch.

    Args:
        sub_file: The source subtitles.
        output_file: Where progress is saved. An existing file is resumed.
        context_prompt: Background notes to guide the translation.
        llm: The model to translate with.
        fallback_llm: A model to try on a batch the main model can't get right.
        lang_from: BCP 47 code for the source language.
        lang_to: BCP 47 code for the target language.
        options: Batch sizes and retry limits.
        line_range: Translates only these lines.
        report_file: Where the report is saved with every batch. Kept when resuming.
        frames: Video frames sent with each request to models that take images.

    Returns:
        The lines up to and including the last one translated, and a report of
        each batch: the requests made and the lines left unresolved.
    """
    starting_index: int = 0
    translated_lines: list[str] = []
    ending_translated_lines: list[str] = []
    range_start = line_range.start if line_range else 0

    if line_range:
        starting_index = line_range.start
        translated_lines = [i.text for i in sub_file.events[0:line_range.start]]
        ending_translated_lines = [i.text for i in sub_file.events[line_range.end + 1:]]

    resuming = output_file.exists()
    if resuming:
        translated_lines = output_file.read_text(encoding="utf-8").splitlines()
        starting_index = len(translated_lines)
    report = _load_report(report_file) if resuming else _new_report()

    total_items = len(sub_file.events)
    if starting_index >= total_items:
        logger.warning(f"All {starting_index} lines are already translated.")
        return translated_lines, report

    llms: list[LLMProvider] = [llm] + ([fallback_llm] if fallback_llm else [])
    ending_item = line_range.end + 1 if line_range else total_items
    start_time: float = time.perf_counter()
    i: int = starting_index
    processed_batches: int = 0

    try:
        while i < ending_item:
            batch_end: int = min(i + options.batch_lines, ending_item)
            continues_added: int = 0
            while batch_end < ending_item and \
                    sub_file.events[batch_end - 1].text.endswith(constants.CONTINUE_FLAG) and \
                    continues_added < options.max_cont_lines:
                batch_end += 1
                continues_added += 1

            batch_size: int = batch_end - i
            lookahead: int = min(options.discard_lines, ending_item - batch_end)
            current_batch = sub_file.events[i:(batch_end + lookahead)]

            processed_batches += 1
            remaining: int = ending_item - i
            total_batches: int = (remaining + options.batch_lines - 1) // options.batch_lines + processed_batches - 1
            _print_progress(time.perf_counter() - start_time, processed_batches, total_batches, i, batch_end, ending_item)
            logger.debug(f"Prompting batch {processed_batches}/{total_batches} (Lines {i+1}-{batch_end})")

            # Only this run's translations are context; source lines before a --lines range are not.
            previous = translated_lines[range_start:]
            batch = _Batch(context_prompt, llms, lang_from, lang_to, options, frames)
            results = _translate_batch(batch, current_batch, batch_size, previous, llms)
            new_lines, unresolved = _finish(results, current_batch, batch_size, i, batch)

            translated_lines.extend(new_lines)
            report["batches"].append({
                "lines": [i + 1, batch_end],
                "attempts": batch.attempts,
                "unresolved": unresolved,
            })
            write_lines(translated_lines, output_file)
            _save_report(report, report_file)

            i = batch_end

    except Exception as e:
        logger.error(f"Unexpected error during translation: {str(e)}")
        raise
    finally:
        duration_minutes, duration_seconds = divmod(int(time.perf_counter() - start_time), 60)
        logger.debug(f"Translation ran for {duration_minutes} min {duration_seconds} s")

    if ending_translated_lines:
        write_lines(translated_lines + ending_translated_lines, output_file)

    return translated_lines, report


def unresolved_lines(report: dict) -> dict[int, str]:
    """Every line a report lists as unresolved, by its line number."""
    found: dict[int, str] = {}
    for batch in report.get("batches", []):
        found.update({int(line): reason for line, reason in batch.get("unresolved", {}).items()})
    return dict(sorted(found.items()))


def _translate_batch(batch: _Batch,
                     events: list[SSAEvent],
                     kept: int,
                     previous: list[str],
                     llms: list[LLMProvider],
                     seed: list[str | None] | None = None) -> list[str | None]:
    """Translates the first `kept` events, trying each model in turn.

    Each model gets a full request, then targeted retries for the entries still
    wrong. A truncated response splits the batch instead of being resent. A
    refusal, an empty response, an error, or running out of retries moves on to
    the next model. Returns a translation, or None, for each kept event.

    A model that takes images also gets video frames: for the kept events in a
    full request, and for the entries asked for in a targeted retry.
    """
    results: list[str | None] = list(seed) if seed else [None] * len(events)

    for index, llm in enumerate(llms):
        is_last = index == len(llms) - 1
        for _ in range(batch.options.translate_retries):
            needed = _problems(batch, results, events, kept)
            if not needed:
                return results[:kept]

            targeted = len(needed) < kept
            text = _prompt(batch, events, results, needed, previous) if targeted else \
                translation_prompts.translate_prompt(previous, batch.context_prompt, events, batch.lang_from,
                                                     batch.lang_to, batch.options.context_lines)
            shown = _frames(batch, llm, [events[i] for i in sorted(needed)] if targeted else events[:kept])
            prompt = translation_prompts.with_frames(text, shown, events)
            attempt = {"model": llm.model, "asked": len(needed) if targeted else len(events)}
            if shown:
                attempt["frames"] = len(shown)
            batch.attempts.append(attempt)

            try:
                completion = llm.complete(prompt, cache_prefix=translation_prompts.context_section(batch.context_prompt))
            except Exception as error:
                attempt["status"] = f"error: {type(error).__name__}"
                if is_last:
                    raise
                logger.warning(f"{llm.model} failed ({type(error).__name__}: {error}). Trying {llms[index + 1].model}.")
                break

            attempt["status"] = completion.status
            if completion.status == TRUNCATED:
                return _split(batch, events, kept, previous, llms[index:], results)
            if completion.status != COMPLETE:
                logger.warning(f"Moving on from {llm.model} after a {completion.status} response.")
                break

            parsed = translation_response.parse_response(completion.text, len(events))
            for i, line in enumerate(parsed):
                if line is None or not (i in needed or (not targeted and i >= kept)):
                    continue
                if needed.get(i) == translation_response.UNTRANSLATED and line == results[i]:
                    batch.confirmed.add(id(events[i]))
                results[i] = line
            attempt["problems"] = len(_problems(batch, results, events, kept))
            if attempt["problems"]:
                logger.warning(f"Translation attempt with {llm.model}: {attempt['problems']} of {kept} lines still need work.")

    return results[:kept]


def _frames(batch: _Batch, llm: LLMProvider, events: list[SSAEvent]) -> list[Frame]:
    """The frames for these events, or none when there is no video or the model can't take images."""
    if batch.frames is None or not llm.takes_images():
        return []
    return batch.frames.for_events(events)


def _problems(batch: _Batch, results: list[str | None], events: list[SSAEvent], kept: int) -> dict[int, str]:
    found = translation_response.problems(results, events, kept, batch.same_language)
    return {i: reason for i, reason in found.items() if id(events[i]) not in batch.confirmed}


def _prompt(batch: _Batch,
            events: list[SSAEvent],
            results: list[str | None],
            needed: dict[int, str],
            previous: list[str]) -> str:
    accepted = {i: line for i, line in enumerate(results) if line is not None and i not in needed}
    return translation_prompts.targeted_retry_prompt(
        previous, batch.context_prompt, events, accepted, needed,
        batch.lang_from, batch.lang_to, batch.options.context_lines)


def _split(batch: _Batch,
           events: list[SSAEvent],
           kept: int,
           previous: list[str],
           llms: list[LLMProvider],
           results: list[str | None]) -> list[str | None]:
    """Answers a truncated response with smaller requests, keeping what was already accepted."""
    if len(events) > kept:
        logger.warning("Response was truncated. Retrying without the lookahead lines.")
        return _translate_batch(batch, events[:kept], kept, previous, llms, results[:kept])
    if kept == 1:
        logger.warning("Response was truncated on a single line.")
        if len(llms) > 1:
            return _translate_batch(batch, events, kept, previous, llms[1:], results)
        return results[:kept]

    half = kept // 2
    logger.warning(f"Response was truncated. Splitting {kept} lines into {half} and {kept - half}.")
    first = _translate_batch(batch, events[:half], half, previous, llms, results[:half])
    done = previous + [line or "" for line in first]
    second = _translate_batch(batch, events[half:], kept - half, done, llms, results[half:kept])
    return first + second


def _finish(results: list[str | None],
            events: list[SSAEvent],
            kept: int,
            offset: int,
            batch: _Batch) -> tuple[list[str], dict[str, str]]:
    """Fills the lines still wrong, and lists them by line number.

    A missing or empty line gets a placeholder at exactly its place. A line that
    still matches its source is kept and listed, unless the model returned it
    unchanged twice; then it isn't listed at all.
    """
    lines = list(results[:kept]) + [None] * (kept - len(results))
    unresolved: dict[str, str] = {}
    for i, reason in _problems(batch, lines, events, kept).items():
        unresolved[str(offset + i + 1)] = reason
        if reason != translation_response.UNTRANSLATED:
            lines[i] = PLACEHOLDER
    if unresolved:
        logger.warning(f"Lines left unresolved: {', '.join(unresolved)}")
    return [line or "" for line in lines], unresolved


def _new_report() -> dict:
    return {"version": REPORT_VERSION, "batches": []}


def _load_report(report_file: Path | None) -> dict:
    if report_file is None or not report_file.exists():
        return _new_report()
    try:
        report = json.loads(report_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        logger.warning(f"Can't read {report_file.name} ({error}); starting a new one.")
        return _new_report()
    return report if isinstance(report.get("batches"), list) else _new_report()


def _save_report(report: dict, report_file: Path | None) -> None:
    if report_file is not None:
        atomic_write_text(report_file, json.dumps(report, ensure_ascii=False, indent=2))


def generate_context(subtitles: SSAFile, llm: LLMProvider, filename: str, lang_from: str, lang_to: str,
                     details: str = "") -> str:
    """Asks a web-grounded model for a summary and glossary to guide translation."""
    prompt = translation_prompts.context_prompt(subtitles, filename, lang_from, lang_to, details)
    completion = llm.complete(prompt, grounding=True)
    if not completion.text.strip():
        raise BaberuError(f"{llm.model} returned no context ({completion.status}). "
                          f"Try again, or give your own notes with a context file.")
    return completion.text


def load_context(file_path: Path) -> str:
    """Loads a context prompt from a text file.

    Args:
        file_path: The path to the context file.

    Returns:
        The content of the file as a string, or an empty string on error.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning(f"Warning: Context file '{file_path}' not found. Using empty context.")
        return ""
    except Exception as e:
        logger.warning(f"Error reading context file: {str(e)}. Using empty context.")
        return ""


def write_lines(lines: list[str], output_file: Path) -> None:
    """Writes each string on its own line, with a newline after the last one.

    Args:
        lines: The list of strings to write.
        output_file: The path to the output text file.
    """
    atomic_write_text(output_file, '\n'.join(lines) + '\n')


def _print_progress(elapsed_time: float,
                    current_batch: int,
                    total_batches: int,
                    start_item: int,
                    end_item: int,
                    total_item: int) -> None:
    """Prints translation progress and estimated time remaining."""
    time_per_batch: float = (ESTIMATED_SECONDS_PER_BATCH * 4 + elapsed_time) / (4 + current_batch - 1)
    remaining_time: float = time_per_batch * (total_batches - current_batch + 1)
    remaining_minutes, remaining_seconds = divmod(int(remaining_time), 60)
    print(f"Translating batch {current_batch}/{total_batches} ({start_item+1}-{end_item}/{total_item} items) - ETC: {remaining_minutes}m {remaining_seconds}s", file=sys.stderr)
