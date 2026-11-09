import sys
import time
from pathlib import Path
import logging

from LLMFactory.llm.base import LLMProvider
from pysubs2 import SSAEvent, SSAFile

from baberu import constants
from baberu.settings import TranslationSettings
from baberu.subtitling import translation_prompts, translation_response
from baberu.subtitling.types import LineRange
from baberu.tools.file_utils import atomic_write_text

logger = logging.getLogger(__name__)

ESTIMATED_SECONDS_PER_BATCH: float = 81


def translate(sub_file: SSAFile,
              output_file: Path,
              context_prompt: str,
              llm: LLMProvider,
              fallback_llm: LLMProvider | None,
              lang_from: str,
              lang_to: str,
              options: TranslationSettings,
              line_range: LineRange | None = None) -> list[str]:
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

    Returns:
        The lines up to and including the last one translated.
    """
    starting_index: int = 0
    translated_lines: list[str] = []
    ending_translated_lines: list[str] = []

    if line_range:
        starting_index = line_range.start
        translated_lines = [i.text for i in sub_file.events[0:line_range.start]]
        ending_translated_lines = [i.text for i in sub_file.events[line_range.end + 1:]]

    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            translated_lines = [line.strip() for line in f.readlines()]
        starting_index = len(translated_lines)

    total_items = len(sub_file.events)
    if starting_index >= total_items:
        logger.warning(f"All {starting_index} lines are already translated.")
        return translated_lines

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
            if not current_batch:
                break

            prompt: str = translation_prompts.translate_prompt(
                translated_lines, context_prompt, current_batch, lang_from, lang_to, options.context_lines)

            processed_batches += 1
            remaining: int = ending_item - i
            total_batches: int = (remaining + options.batch_lines - 1) // options.batch_lines + processed_batches - 1
            _print_progress(time.perf_counter() - start_time, processed_batches, total_batches, i, batch_end, ending_item)

            logger.debug(f"Prompting batch {processed_batches}/{total_batches} (Lines {i+1}-{batch_end})")
            new_lines: list[str] = _translate_batch(
                prompt, current_batch, llms, lang_from, lang_to, options.translate_retries)

            # The lookahead lines were only there for context.
            translated_lines.extend(new_lines[:batch_size])
            write_lines(translated_lines, output_file)

            i = batch_end

    except Exception as e:
        logger.error(f"Unexpected error during translation: {str(e)}")
        raise
    finally:
        duration_minutes, duration_seconds = divmod(int(time.perf_counter() - start_time), 60)
        logger.debug(f"Translation ran for {duration_minutes} min {duration_seconds} s")

    if ending_translated_lines:
        write_lines(translated_lines + ending_translated_lines, output_file)

    return translated_lines


def _translate_batch(prompt: str,
                     current_batch: list[SSAEvent],
                     llms: list[LLMProvider],
                     lang_from: str,
                     lang_to: str,
                     translate_retries: int) -> list[str]:
    """Gets a translation with the right line count, trying each model in turn."""
    new_lines: list[str] = []

    for llm in llms:
        attempt_prompt: str = prompt
        for attempt in range(translate_retries):
            new_lines = translation_response.parse_response(llm.prompt(attempt_prompt))

            if len(new_lines) == len(current_batch):
                return new_lines

            logger.warning(f"Translation attempt {attempt+1} with {llm.model} failed: expected {len(current_batch)} lines, got {len(new_lines)}.")
            if abs(len(current_batch) - len(new_lines)) <= 10:
                attempt_prompt = translation_prompts.retry_prompt(new_lines, current_batch, lang_from, lang_to)

    logger.warning(f"No model returned the exact line count after {translate_retries} attempts each. Proceeding with best effort.")
    return translation_response.force_line_count(new_lines, current_batch)


def generate_context(subtitles: SSAFile, llm: LLMProvider, filename: str, lang_from: str, lang_to: str) -> str:
    """Asks a web-grounded model for a summary and glossary to guide translation."""
    return llm.prompt(translation_prompts.context_prompt(subtitles, filename, lang_from, lang_to), grounding=True)


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
    """Writes a list of strings to a text file, with each string on a new line.

    Args:
        lines: The list of strings to write.
        output_file: The path to the output text file.
    """
    atomic_write_text(output_file, '\n'.join(lines))


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
