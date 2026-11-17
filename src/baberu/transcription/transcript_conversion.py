from collections.abc import Callable
import functools
import hashlib
import json
from pysubs2 import SSAFile, SSAEvent
from typing import Any
from pathlib import Path
import logging

import pysubs2

from baberu.providers.factory import AIToolFactory
from baberu.providers.llm.base import LLMProvider
from baberu.providers.transcription.base import TranscriptionResult, TranscribedSegment, TranscribedWord
from baberu.constants import CONTINUE_FLAG
from baberu.settings import AnomalySettings, ParsingSettings
from baberu.subtitling.types import SubtitleLine
from baberu.tools.file_utils import atomic_write_text
from baberu.transcription.health import count_repair, letters
from baberu.transcription.normalize import normalize
from baberu.transcription.repair import repair

logger = logging.getLogger(__name__)

SPLIT_SYSTEM_PROMPT = "Provide only the requested text without commentary or special formatting."
SPLIT_USER_PROMPT = "Split the following text into two lines at a logical point without modifications to the text or punctuation:"

def write_transcript_json(json_data: dict[str, Any], 
                          output_file: Path) -> Path:
    atomic_write_text(output_file, json.dumps(json_data, ensure_ascii=False, indent=2))
    return output_file

def load_transcript_json(json_file: Path) -> TranscriptionResult:
    """Reads a saved transcript, using the parser for whichever service wrote it."""
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Scribe responses have language_code; Whisper responses have language.
    model = "scribe" if "language_code" in json_data else "whisper-1"
    provider_type = AIToolFactory.get_transcription_provider_type(model)
    return provider_type.parse(provider_type.validate(json_data))

def split_key(model: str, text: str) -> str:
    """A cache key for one line-split question, from the model and the text."""
    return hashlib.sha256(f"{model}\n{text}".encode("utf-8")).hexdigest()[:24]

def _split_asker(parsing: ParsingSettings, split_cache: dict[str, str] | None) -> Callable[[str], str]:
    """Asks the parsing model where to split a line, remembering answers in `split_cache`."""
    get_splitter: Callable[[], LLMProvider] = functools.cache(lambda: AIToolFactory.get_llm_provider(
        model_name=parsing.parsing_model, system_prompt=SPLIT_SYSTEM_PROMPT,
        effort=parsing.parsing_effort))
    cache = split_cache if split_cache is not None else {}

    def ask(text: str) -> str:
        key = split_key(parsing.parsing_model, text)
        if key not in cache:
            logger.debug(f"Using AI to split line: '{text}'")
            cache[key] = get_splitter().prompt(f"{SPLIT_USER_PROMPT}\n{text}")
        return cache[key]

    return ask

def convert_transcript_to_subs(transcript: TranscriptionResult,
                               parsing: ParsingSettings,
                               split_cache: dict[str, str] | None = None,
                               limits: AnomalySettings | None = None) -> SSAFile:
    """Turns word-level timings into subtitle lines using delimiters and length limits.

    The transcript is normalized and repaired in a copy; the caller's is unchanged.
    `split_cache` holds the parsing model's answers by question; a question already
    in it is not asked again. Every line is moved by `parsing.shift_sec`, starting
    no earlier than 0.
    """
    get_splitter = _split_asker(parsing, split_cache)
    transcript = repair(normalize(transcript, limits), limits or AnomalySettings())
    all_lines: list[SubtitleLine] = []

    for segment in transcript.segments:
        all_lines.extend(_delimit_segment(segment, parsing, get_splitter))
    all_lines = _merge_fragments(all_lines, parsing)

    sub_file: SSAFile = SSAFile()
    shift_ms = pysubs2.time.times_to_ms(s=parsing.shift_sec)

    for line in all_lines:
        start = max(pysubs2.time.times_to_ms(s=line.start) + shift_ms, 0)
        event = SSAEvent(
            start=start,
            end=max(pysubs2.time.times_to_ms(s=line.end) + shift_ms, start + 1),  # never zero or negative length
            text=line.text,
            style="Default"
        )
        sub_file.events.append(event)

    logger.debug(f"Converted {len(all_lines)} lines to ASS format")
    return sub_file

def _delimit_segment(segment: TranscribedSegment,
                     parsing: ParsingSettings,
                     get_splitter: Callable[[str], str]) -> list[SubtitleLine]:
    """Merges a list of word objects into formatted subtitle lines."""
    delimiters = parsing.delimiters
    soft_delimiters = parsing.soft_delimiters
    remove_text = parsing.remove_text

    sub_lines: list[SubtitleLine] = []
    current_words: list[TranscribedWord] = []

    def finish_line(words: list[TranscribedWord]) -> None:
        sub_line = _create_subtitle_line(words, remove_text, clean=True)
        sub_line.open_end = not _ends_with(words[-1].text, delimiters + soft_delimiters)             and not words[-1].text.endswith(CONTINUE_FLAG)
        if letters(sub_line.text):  # Lines of only punctuation, like "。」" or "...", are dropped
            sub_lines.append(sub_line)

    for word in segment.words:
        # Break on time gap
        if current_words and word.start - current_words[-1].end > parsing.max_time_gap_sec:
            finish_line(current_words)
            current_words = []

        current_words.append(word)
        current_text: str = "".join(w.text for w in current_words)
        force_break = False

        # Break if first character is a delimiter
        if (len(current_words) > 1 and
            len(word.text) > 1 and
            any(word.text.startswith(d) for d in delimiters)):

            # Include the delimiter in prev group
            current_words.pop()
            current_words[-1].text = current_words[-1].text + word.text[0]
            finish_line(current_words)

            # Start new group
            remaining_word = word.model_copy()
            remaining_word.text = word.text[1:]
            current_words = [remaining_word]
            current_text = remaining_word.text

        # Break on optional delimiters
        elif _ends_with(word.text, delimiters):
            force_break = True

        # Break on hard max limit
        if len(current_text) > parsing.hard_max_chars:
            carryover_word_count = _count_carryover_words(current_words, current_text, parsing, get_splitter)

            if 0 < carryover_word_count < len(current_words):
                # Keep some words in group
                words_to_keep = current_words[:-carryover_word_count]
                if words_to_keep:
                    words_to_keep[-1].text = words_to_keep[-1].text + f"{CONTINUE_FLAG}"
                    finish_line(words_to_keep)

                # Start the new group with words carried over
                current_words = current_words[-carryover_word_count:]
                current_text = "".join(w.text for w in current_words)
            else:
                # The current word *alone* is longer than the limit.
                force_break = True

        # Break on soft delimiters if over soft max limit
        elif len(current_text) > parsing.soft_max_chars and _ends_with(word.text, soft_delimiters):
            force_break = True

        if force_break and current_words:
            finish_line(current_words)
            current_words = []

    # Add any remaining words
    if current_words:
        finish_line(current_words)

    return sub_lines

_OPEN_BRACKETS = "（("
_CLOSE_BRACKETS = "）)"


def _merge_fragments(lines: list[SubtitleLine], parsing: ParsingSettings) -> list[SubtitleLine]:
    """Joins a word that a pause cut in two back onto its line, such as 乾…杯 or 優勝で…す.

    The tail is joined
    back when the line before it ends mid-word (no delimiter, comma or space) and
    the tail is only a letter or two, or when the break falls inside brackets such
    as （聞き取り不能）. A line that starts with a space is never joined.
    """
    merged: list[SubtitleLine] = []
    for line in lines:
        previous = merged[-1] if merged else None
        if previous and _cut_mid_word(previous, line, parsing):
            previous.text += line.text
            previous.end = max(previous.end, line.end)
            previous.open_end = line.open_end
            count_repair("fragment_rejoined")
            continue
        merged.append(line)
    return merged


def _cut_mid_word(previous: SubtitleLine, line: SubtitleLine, parsing: ParsingSettings) -> bool:
    text = previous.text
    if not previous.open_end or line.text[:1].isspace():
        return False
    if line.start - previous.end > parsing.fragment_max_gap_sec:
        return False
    unclosed = sum(text.count(c) for c in _OPEN_BRACKETS) > sum(text.count(c) for c in _CLOSE_BRACKETS)
    return unclosed or letters(line.text) <= parsing.fragment_max_letters


def _count_carryover_words(words: list[TranscribedWord],
                           text: str,
                           parsing: ParsingSettings,
                           get_splitter: Callable[[str], str]) -> int:
    """Decides how many trailing words move to the next subtitle line."""
    if parsing.parsing_model:
        response = ""
        try:
            response = get_splitter(text.strip())
            count = _words_in_second_line(words, text, response)
            count_repair("line_split_by_model")
            return count
        except Exception as e:
            logger.warning(f"Warning: AI split failed: '{e}'\nAPI response: {response}\nOriginal:    {text}")

    count_repair("line_split_by_length")
    carryover_chars = 0
    carryover_words = 0
    for word in reversed(words):
        if carryover_chars + len(word.text) > parsing.hard_max_carryover:
            break
        carryover_chars += len(word.text)
        carryover_words += 1
    return carryover_words

def _words_in_second_line(words: list[TranscribedWord], text: str, response: str) -> int:
    """Checks a model's two-line split against the source and maps it back to words."""
    leading_space = text[:len(text) - len(text.lstrip())]
    trailing_space = text[len(text.rstrip()):]

    lines = response.strip().split('\n')
    if len(lines) != 2:
        raise ValueError("expected exactly two lines")

    first_line = leading_space + lines[0].strip()
    second_line = lines[1].strip() + trailing_space
    if not (text.startswith(first_line) and text.endswith(second_line)):
        raise ValueError("split does not match the original text")

    rebuilt = ""
    for count, word in enumerate(reversed(words), start=1):
        rebuilt = word.text + rebuilt
        if rebuilt.endswith(second_line):
            return count
    raise ValueError("split does not fall on a word boundary")

def _ends_with(text: str, delimiters: list[str]) -> bool:
    """Whether a word ends a phrase. A space after the word doesn't hide its punctuation."""
    return any(text.endswith(d) or (d.strip() and text.rstrip().endswith(d)) for d in delimiters)

def _create_subtitle_line(words: list[TranscribedWord], remove_text: list[str], clean: bool = False) -> SubtitleLine:
    """Combines a group of word objects into a single line dictionary."""
    merged_text = "".join(w.text for w in words)

    sub_line = SubtitleLine(
        text=merged_text,
        start=words[0].start,
        end=words[-1].end,
        speaker=words[0].speaker
    )

    if clean:
        sub_line = _clean_subtitle_line(sub_line, remove_text)

    return sub_line

def _clean_subtitle_line(line: SubtitleLine, remove_text: list[str]) -> SubtitleLine:
    line_text = line.text

    if remove_text:
        for text_to_remove in remove_text:
            line_text = line_text.replace(text_to_remove, "")

    if line_text.startswith("-"):
            line_text = line_text[1:]

    line_text = line_text.strip()

    line.text = line_text
    return line

