from collections.abc import Callable
import functools
import json
from pysubs2 import SSAFile, SSAEvent
from typing import Any
from pathlib import Path
import re
import logging

import pysubs2

from LLMFactory.factory import AIToolFactory
from LLMFactory.llm.base import LLMProvider
from LLMFactory.transcription.base import TranscriptionResult, TranscribedSegment, TranscribedWord
from baberu.constants import CONTINUE_FLAG
from baberu.settings import ParsingSettings
from baberu.subtitling.types import SubtitleLine

logger = logging.getLogger(__name__)

SPLIT_SYSTEM_PROMPT = "Provide only the requested text without commentary or special formatting."
SPLIT_USER_PROMPT = "Split the following text into two lines at a logical point without modifications to the text or punctuation:"

def write_transcript_json(json_data: dict[str, Any], 
                          output_file: Path) -> Path:
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Audio transcription saved to {output_file}")
    return output_file

def load_transcript_json(json_file: Path) -> TranscriptionResult:
    """Reads a saved transcript, using the parser for whichever service wrote it."""
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Scribe responses have language_code; Whisper responses have language.
    model = "scribe" if "language_code" in json_data else "whisper-1"
    provider_type = AIToolFactory.get_transcription_provider_type(model)
    return provider_type.parse(provider_type.validate(json_data))

def convert_transcript_to_subs(transcript: TranscriptionResult, parsing: ParsingSettings) -> SSAFile:
    """Turns word-level timings into subtitle lines using delimiters and length limits."""
    get_splitter: Callable[[], LLMProvider] = functools.cache(lambda: AIToolFactory.get_llm_provider(
        model_name=parsing.parsing_model, system_prompt=SPLIT_SYSTEM_PROMPT))
    all_lines: list[SubtitleLine] = []

    for segment in transcript.segments:
        segment.words = [
            word for word in segment.words if word.type in ("word", "spacing")
        ]

        if segment.words:
            all_lines.extend(_delimit_segment(segment, parsing, get_splitter))

    sub_file: SSAFile = SSAFile()

    for line in all_lines:
        event = SSAEvent(
            start=pysubs2.time.times_to_ms(s=line.start),
            end=pysubs2.time.times_to_ms(s=line.end),
            text=line.text,
            style="Default"
        )
        sub_file.events.append(event)

    logger.debug(f"Converted {len(all_lines)} lines to ASS format")
    return sub_file

def _delimit_segment(segment: TranscribedSegment,
                     parsing: ParsingSettings,
                     get_splitter: Callable[[], LLMProvider]) -> list[SubtitleLine]:
    """Merges a list of word objects into formatted subtitle lines."""
    delimiters = parsing.delimiters
    soft_delimiters = parsing.soft_delimiters
    remove_text = parsing.remove_text

    sub_lines: list[SubtitleLine] = []
    current_words: list[TranscribedWord] = []

    for word in segment.words:
        # Break on time gap
        if current_words and word.start - current_words[-1].end > parsing.max_time_gap_sec:
            sub_line = _create_subtitle_line(current_words, remove_text, clean=True)
            if sub_line.text not in soft_delimiters + delimiters:
                sub_lines.append(sub_line)
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
            sub_line = _create_subtitle_line(current_words, remove_text, clean=True)
            sub_lines.append(sub_line)

            # Start new group
            remaining_word = word.model_copy()
            remaining_word.text = word.text[1:]
            current_words = [remaining_word]
            current_text = remaining_word.text

        # Break on optional delimiters
        elif any(word.text.endswith(d) for d in delimiters):
            force_break = True

        # Break on audio events
        elif word.type == "audio_event":
            force_break = True

        # Break on hard max limit
        if len(current_text) > parsing.hard_max_chars:
            carryover_word_count = _count_carryover_words(current_words, current_text, parsing, get_splitter)

            if 0 < carryover_word_count < len(current_words):
                # Keep some words in group
                words_to_keep = current_words[:-carryover_word_count]
                if words_to_keep:
                    words_to_keep[-1].text = words_to_keep[-1].text + f"{CONTINUE_FLAG}"
                    sub_line = _create_subtitle_line(words_to_keep, remove_text, clean=True)
                    sub_lines.append(sub_line)

                # Start the new group with words carried over
                current_words = current_words[-carryover_word_count:]
                current_text = "".join(w.text for w in current_words)
            else:
                # The current word *alone* is longer than the limit.
                force_break = True

        # Break on soft delimiters if over soft max limit
        elif len(current_text) > parsing.soft_max_chars and any(word.text.endswith(d) for d in soft_delimiters):
            force_break = True

        if force_break:
            if current_words:
                # Skip lines containing only a delimiter
                sub_line = _create_subtitle_line(current_words, remove_text, clean=True)
                if sub_line.text not in soft_delimiters + delimiters:
                    sub_lines.append(sub_line)

                current_words = []

    # Add any remaining words
    if current_words:
        sub_line = _create_subtitle_line(current_words, remove_text, clean=True)
        if sub_line.text not in soft_delimiters + delimiters:
            sub_lines.append(sub_line)

    return sub_lines

def _count_carryover_words(words: list[TranscribedWord],
                           text: str,
                           parsing: ParsingSettings,
                           get_splitter: Callable[[], LLMProvider]) -> int:
    """Decides how many trailing words move to the next subtitle line."""
    if parsing.parsing_model:
        response = ""
        try:
            logger.debug(f"Using AI to split line: '{text.strip()}'")
            response = get_splitter().prompt(f"{SPLIT_USER_PROMPT}\n{text.strip()}")
            return _words_in_second_line(words, text, response)
        except Exception as e:
            logger.warning(f"Warning: AI split failed: '{e}'\nAPI response: {response}\nOriginal:    {text}")

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

    # Remove text
    if remove_text:
        for text_to_remove in remove_text:
            line_text = line_text.replace(text_to_remove, "")

    # Remove hyphens at line start
    if line_text.startswith("-"):
            line_text = line_text[1:]

    # Trim white space
    line_text = line_text.strip()

    # Truncate repeated characters (5+ occurrences)
    line_text = re.sub(r'(.{1,6}?)(\1{4,})', r'\1\1\1', line_text)

    line.text = line_text
    return line

