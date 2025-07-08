import json
from io import BytesIO
from pysubs2 import SSAFile, SSAEvent
from os import environ
from typing import Any
from pathlib import Path
from itertools import groupby
import re
import logging

from elevenlabs.types import SpeechToTextChunkResponseModel

from elevenlabs.client import ElevenLabs
import dotenv
import pysubs2

from baberu.LLMFactory.factory import AIToolFactory
from baberu.LLMFactory.llm.base import LLMProvider
from baberu.LLMFactory.transcription.base import TranscriptionProvider, TranscriptionResult, TranscribedSegment, TranscribedWord
from baberu.subtitling.types import SubtitleLine

logger = logging.getLogger(__name__)

CONTINUE_FLAG: str = "%%CONT%%"



def parse_elevenlabs_segmented(json_data: dict[str, Any],
                    delimiters: str | list[str] = [],
                    soft_delimiters: str | list[str] = [],
                    soft_max_lines: int = 20,
                    hard_max_lines: int = 50,
                    hard_max_carryover: int = 10,
                    model: str = "") -> SSAFile:
    """Converts an ElevenLabs transcription dictionary to a subtitle file object.

    This function processes word-level timestamp data, merges words into lines based
    on delimiters and length constraints, and formats the result as an SSAFile object.

    Args:
        json_data (dict[str, Any]): The ElevenLabs transcription data.
        delimiters (str | list[str]): Characters that force a line break.
        soft_delimiters (str | list[str]): Characters that suggest a line break
            when a line exceeds the soft length limit.
        include_audio_events (bool): If True, includes audio events (e.g., [laughs])
            in the subtitles.
        soft_max_lines (int): The preferred maximum character length for a line.
        hard_max_lines (int): The absolute maximum character length for a line before
            a hard split is performed.
        hard_max_carryover (int): The number of characters to carry to the next line
            during a hard split.
        model (str): The LLM model to use for intelligent line splitting.

    Returns:
        SSAFile: A pysubs2 SSAFile object containing the subtitles.
    """
    segments: list[dict[str, Any]] = json_data['segments']

    all_lines: list[dict[str, Any]] = []
    for segment in segments:
        if 'words' not in segment:
            continue

        # Separate audio events and words
        audio_events: list[dict[str, Any]] = []
        words: list[dict[str, Any]] = []

        for item in segment["words"]:
            if item["type"] == "word":
                words.append(item)

        # Merge words into lines by delimiter
        text_lines = _merge_words(words, delimiters, soft_delimiters, soft_max_lines, hard_max_lines, hard_max_carryover, model)

        # Combine audio events and merged words, and sort by start time
        combined_lines = sorted(audio_events + text_lines, key=lambda x: x["start"])

        all_lines.extend(combined_lines)

    # Create subtitle file objects
    sub_file: SSAFile = SSAFile()

    for line in all_lines:
        event = SSAEvent(
            start=pysubs2.time.times_to_ms(s=line.get("start", 0)),
            end=pysubs2.time.times_to_ms(s=line.get("end", 0)),
            text=line['text'],
            style="Default"
        )
        sub_file.events.append(event)

    logger.info(f"Converted {len(all_lines)} lines to ASS format")
    return sub_file


def write_transcript_json(json_data: dict[str, Any], 
                          output_file: Path) -> Path:
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Audio transcription saved to {output_file}")
    return output_file

def convert_transcript_to_subs(transcript: TranscriptionResult,
                    delimiters: str | list[str] = [],
                    soft_delimiters: str | list[str] = [],
                    soft_max_lines: int = 20,
                    hard_max_lines: int = 50,
                    hard_max_carryover: int = 10,
                    model: str = "") -> SSAFile:
    """Converts a transcription object to a subtitle file object.

    This function processes word-level timestamp data, merges words into lines based
    on delimiters and length constraints, and formats the result as an SSAFile object.
    """
    all_lines: list[SubtitleLine] = []

    for segment in transcript.segments:
        segment.words = [
            word for word in segment.words if word.type in ("word", "spacing")
        ]

        if not segment.words:
            continue

        # Merge segment into lines by delimiter
        segment_lines = _delimit_segment(segment, delimiters, soft_delimiters, soft_max_lines, hard_max_lines, hard_max_carryover, model)

        all_lines.extend(segment_lines)


    # Create subtitle file objects
    sub_file: SSAFile = SSAFile()

    for line in all_lines:
        event = SSAEvent(
            start=pysubs2.time.times_to_ms(s=line.start),
            end=pysubs2.time.times_to_ms(s=line.end),
            text=line.text,
            style="Default"
        )
        sub_file.events.append(event)

    logger.info(f"Converted {len(all_lines)} lines to ASS format")
    return sub_file

def _delimit_segment(
        segment: TranscribedSegment,
        delimiters: str | list[str],
        soft_delimiters: str | list[str],
        soft_max_lines: int,
        hard_max_lines: int,
        hard_max_carryover: int,
        model: str
        ) -> list[SubtitleLine]:
    """Merges a list of word objects into formatted subtitle lines."""

    sub_lines: list[SubtitleLine] = []
    current_words: list[TranscribedWord] = []

    for word in segment.words:
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
            sub_line = _create_subtitle_line(current_words, clean=True)
            sub_lines.append(sub_line)

            # Start new group
            remaining_word = word.model_copy()
            remaining_word.text = word.text[1:]
            current_words = [remaining_word]
            current_text = remaining_word.text

        # Force break on close quote
        if word.text.endswith("」"):
            force_break = True

        # Force break on open quote if length is greater than 1 (it will be stripped later if length is 1)
        elif word.text.endswith("「") and len(word.text.strip()) > 1:
            force_break = True

        # Break on optional delimiters
        elif any(word.text.endswith(d) for d in delimiters):
            force_break = True

        # Break on audio events
        elif word.type == "audio_event":
            force_break = True


        # Break on hard max limit
        if len(current_text) > hard_max_lines:
            carryover_word_count = 0
            try:
                leading_space: str = current_text[:len(current_text) - len(current_text.lstrip())]
                trailing_space: str = current_text[len(current_text.rstrip()):]
                trimmed_current_text: str = current_text.strip()

                sys_prompt = "Provide only the requested text without commentary or special formatting."
                user_prompt = f"Split the following text into two lines at a logical point without modifications to the text or punctuation:\n{trimmed_current_text}"

                client = AIToolFactory.get_llm_provider(model_name=model, system_prompt=sys_prompt)
                api_response = client.prompt(user_prompt)

                lines = api_response.strip().split('\n')
                line1_text = leading_space + lines[0].strip()
                line2_text = lines[1].strip() + trailing_space

                # Validate the API response
                if len(lines) == 2 and current_text.startswith(line1_text) and current_text.endswith(line2_text):
                    rebuilt_carryover_text = ""

                    # Reconstruct the second line from words to get an accurate word count
                    for word_obj in reversed(current_words):
                        rebuilt_carryover_text = word_obj.text + rebuilt_carryover_text
                        carryover_word_count += 1
                        if rebuilt_carryover_text.endswith(line2_text):
                            break # Match found
                    else:
                        # mismatch between the AI's split text and the source words.
                        logger.warning(f"Warning: AI-split line did not match word objects: '{line2_text}'")
                        raise ValueError
                else:
                    logger.warning(f"Warning: AI returned invalid value.")
                    raise ValueError

            except Exception as e:
                logger.warning(f"Warning: AI returned error: '{e}'\nAPI response: {api_response}\nOriginal:    {current_text}")
                # Fall back to character count method
                carryover_chars = 0
                
                # Count words to carry over
                for i in range(len(current_words) - 1, -1, -1):
                    word_text = current_words[i].text
                    if carryover_chars + len(word_text) <= hard_max_carryover:
                        carryover_chars += len(word_text)
                        carryover_word_count += 1
                    else:
                        break
            
            if 0 < carryover_word_count < len(current_words):
                # Keep some words in group
                words_to_keep = current_words[:-carryover_word_count]
                if words_to_keep:
                    words_to_keep[-1].text = words_to_keep[-1].text + f"{CONTINUE_FLAG}"
                    sub_line = _create_subtitle_line(words_to_keep, clean=True)
                    sub_lines.append(sub_line)

                # Start the new group with words carried over                 
                current_words = current_words[-carryover_word_count:]
                current_text = "".join(w.text for w in current_words)
            else:
                # The current word *alone* is longer than hard_max_lines.
                force_break = True

        # Break on soft delimiters if over soft max limit
        elif len(current_text) > soft_max_lines and any(word.text.endswith(d) for d in soft_delimiters):
            force_break = True

        if force_break:
            if current_words:
                # Skip lines containing only a delimiter
                sub_line = _create_subtitle_line(current_words, clean=True)
                if sub_line.text not in soft_delimiters + delimiters:
                    sub_lines.append(sub_line)

                current_words = []

    # Add any remaining words
    if current_words:
        sub_line = _create_subtitle_line(current_words, clean=True)
        if sub_line.text not in soft_delimiters + delimiters:
            sub_lines.append(sub_line)

    return sub_lines

def _create_subtitle_line(words: list[TranscribedWord], clean: bool = False) -> SubtitleLine:
    """Combines a group of word objects into a single line dictionary."""
    merged_text = "".join(w.text for w in words)

    sub_line = SubtitleLine(
        text=merged_text,
        start=words[0].start,
        end=words[-1].end,
        speaker=words[0].speaker
    )

    if clean:
        sub_line = _clean_subtitle_line(sub_line)

    return sub_line

def _clean_subtitle_line(line: SubtitleLine) -> SubtitleLine:
    line_text = line.text

    # Remove Japanese quotation marks
    line_text = line_text.replace("「", "").replace("」", "")

    # Remove hyphens at line start
    if line_text.startswith("-"):
            line_text = line_text[1:]

    # Trim white space
    line_text = line_text.strip()

    # Truncate repeated characters (5+ occurrences)
    line_text = re.sub(r'(.{1,6}?)(\1{4,})', r'\1\1\1', line_text)

    line.text = line_text
    return line