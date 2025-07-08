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

logger = logging.getLogger(__name__)

CONTINUE_FLAG: str = "%%CONT%%"

def load_elevenlabs_json_segmented(file_path: Path) -> dict[str, Any]:
    """Loads JSON data from a file.

    Args:
        file_path (Path): Path to the JSON file.

    Returns:
        dict[str, Any]: A dictionary containing the loaded JSON data.
    """

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: File '{file_path}' not found.")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error: File '{file_path}' contains invalid JSON.")
        raise
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        raise

def transcribe_audio_segmented(audio_file: Path, lang: str, model: str) -> dict[str, Any]:
    """Transcribes an audio file using the ElevenLabs API.

    Args:
        audio_file (Path): Path to the audio file to transcribe.
        num_speakers (int): The number of speakers to detect in the audio.
        lang (str): The language code of the audio (e.g., 'en', 'ja').
        model (str): The model ID to use for transcription.

    Returns:
        dict[str, Any]: The transcription result as a dictionary.
    """
    logger.info(f"Transcribing audio from {audio_file}...")
    
    dotenv.load_dotenv()
    client: ElevenLabs = ElevenLabs(
        api_key=environ.get("ELEVENLABS_API_KEY"),
    )

    with open(audio_file, 'rb') as f:
        audio_data: BytesIO = BytesIO(f.read())

    segmented_json = {
        "format": "segmented_json"
    }

    transcription: SpeechToTextChunkResponseModel = client.speech_to_text.convert(
        file=audio_data,
        model_id=model,
        tag_audio_events=False,
        language_code=lang,
        diarize=True,
        diarization_threshold=0.1,
        timestamps_granularity="word",
        additional_formats= json.dumps([segmented_json]),
        request_options = {"timeout_in_seconds": 3600}
    )
        
    # Parse JSON
    content: str = transcription.additional_formats[0].content
    parsed_content = json.loads(content)

    return parsed_content

def write_elevenlabs_json_segmented(json_data: dict[str, Any], 
                          output_file: Path) -> Path:
    """Writes a dictionary to a JSON file.

    Args:
        json_data (dict[str, Any]): The dictionary to write.
        output_file (Path): The path to the output JSON file.

    Returns:
        Path: The path to the created file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Audio transcription completed. JSON saved to {output_file}")
    return output_file

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

def load_elevenlabs_json(file_path: Path) -> SpeechToTextChunkResponseModel:
    """Loads JSON data from a file.

    Args:
        file_path (Path): Path to the JSON file.
    """

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data: dict[str, Any] = json.load(f)
        return SpeechToTextChunkResponseModel.model_validate(json_data)
    except FileNotFoundError:
        logger.error(f"Error: File '{file_path}' not found.")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error: File '{file_path}' contains invalid JSON.")
        raise
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        raise

def transcribe_audio(audio_file: Path, lang: str, model: str) -> SpeechToTextChunkResponseModel:
    """Transcribes an audio file using the ElevenLabs API.

    Args:
        audio_file (Path): Path to the audio file to transcribe.
        num_speakers (int): The number of speakers to detect in the audio.
        lang (str): The language code of the audio (e.g., 'en', 'ja').
        model (str): The model ID to use for transcription.
    """
    logger.info(f"Transcribing audio from {audio_file}...")
    
    dotenv.load_dotenv()
    client: ElevenLabs = ElevenLabs(
        api_key=environ.get("ELEVENLABS_API_KEY"),
    )

    with open(audio_file, 'rb') as f:
        audio_data: BytesIO = BytesIO(f.read())

    transcription: SpeechToTextChunkResponseModel = client.speech_to_text.convert(
        file=audio_data,
        model_id=model,
        tag_audio_events=False,
        language_code=lang,
        diarize=True,
        diarization_threshold=0.1,
        timestamps_granularity="word",
        request_options = {"timeout_in_seconds": 3600}
    )

    return transcription

def write_elevenlabs_json(el_data: SpeechToTextChunkResponseModel, 
                          output_file: Path) -> Path:
    json_data = el_data.model_dump()
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Audio transcription completed. JSON saved to {output_file}")
    return output_file


def parse_elevenlabs(el_data: SpeechToTextChunkResponseModel,
                    delimiters: str | list[str] = [],
                    soft_delimiters: str | list[str] = [],
                    soft_max_lines: int = 20,
                    hard_max_lines: int = 50,
                    hard_max_carryover: int = 10,
                    model: str = "") -> SSAFile:
    """Converts an ElevenLabs transcription object to a subtitle file object.

    This function processes word-level timestamp data, merges words into lines based
    on delimiters and length constraints, and formats the result as an SSAFile object.
    """
    transcription_words = el_data.words
    all_lines: list[dict[str, Any]] = []

    # Group words by speaker_id to define segments
    for _, segment_words_iterator in groupby(transcription_words, key=lambda w: w.speaker_id):
        segment_words = list(segment_words_iterator)
        
        words: list[dict[str, Any]] = [
            item.model_dump() for item in segment_words if item.type in ["word", "spacing"]
        ]

        if not words:
            continue

        # Merge words into lines by delimiter
        text_lines = _merge_words(words, delimiters, soft_delimiters, soft_max_lines, hard_max_lines, hard_max_carryover, model)

        all_lines.extend(text_lines)


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

def load_transcript_json(file_path: Path) -> TranscriptionResult:
    """Loads JSON data from a file.

    Args:
        file_path (Path): Path to the JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data: dict[str, Any] = json.load(f)
        return TranscriptionResult.model_validate(json_data)
    except FileNotFoundError:
        logger.error(f"Error: File '{file_path}' not found.")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error: File '{file_path}' contains invalid JSON.")
        raise
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        raise
    
def write_transcript_json(json_data: dict[str, Any], 
                          output_file: Path) -> Path:
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Audio transcription completed. JSON saved to {output_file}")
    return output_file

def parse_transcript(transcript: TranscriptionResult,
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
    all_lines: list[dict[str, Any]] = []

    # Group words by speaker_id to define segments
    for segments in transcript.segments:
        words: list[dict[str, Any]] = [
            item.model_dump() for item in segments.words if item.type in ["word", "spacing"]
        ]

        if not words:
            continue

        # Merge words into lines by delimiter
        text_lines = _merge_words(words, delimiters, soft_delimiters, soft_max_lines, hard_max_lines, hard_max_carryover, model)

        all_lines.extend(text_lines)


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

def _merge_words(words: list[dict[str, Any]],
                delimiters: str | list[str],
                soft_delimiters: str | list[str],
                soft_max_lines: int,
                hard_max_lines: int,
                hard_max_carryover: int,
                model: str) -> list[dict[str, Any]]:
    """Merges a list of word objects into formatted subtitle lines."""

    merged_words: list[dict[str, Any]] = []
    current_group: list[dict[str, Any]] = []

    for word in words:
        current_group.append(word)
        current_text: str = "".join(w["text"] for w in current_group)
        force_break = False

        # Break if first character is a delimiter
        if (len(current_group) > 1 and
            len(word["text"]) > 1 and
            any(word["text"].startswith(d) for d in delimiters)):

            # Include the delimiter in prev group
            current_group.pop()
            current_group[-1]["text"] = current_group[-1]["text"] + word["text"][0]
            merged_words.append(_create_merged_line(current_group))

            # Start new group
            remaining_word = word.copy()
            remaining_word["text"] = word["text"][1:]
            current_group = [remaining_word]
            current_text = remaining_word["text"]

        # Force break on close quote
        if word["text"].endswith("」"):
            force_break = True

        # Force break on open quote if length is greater than 1 (it will be stripped later if length is 1)
        elif word["text"].endswith("「") and len(word["text"].strip()) > 1:
            force_break = True

        # Break on optional delimiters
        elif any(word["text"].endswith(d) for d in delimiters):
            force_break = True

        # Break on audio events
        elif word["type"] == "audio_event":
            force_break = True


        # Break on hard max limit
        if len(current_text) > hard_max_lines:
            carryover_word_count = 0
            try:
                leading_space: str = current_text[:len(current_text) - len(current_text.lstrip())]
                trailing_space: str = current_text[len(current_text.rstrip()):]
                trimmed_current_text: str = current_text.strip()

                client: LLMProvider = AIToolFactory.get_llm_provider(model_name=model, 
                                                        system_prompt="Provide only the requested text without commentary or special formatting.")
                api_response = client.prompt(f"Split the following text into two lines at a logical point without modifications to the text or punctuation:\n{trimmed_current_text}")
                lines = api_response.strip().split('\n')
                line1_text = leading_space + lines[0].strip()
                line2_text = lines[1].strip() + trailing_space

                # Validate the API response
                if len(lines) == 2 and current_text.startswith(line1_text) and current_text.endswith(line2_text):
                    rebuilt_carryover_text = ""

                    # Reconstruct the second line from words to get an accurate word count
                    for word_obj in reversed(current_group):
                        rebuilt_carryover_text = word_obj["text"] + rebuilt_carryover_text
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
                for i in range(len(current_group) - 1, -1, -1):
                    word_text = current_group[i]["text"]
                    if carryover_chars + len(word_text) <= hard_max_carryover:
                        carryover_chars += len(word_text)
                        carryover_word_count += 1
                    else:
                        break
            
            if 0 < carryover_word_count < len(current_group):
                # Keep some words in group
                words_to_keep = current_group[:-carryover_word_count]
                if words_to_keep:
                    words_to_keep[-1]["text"] = words_to_keep[-1]["text"] + f"{CONTINUE_FLAG}"
                    merged_words.append(_create_merged_line(words_to_keep))

                # Start the new group with words carried over                 
                current_group = current_group[-carryover_word_count:]
                current_text = "".join(w["text"] for w in current_group)
            else:
                # The current word *alone* is longer than hard_max_lines.
                force_break = True

        # Break on soft delimiters if over soft max limit
        elif len(current_text) > soft_max_lines and any(word["text"].endswith(d) for d in soft_delimiters):
            force_break = True

        if force_break:
            if current_group:
                # Skip lines containing only a delimiter
                merged_w = _create_merged_line(current_group)
                if merged_w["text"] not in soft_delimiters + delimiters:
                    merged_words.append(merged_w)

                current_group = []

    # Add any remaining words
    if current_group:
        merged_w = _create_merged_line(current_group)
        if merged_w["text"] not in soft_delimiters + delimiters:
            merged_words.append(merged_w)

    return merged_words

def _create_merged_line(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Combines a group of word objects into a single line dictionary."""
    merged_text = "".join(w["text"] for w in group)

    # Remove Japanese quotation marks
    merged_text = merged_text.replace("「", "").replace("」", "")

    # Remove hyphens at line start
    if merged_text.startswith("-"):
            merged_text = merged_text[1:]

    # Trim white space
    merged_text = merged_text.strip()

    # Truncate repeated characters (5+ occurrences)
    merged_text = re.sub(r'(.{1,6}?)(\1{4,})', r'\1\1\1', merged_text)

    return {
        "text": merged_text,
        "start": group[0]["start"],
        "end": group[-1]["end"],
        "type": "merged_words",
        "speaker_id": group[0].get("speaker_id", "Unknown")
    }


