import tempfile
from pathlib import Path
import logging

from pysubs2 import SSAFile

from baberu.subtitling import sub_utils, sub_correction
from baberu.tools import av_utils
from baberu.LLMFactory.factory import AIToolFactory
from baberu.LLMFactory.transcription.base import TranscriptionResult
from baberu.transcription import transcript_conversion, transcript_chunked

logger = logging.getLogger(__name__)

def find_segments(subtitles: SSAFile, 
                  threshold_duration: float,
                  grp_min_lines: int = 4,
                  grp_backtrace_limit: int = 20,
                  grp_foretrace_limit: int = 5,
                  grp_min_delay_sec: float = 10,
                  grp_max_gap: int = 4) -> list[list[int]]:
    """Identifies and groups segments of potentially mistimed subtitles.

    Args:
        subtitles (SSAFile): The subtitle file to analyze.
        threshold_duration (float): Duration threshold for identifying mistimed lines.
        grp_min_lines (int): Minimum number of consecutive lines to form a group.
        grp_backtrace_limit (int): How many previous lines to check for grouping.
        grp_foretrace_limit (int): How many subsequent lines to check for grouping.
        grp_min_delay_sec (float): Minimum delay between lines to be a group boundary.
        grp_max_gap (int): Maximum non-mistimed lines allowed within a group.

    Returns:
        list[list[int]]: A list of segments, where each segment is a list of
            subtitle indices.
    """
    # Find mistimed subtitles and group them
    mistimed_indices: set[int] = sub_correction.find_mistimed_lines(subtitles, threshold_duration, grp_min_lines, grp_backtrace_limit, grp_foretrace_limit, grp_min_delay_sec, grp_max_gap)
    segments: list[list[int]] = sub_correction.find_mistimed_groups(mistimed_indices, grp_min_lines)

    # Output timings for segments
    if segments:
        _print_preview(segments, subtitles, grp_min_lines)
    else:
        logger.debug(f"No mistimed segments found.")
        return segments
    
    return segments

def transcribe_segments(subtitles: SSAFile, 
                        segments: list[list[int]], 
                        audio_file: Path, 
                        lang: str,
                        delimiters: list[str],
                        soft_delimiters: list[str],
                        soft_max_lines: int,
                        hard_max_lines: int,
                        hard_max_carryover: int,
                        transcription_model: str,
                        parsing_model: str) -> SSAFile:
    """Re-transcribes specified segments of audio and replaces the original subtitles.

    Args:
        subtitles (SSAFile): The original subtitle file.
        segments (list[list[int]]): A list of segments to be re-transcribed.
        audio_file (Path): Path to the full audio file.
        num_speakers (int): The number of speakers for transcription.
        lang (str): The language of the audio.
        delimiters (str | list[str]): Delimiters for sentence splitting.
        soft_delimiters (str | list[str]): Soft delimiters for sentence splitting.
        audio_events (bool): Flag to include audio events in transcription.
        soft_max_lines (int): Preferred maximum lines per subtitle event.
        hard_max_lines (int): Absolute maximum lines per subtitle event.
        hard_max_carryover (int): Max characters to carry over to the next line.
        transcription_model (str): The model to use for transcription.
        parsing_model (str): The model to use for parsing transcription results.

    Returns:
        SSAFile: The modified subtitle file with re-transcribed segments.
    """
    if not segments:
        return subtitles

    # Fetch transcription providers
    transcript_provider_type = AIToolFactory.get_transcription_provider_type(transcription_model)
    transcript_provider = AIToolFactory.get_transcription_provider(transcription_model)

    # Process segments in reverse order to avoid index shifting issues when splicing
    for segment in sorted(segments, key=lambda g: g[0], reverse=True):
        segment = sorted(segment)

        # Calculate start and end times for the segment
        start_time_ms = subtitles.events[min(segment)].start
        end_time_ms = subtitles.events[max(segment)].end
        
        output_file: Path = audio_file.with_suffix(f".{start_time_ms}-{end_time_ms}.ass")

        # If a cached transcription for this exact segment exists, use it
        if output_file.exists():
            new_subtitles = SSAFile.load(output_file)
            subtitles = sub_utils.splice(subtitles, segment, new_subtitles)
            continue  # Proceed to the next segment

        # Convert times to seconds for ffmpeg
        start_time_sec = start_time_ms / 1000.0
        end_time_sec = end_time_ms / 1000.0
        duration_sec = end_time_sec - start_time_sec
        
        # Create a temporary file for the audio segment
        with tempfile.NamedTemporaryFile(suffix='.oga', delete=False) as temp_audio:
            temp_audio_path: Path = Path(temp_audio.name)

        # Cut the audio segment using ffmpeg
        try:
            av_utils.cut_audio(audio_file, start_time_sec, duration_sec, temp_audio_path)
        except Exception as e:
            logger.error(f"Failed to cut audio for segment starting at {start_time_ms}ms: {e}")
            if temp_audio_path.exists():
                Path.unlink(temp_audio_path)
            continue # Skip this segment and proceed to the next

        max_size = transcript_provider.max_size_bytes
        file_size = temp_audio_path.stat().st_size

        # Transcribe the audio segment and splice the results
        try:
            if max_size and file_size > max_size:
                transcript = transcript_chunked.transcribe_in_chunks(temp_audio_path, transcript_provider, lang=lang)
            else:
                json_data = transcript_provider.transcribe(temp_audio_path, lang=lang)
                transcript: TranscriptionResult = transcript_provider_type.parse(json_data)
            new_subtitles: SSAFile = transcript_conversion.convert_transcript_to_subs(
                transcript, delimiters, soft_delimiters, soft_max_lines, 
                hard_max_lines, hard_max_carryover, parsing_model
            )
            
            new_subtitles.shift(ms=start_time_ms)
            sub_utils.write(new_subtitles, output_file)  # Cache the result

            subtitles = sub_utils.splice(subtitles, segment, new_subtitles)
        except Exception as e:
            logger.error(f"Error during transcription for segment starting at {start_time_ms}ms: {e}")
            raise # Stop the entire process on a critical error
        finally:
            # Ensure the temporary audio file is always deleted
            if temp_audio_path.exists():
                Path.unlink(temp_audio_path)

    logger.info(f"Processed {len(segments)} segments using two-pass transcription")
    return subtitles

def pad_segments(subtitles: SSAFile, segments: list[list[int]]) -> list[list[int]]:
    """Expands each subtitle segment to include one preceding and one succeeding line.

    This padding provides a buffer for audio extraction.

    Args:
        subtitles (SSAFile): The subtitle file, used to check boundaries.
        segments (list[list[int]]): The list of segments to pad.

    Returns:
        list[list[int]]: The list of padded segments.
    """
    padded_segments = []
    
    for segment in sorted(segments, key=lambda g: g[0], reverse=True):
        # Expand segment for buffer
        modified_segment = sorted(segment)
        if modified_segment[0] > 0:
            modified_segment = [modified_segment[0] - 1] + modified_segment
        if modified_segment[-1] < (len(subtitles.events) - 1):
            modified_segment = modified_segment + [modified_segment[-1] + 1]
        
        padded_segments.append(modified_segment)
           
    return padded_segments

def _print_preview(segments: list[list[int]], 
                   subtitles: SSAFile,
                   min_lines: int) -> None:
    """Prints a formatted preview of the identified mistimed segments."""
    logger.info(f"Segments of over {min_lines} consecutive short subtitles found:")
    for segment in segments:
        end_time_sec = subtitles.events[segment[-1]].end / 1000.0
        start_time_sec = subtitles.events[segment[0]].start / 1000.0
        # Convert to minutes and seconds
        start_min, start_sec = divmod(start_time_sec, 60)
        end_min, end_sec = divmod(end_time_sec, 60)
        duration_sec = end_time_sec - start_time_sec
        duration_min, duration_sec = divmod(duration_sec, 60)
        
        logger.info(f"  Timing: {int(start_min)}:{start_sec:05.2f} to {int(end_min)}:{end_sec:05.2f} | Duration: {int(duration_min)}:{duration_sec:05.2f} | {len(segment)} lines")
    return