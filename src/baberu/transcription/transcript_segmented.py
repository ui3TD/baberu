import tempfile
from pathlib import Path
import logging

from pysubs2 import SSAFile

from LLMFactory.factory import AIToolFactory

from baberu.artifacts import Artifacts
from baberu.settings import MistimedSegmentSettings, ParsingSettings
from baberu.subtitling import sub_utils, sub_correction
from baberu.subtitling.types import LineRange
from baberu.tools import av_utils
from baberu.transcription import transcript_conversion, transcript_chunked

logger = logging.getLogger(__name__)

def find_segments(subtitles: SSAFile,
                  threshold_duration: float,
                  segs: MistimedSegmentSettings) -> list[LineRange]:
    """Finds runs of subtitles that are probably mistimed."""
    mistimed_indices: set[int] = sub_correction.find_mistimed_lines(subtitles, threshold_duration, segs)
    groups: list[list[int]] = sub_correction.find_mistimed_groups(mistimed_indices, segs.seg_min_lines)
    ranges = [LineRange(group[0], group[-1]) for group in groups]

    if ranges:
        _print_preview(ranges, subtitles, segs.seg_min_lines)
    else:
        logger.debug("No mistimed segments found.")

    return ranges

def transcribe_segments(subtitles: SSAFile,
                        ranges: list[LineRange],
                        audio_file: Path,
                        artifacts: Artifacts,
                        lang: str,
                        parsing: ParsingSettings,
                        transcription_model: str) -> SSAFile:
    """Re-transcribes the audio behind each range and replaces those subtitles."""
    if not ranges:
        return subtitles

    transcript_provider = AIToolFactory.get_transcription_provider(transcription_model)

    # Work backwards so replacing one range doesn't move the ones still to do.
    for line_range in sorted(ranges, key=lambda r: r.start, reverse=True):
        start_time_ms = subtitles.events[line_range.start].start
        end_time_ms = subtitles.events[line_range.end].end

        output_file: Path = artifacts.segment_transcript(audio_file, start_time_ms, end_time_ms)

        if output_file.exists():
            subtitles = sub_utils.splice(subtitles, line_range, sub_utils.load(output_file))
            continue

        start_time_sec = start_time_ms / 1000.0
        duration_sec = (end_time_ms - start_time_ms) / 1000.0

        with tempfile.NamedTemporaryFile(suffix='.oga', delete=False) as temp_audio:
            temp_audio_path: Path = Path(temp_audio.name)

        try:
            av_utils.cut_audio(audio_file, start_time_sec, duration_sec, temp_audio_path)
        except Exception as e:
            logger.error(f"Failed to cut audio for segment starting at {start_time_ms}ms: {e}")
            temp_audio_path.unlink(missing_ok=True)
            continue

        try:
            transcript, _ = transcript_chunked.transcribe_file(temp_audio_path, transcript_provider, lang)
            new_subtitles: SSAFile = transcript_conversion.convert_transcript_to_subs(transcript, parsing)
            new_subtitles.shift(ms=start_time_ms)
            sub_utils.write(new_subtitles, output_file)
            subtitles = sub_utils.splice(subtitles, line_range, new_subtitles)
        except Exception as e:
            logger.error(f"Error during transcription for segment starting at {start_time_ms}ms: {e}")
            raise
        finally:
            temp_audio_path.unlink(missing_ok=True)

    logger.info(f"Processed {len(ranges)} segments using two-pass transcription")
    return subtitles

def pad_segments(subtitles: SSAFile, ranges: list[LineRange]) -> list[LineRange]:
    """Widens each range by one line on both sides, as a buffer for cutting audio."""
    last_line = len(subtitles.events) - 1
    return [LineRange(max(r.start - 1, 0), min(r.end + 1, last_line)) for r in ranges]

def _print_preview(ranges: list[LineRange],
                   subtitles: SSAFile,
                   min_lines: int) -> None:
    """Prints a formatted preview of the identified mistimed segments."""
    print(f"Segments of over {min_lines} consecutive short subtitles found:")
    for line_range in ranges:
        end_time_sec = subtitles.events[line_range.end].end / 1000.0
        start_time_sec = subtitles.events[line_range.start].start / 1000.0
        start_min, start_sec = divmod(start_time_sec, 60)
        end_min, end_sec = divmod(end_time_sec, 60)
        duration_min, duration_sec = divmod(end_time_sec - start_time_sec, 60)

        print(f"  Timing: {int(start_min)}:{start_sec:05.2f} to {int(end_min)}:{end_sec:05.2f} | Duration: {int(duration_min)}:{duration_sec:05.2f} | {line_range.count} lines")
