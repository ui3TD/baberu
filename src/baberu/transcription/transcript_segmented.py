import sys
import tempfile
from pathlib import Path
import logging

from pysubs2 import SSAFile

from baberu.providers.factory import AIToolFactory
from baberu.providers.transcription.base import TranscribedWord, TranscriptionResult

from baberu.job import Job
from baberu.settings import AnomalySettings, MistimedSegmentSettings, ParsingSettings
from baberu.subtitling import sub_utils, sub_correction
from baberu.subtitling.types import LineRange
from baberu.tools import av_utils
from baberu.transcription import health, repair, transcript_conversion, transcript_chunked
from baberu.transcription.health import Anomaly

logger = logging.getLogger(__name__)

def find_segments(subtitles: SSAFile,
                  threshold_duration: float,
                  segs: MistimedSegmentSettings) -> list[LineRange]:
    """Finds runs of subtitles that are probably mistimed."""
    mistimed_indices: set[int] = sub_correction.find_mistimed_lines(subtitles, threshold_duration, segs)
    groups: list[list[int]] = sub_correction.find_mistimed_groups(mistimed_indices, segs.seg_min_lines)
    ranges = [LineRange(group[0], group[-1]) for group in groups]
    health.count_repair("line_detector_ranges", len(ranges))

    if ranges:
        _print_preview(ranges, subtitles, segs.seg_min_lines)
    else:
        logger.debug("No mistimed segments found.")

    return ranges

def anomaly_ranges(subtitles: SSAFile, anomalies: list[Anomaly]) -> list[LineRange]:
    """The lines around each anomaly's time. An anomaly between lines takes the line on each side."""
    events = subtitles.events
    ranges = []
    for anomaly in anomalies:
        start_ms, end_ms = anomaly.start * 1000, anomaly.end * 1000
        lines = [i for i, event in enumerate(events) if event.start <= end_ms and event.end >= start_ms]
        if not lines:
            before = [i for i, event in enumerate(events) if event.end <= start_ms]
            after = [i for i, event in enumerate(events) if event.start >= end_ms]
            lines = before[-1:] + after[:1]
        if lines:
            ranges.append(LineRange(min(lines), max(lines)))
    return ranges


def merge_ranges(ranges: list[LineRange]) -> list[LineRange]:
    """Joins ranges that overlap or touch."""
    merged: list[LineRange] = []
    for line_range in sorted(ranges, key=lambda r: r.start):
        if merged and line_range.start <= merged[-1].end + 1:
            merged[-1] = LineRange(merged[-1].start, max(merged[-1].end, line_range.end))
        else:
            merged.append(line_range)
    return merged


def _problems(events, anomalies: list[Anomaly], short_line_sec: float) -> int:
    """How much is wrong in a stretch of subtitles: anomalies plus lines too short to read."""
    return len(anomalies) + sum(1 for e in events if (e.end - e.start) / 1000 <= short_line_sec)


def transcribe_segments(subtitles: SSAFile,
                        ranges: list[LineRange],
                        audio_file: Path,
                        job: Job,
                        lang: str,
                        parsing: ParsingSettings,
                        transcription_model: str,
                        options: dict | None = None,
                        split_cache: dict[str, str] | None = None,
                        limits: AnomalySettings | None = None,
                        kinds: tuple[str, ...] = (),
                        anomalies: list[Anomaly] | None = None,
                        short_line_sec: float = 0.1,
                        always_accept: bool = False,
                        margin_sec: float = 0.0) -> SSAFile:
    """Re-transcribes the audio behind each range, and keeps the result only if it is better.

    Better means not empty, and fewer problems than the lines it replaces:
    first-pass `anomalies` of the given `kinds` in that time, plus short lines.
    `always_accept` skips the comparison, for a range the user chose.
    `margin_sec` widens the audio cut on each side. Words whose middle falls in the
    margin are dropped.
    """
    if not ranges:
        return subtitles

    limits = limits or AnomalySettings()
    transcript_provider = None

    # Works from the last range to the first.
    for line_range in sorted(ranges, key=lambda r: r.start, reverse=True):
        start_time_ms = subtitles.events[line_range.start].start
        end_time_ms = subtitles.events[line_range.end].end
        cut_start_ms, cut_end_ms = _clip_bounds(start_time_ms, end_time_ms, margin_sec)

        output_file: Path = job.segment(cut_start_ms, cut_end_ms, transcription_model)

        if output_file.exists():
            transcript = transcript_conversion.load_transcript_json(output_file)
        else:
            if transcript_provider is None:
                transcript_provider = AIToolFactory.get_transcription_provider(transcription_model, options)
            transcript = _transcribe_audio(audio_file, cut_start_ms, cut_end_ms, transcript_provider, lang, output_file)
            if transcript is None:
                continue
        if (cut_start_ms, cut_end_ms) != (start_time_ms, end_time_ms):
            transcript = _within(transcript, (start_time_ms - cut_start_ms) / 1000, (end_time_ms - cut_start_ms) / 1000)

        new_subtitles = transcript_conversion.convert_transcript_to_subs(transcript, parsing, split_cache, limits)
        new_subtitles.shift(ms=cut_start_ms)

        old_anomalies = [a for a in anomalies or [] if a.kind in kinds
                         and a.start * 1000 <= end_time_ms and a.end * 1000 >= start_time_ms]
        new_anomalies = [a for a in repair.unresolved(transcript, limits) if a.kind in kinds]
        old = _problems(subtitles.events[line_range.start:line_range.end + 1], old_anomalies, short_line_sec)
        new = _problems(new_subtitles.events, new_anomalies, short_line_sec)
        where = f"lines {line_range.start + 1}-{line_range.end + 1}"

        if new_subtitles.events and (always_accept or new < old):
            subtitles = sub_utils.splice(subtitles, line_range, new_subtitles)
            health.count_repair("segments_retranscribed")
            logger.info(f"retranscribe: {where} replaced ({old} problems before, {new} after)")
        else:
            health.count_repair("segments_rejected")
            reason = "nothing came back" if not new_subtitles.events else f"{new} problems, not fewer than {old}"
            logger.info(f"retranscribe: kept {where} as they were: {reason}")

    logger.debug(f"Processed {len(ranges)} segments using two-pass transcription")
    return subtitles


def _clip_bounds(start_ms: int, end_ms: int, margin_sec: float) -> tuple[int, int]:
    """The audio to cut for a range: its time plus a margin on each side."""
    margin = max(int(margin_sec * 1000), 0)
    return max(start_ms - margin, 0), end_ms + margin


def _within(transcript: TranscriptionResult, start_sec: float, end_sec: float) -> TranscriptionResult:
    """Keeps the words whose middle falls between start and end, and words with no times."""
    def inside(word: TranscribedWord) -> bool:
        if word.start is None or word.end is None:
            return True
        return start_sec <= (word.start + word.end) / 2 <= end_sec

    segments = []
    for segment in transcript.segments:
        words = [w for w in segment.words if inside(w)]
        if any(w.type == "word" for w in words):
            segments.append(segment.model_copy(update={"words": words}))
    return transcript.model_copy(update={"segments": segments})


def _transcribe_audio(audio_file: Path, start_ms: int, end_ms: int, provider, lang: str, output_file: Path):
    """Transcribes one stretch of audio and saves the raw response. Returns None if the audio can't be cut."""
    with tempfile.NamedTemporaryFile(suffix='.oga', delete=False) as temp_audio:
        temp_audio_path: Path = Path(temp_audio.name)

    try:
        av_utils.cut_audio(audio_file, start_ms / 1000.0, (end_ms - start_ms) / 1000.0, temp_audio_path)
    except Exception as e:
        logger.error(f"Failed to cut audio for segment starting at {start_ms}ms: {e}")
        temp_audio_path.unlink(missing_ok=True)
        return None

    try:
        transcript, json_data = transcript_chunked.transcribe_file(temp_audio_path, provider, lang)
        transcript_conversion.write_transcript_json(json_data, output_file)
        return transcript
    except Exception as e:
        logger.error(f"Error during transcription for segment starting at {start_ms}ms: {e}")
        raise
    finally:
        temp_audio_path.unlink(missing_ok=True)

def pad_segments(subtitles: SSAFile, ranges: list[LineRange]) -> list[LineRange]:
    """Widens each range by one line on both sides, within the file's lines."""
    last_line = len(subtitles.events) - 1
    return [LineRange(max(r.start - 1, 0), min(r.end + 1, last_line)) for r in ranges]

def _print_preview(ranges: list[LineRange],
                   subtitles: SSAFile,
                   min_lines: int) -> None:
    """Prints a formatted preview of the identified mistimed segments."""
    print(f"Segments of over {min_lines} consecutive short subtitles found:", file=sys.stderr)
    for line_range in ranges:
        end_time_sec = subtitles.events[line_range.end].end / 1000.0
        start_time_sec = subtitles.events[line_range.start].start / 1000.0
        start_min, start_sec = divmod(start_time_sec, 60)
        end_min, end_sec = divmod(end_time_sec, 60)
        duration_min, duration_sec = divmod(end_time_sec - start_time_sec, 60)

        print(f"  Timing: {int(start_min)}:{start_sec:05.2f} to {int(end_min)}:{end_sec:05.2f} | Duration: {int(duration_min)}:{duration_sec:05.2f} | {line_range.count} lines", file=sys.stderr)
