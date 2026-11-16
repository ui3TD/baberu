"""Puts every service's tokens into one shape before they become subtitle lines.

Services and languages disagree on small things: Japanese punctuation comes
as its own token while English punctuation is part of the word; spacing is a
token in some responses and missing in others; a timestamp can be null. After
normalize, every token is a timed word that carries its own punctuation and
any space after it.
"""
from __future__ import annotations

import logging
import unicodedata

from baberu.providers.transcription.base import TranscribedSegment, TranscribedWord, TranscriptionResult
from baberu.settings import AnomalySettings
from baberu.transcription import health

logger = logging.getLogger(__name__)

SPEECH_TYPES = {"word", "spacing"}
KNOWN_TYPES = SPEECH_TYPES | {"audio_event"}
LOOP_KEEP = 3


def normalize(transcript: TranscriptionResult, limits: AnomalySettings | None = None) -> TranscriptionResult:
    """Returns a cleaned copy; the transcript passed in is left alone."""
    limits = limits or AnomalySettings()
    result = transcript.model_copy(deep=True)

    _drop_non_speech(result)
    _fill_missing_times(result)
    for segment in result.segments:
        segment.words = _attach_punctuation_and_spacing(segment.words)
    _truncate_loops(result, limits)
    result.segments = [segment for segment in result.segments if segment.words]
    return result


def _drop_non_speech(transcript: TranscriptionResult) -> None:
    unknown = {w.type for s in transcript.segments for w in s.words} - KNOWN_TYPES
    if unknown:
        logger.info(f"Ignoring token types this version doesn't know: {', '.join(sorted(unknown))}")
    for segment in transcript.segments:
        segment.words = [word for word in segment.words if word.type in SPEECH_TYPES]


def _fill_missing_times(transcript: TranscriptionResult) -> None:
    """A missing time becomes the token's other time, or the end of the token before.

    The token is left with no length.
    """
    missing = 0
    last_end = 0.0
    for word in health.tokens(transcript):
        if word.start is None or word.end is None:
            missing += 1
            known = word.start if word.start is not None else word.end
            word.start = word.end = known if known is not None else last_end
        last_end = max(last_end, word.end)
    if missing:
        health.count_repair("missing_times_filled", missing)


def _opens(text: str) -> bool:
    """Whether the text is only opening brackets and quotes."""
    stripped = text.strip()
    return bool(stripped) and all(unicodedata.category(char) in ("Ps", "Pi") for char in stripped)


def _attach_punctuation_and_spacing(words: list[TranscribedWord]) -> list[TranscribedWord]:
    result: list[TranscribedWord] = []
    prefix = ""
    for word in words:
        if word.type == "spacing":
            if result and not prefix:
                result[-1].text += word.text
            continue
        if health.is_punctuation(word):
            if _opens(word.text):
                prefix += word.text
            elif result:
                result[-1].text += word.text
            # Closing punctuation with no word before it in the segment is dropped.
            continue
        if prefix:
            word.text = prefix + word.text.lstrip()
            prefix = ""
        result.append(word)

    if prefix:
        if result:
            result[-1].text += prefix
        else:
            logger.debug(f"Dropping a segment of punctuation only: {prefix!r}")
    return result


def _truncate_loops(transcript: TranscriptionResult, limits: AnomalySettings) -> None:
    """Keeps the first few repeats of a loop. Repeated numbers are left untouched."""
    for segment in transcript.segments:
        drop: set[int] = set()
        for anomaly in health.find_loops(segment.words, limits):
            drop.update(range(anomaly.first + anomaly.unit * LOOP_KEEP, anomaly.last + 1))
        if drop:
            health.count_repair("loops_truncated")
            segment.words = [word for index, word in enumerate(segment.words) if index not in drop]
