"""Fixes token timing where an anomaly says it's wrong. Free, deterministic, and checked.

Runs on a normalized transcript, before it becomes lines. Each repair works
only on tokens an anomaly points at, and refuses when its result would still
be implausible; what it refuses is left for retranscribe.
"""
from __future__ import annotations

import logging

from baberu.providers.transcription.base import TranscribedWord, TranscriptionResult
from baberu.settings import AnomalySettings
from baberu.transcription import health
from baberu.transcription.normalize import normalize

logger = logging.getLogger(__name__)

# Speech this many times slower than the file's median is still plausible after a clamp.
_CLAMP_MULTIPLE = 2.0
_FALLBACK_SEC_PER_LETTER = 0.15


def repair(transcript: TranscriptionResult, limits: AnomalySettings) -> TranscriptionResult:
    """Clamps smears, then spreads out collapses. Changes the transcript passed in."""
    clamp_smears(transcript, limits)
    spread_collapses(transcript, limits)
    return transcript


def unresolved(transcript: TranscriptionResult, limits: AnomalySettings) -> list[health.Anomaly]:
    """Anomalies the free repairs can't settle, which retranscribe may try.

    Every smear stays on the list, clamped or not. Anything else counts only
    if it is still there after normalizing and repairing.
    """
    counted = health.repairs.copy()  # Repair counts are restored after the look ahead.
    normalized = normalize(transcript, limits)
    smears = [a for a in health.find_anomalies(normalized, limits) if a.kind == "smear"]
    repaired = repair(normalized.model_copy(deep=True), limits)
    rest = [a for a in health.find_anomalies(repaired, limits) if a.kind != "smear"]
    health.repairs.clear()
    health.repairs.update(counted)
    return sorted(smears + rest, key=lambda a: a.start)


def _plausible_duration(word: TranscribedWord, median: float | None, limits: AnomalySettings) -> float:
    per_letter = min((median or _FALLBACK_SEC_PER_LETTER) * _CLAMP_MULTIPLE, limits.smear_sec_per_letter)
    return max(1, health.letters(word.text)) * per_letter


def clamp_smears(transcript: TranscriptionResult, limits: AnomalySettings) -> int:
    """Shortens each smeared token, keeping the edge closest to a neighbour.

    The end is kept on a tie. Returns how many tokens were shortened.
    """
    tokens = health.tokens(transcript)
    median = health.median_sec_per_char(tokens)
    smears = [a for a in health.find_anomalies(transcript, limits) if a.kind == "smear"]

    for anomaly in smears:
        word = tokens[anomaly.first]
        before = tokens[anomaly.first - 1] if anomaly.first > 0 else None
        after = tokens[anomaly.first + 1] if anomaly.first + 1 < len(tokens) else None
        gap_before = word.start - before.end if before else float("inf")
        gap_after = after.start - word.end if after else float("inf")
        duration = _plausible_duration(word, median, limits)

        if gap_after <= gap_before:
            word.start = word.end - duration
        else:
            word.end = word.start + duration
        logger.debug(f"Clamped {word.text!r} from {anomaly.end - anomaly.start:.1f}s to {duration:.2f}s")

    health.count_repair("smears_clamped", len(smears))
    return len(smears)


def spread_collapses(transcript: TranscriptionResult, limits: AnomalySettings) -> int:
    """Spreads words squeezed into no time across the gap around them, by letter count.

    The words are placed between the end of the word before and the start of
    the word after. If that still makes them too fast to say, or so slow they
    would fill a long silence, nothing changes and retranscribe can try.
    """
    tokens = health.tokens(transcript)
    collapses = [a for a in health.find_anomalies(transcript, limits) if a.kind == "collapse"]
    spread = 0

    for anomaly in collapses:
        run = tokens[anomaly.first:anomaly.last + 1]
        left = tokens[anomaly.first - 1].end if anomaly.first > 0 else run[0].start
        right = tokens[anomaly.last + 1].start if anomaly.last + 1 < len(tokens) else run[-1].end
        total_letters = sum(max(1, health.letters(w.text)) for w in run)
        window = right - left

        too_fast = window < total_letters / limits.max_letters_per_sec
        too_slow = window > total_letters * limits.smear_sec_per_letter
        if too_fast or too_slow:
            logger.debug(f"Left {len(run)} collapsed words at {anomaly.start:.2f}s: "
                          f"{window:.2f}s is too {'short' if too_fast else 'long'} for {total_letters} letters")
            continue

        time = left
        for word in run:
            word.start = time
            time += window * max(1, health.letters(word.text)) / total_letters
            word.end = time
        spread += 1

    health.count_repair("collapses_spread", spread)
    health.count_repair("collapses_left", len(collapses) - spread)
    return spread
