"""Measures how healthy a transcript's timing and text are, without changing it.

Every limit is physical (seconds per character) and, where a file's own
behaviour matters, must also be a multiple away from that file's median.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import logging
import statistics
import unicodedata
from typing import Any, Iterator

from baberu.providers.transcription.base import TranscribedWord, TranscriptionResult
from baberu.settings import AnomalySettings

logger = logging.getLogger(__name__)

ZERO_SEC = 0.02
_GAP_BUCKETS = ((0.0, "overlap"), (ZERO_SEC, "touching"), (0.1, "<0.1s"), (0.5, "<0.5s"),
                (2.0, "<2s"), (10.0, "<10s"), (float("inf"), ">=10s"))

# How many times each repair changed something during this run. Reset by the pipeline.
repairs: Counter[str] = Counter()


def count_repair(name: str, times: int = 1) -> None:
    if times:
        repairs[name] += times


@dataclass
class Anomaly:
    kind: str           # smear, collapse, loop, disorder or low_confidence
    start: float        # seconds
    end: float
    first: int          # index of the first token involved, in token order
    last: int
    detail: str = ""
    unit: int = 0       # for a loop, how many tokens repeat


# --- Tokens ---

def letters(text: str) -> int:
    """Characters that are spoken: letters and digits, not punctuation or space."""
    return sum(1 for char in text if unicodedata.category(char)[0] in "LN")


def is_punctuation(word: TranscribedWord) -> bool:
    return word.type == "word" and bool(word.text.strip()) and letters(word.text) == 0


def tokens(transcript: TranscriptionResult) -> list[TranscribedWord]:
    """Every token in order, across segments."""
    return [word for segment in transcript.segments for word in segment.words]


def _duration(word: TranscribedWord) -> float | None:
    if word.start is None or word.end is None:
        return None
    return word.end - word.start


def _sec_per_char(word: TranscribedWord) -> float | None:
    duration = _duration(word)
    return None if duration is None else duration / max(1, letters(word.text))


def _percentiles(values: list[float], points=(50, 90, 99)) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    result = {f"p{p}": round(ordered[min(len(ordered) - 1, int(len(ordered) * p / 100))], 3) for p in points}
    result["max"] = round(ordered[-1], 3)
    return result


def _spoken(all_tokens: list[TranscribedWord]) -> Iterator[tuple[int, TranscribedWord]]:
    """Word tokens that carry letters, with their index."""
    for index, word in enumerate(all_tokens):
        if word.type == "word" and letters(word.text):
            yield index, word


def median_sec_per_char(all_tokens: list[TranscribedWord]) -> float | None:
    rates = [rate for _, word in _spoken(all_tokens) if (rate := _sec_per_char(word)) and rate > 0]
    return statistics.median(rates) if rates else None


def logprob_is_informative(all_tokens: list[TranscribedWord]) -> bool:
    """Whether token confidence varies at all. False when every token has the same value."""
    values = {word.confidence for word in all_tokens if word.confidence is not None}
    return len(values) > 1


# --- Profile ---

def profile(transcript: TranscriptionResult) -> dict[str, Any]:
    """Describes a transcript's tokens and timing."""
    all_tokens = tokens(transcript)
    spoken = [word for _, word in _spoken(all_tokens)]
    timed = [word for word in all_tokens if word.type != "spacing"]

    zero = [word for word in timed if (d := _duration(word)) is not None and d <= ZERO_SEC]
    longest_zero_run = run = 0
    for word in timed:
        duration = _duration(word)
        run = run + 1 if duration is not None and duration <= ZERO_SEC and not is_punctuation(word) else 0
        longest_zero_run = max(longest_zero_run, run)

    gaps: Counter[str] = Counter()
    for previous, word in zip(timed, timed[1:]):
        if previous.end is None or word.start is None:
            continue
        gap = word.start - previous.end
        gaps[next(label for limit, label in _GAP_BUCKETS if gap < limit)] += 1

    rates = []
    for segment in transcript.segments:
        words = [w for w in segment.words if w.type == "word" and w.start is not None and w.end is not None]
        if words and words[-1].end > words[0].start:
            rates.append(sum(letters(w.text) for w in words) / (words[-1].end - words[0].start))

    ends = [word.end for word in all_tokens if word.end is not None]
    return {
        "language": transcript.language,
        "duration_sec": round(max(ends), 2) if ends else 0.0,
        "segments": len(transcript.segments),
        "token_types": dict(Counter(word.type for word in all_tokens)),
        "punctuation_tokens": sum(1 for word in all_tokens if is_punctuation(word)),
        "letters_per_token": round(statistics.mean(letters(w.text) for w in spoken), 2) if spoken else 0.0,
        "single_letter_tokens": sum(1 for word in spoken if letters(word.text) == 1),
        "token_duration_sec": _percentiles([d for w in spoken if (d := _duration(w)) is not None]),
        "sec_per_letter": _percentiles([r for w in spoken if (r := _sec_per_char(w)) is not None]),
        "zero_duration_tokens": {"all": len(zero), "punctuation": sum(1 for w in zero if is_punctuation(w))},
        "longest_zero_duration_run": longest_zero_run,
        "gaps": {label: gaps[label] for _, label in _GAP_BUCKETS if gaps[label]},
        "speakers": len({word.speaker for word in all_tokens}),
        "logprob_informative": logprob_is_informative(all_tokens),
        "letters_per_sec_by_segment": _percentiles(rates, (10, 50, 90)),
    }


# --- Anomalies ---

def find_anomalies(transcript: TranscriptionResult, limits: AnomalySettings) -> list[Anomaly]:
    """Every timing or text problem found, in time order."""
    all_tokens = tokens(transcript)
    median = median_sec_per_char(all_tokens)
    found = [
        *_smears(all_tokens, limits, median),
        *_collapses(all_tokens, limits, median),
        *find_loops(all_tokens, limits),
        *_disorder(all_tokens, limits),
        *_low_confidence(all_tokens, limits),
    ]
    return sorted(found, key=lambda a: (a.start, a.first))


def _smears(all_tokens: list[TranscribedWord], limits: AnomalySettings, median: float | None) -> Iterator[Anomaly]:
    """A token held far longer than its letters could take to say."""
    for index, word in enumerate(all_tokens):
        if word.type != "word":
            continue
        duration, rate = _duration(word), _sec_per_char(word)
        if duration is None or duration < limits.smear_min_sec or rate <= limits.smear_sec_per_letter:
            continue
        if median and rate <= median * limits.median_multiple:
            continue
        yield Anomaly("smear", word.start, word.end, index, index, f"{word.text!r} lasts {duration:.1f}s")


def _collapses(all_tokens: list[TranscribedWord], limits: AnomalySettings, median: float | None) -> Iterator[Anomaly]:
    """A run of words squeezed into almost no time. Punctuation and spacing don't break a run."""
    run: list[int] = []

    def finish() -> Iterator[Anomaly]:
        if len(run) >= limits.collapse_min_tokens:
            first, last = all_tokens[run[0]], all_tokens[run[-1]]
            spoken_letters = sum(letters(all_tokens[i].text) for i in run)
            if last.end - first.start > spoken_letters * limits.collapse_sec_per_letter:
                return  # Short words, but spread out at a speakable pace.
            yield Anomaly("collapse", first.start, last.end, run[0], run[-1], f"{len(run)} words in {last.end - first.start:.2f}s")

    for index, word in enumerate(all_tokens):
        if word.type == "spacing" or is_punctuation(word):
            continue
        rate = _sec_per_char(word)
        fast = word.type == "word" and rate is not None and rate < limits.collapse_sec_per_letter \
            and (not median or rate < median / limits.median_multiple)
        if fast:
            run.append(index)
        else:
            yield from finish()
            run = []
    yield from finish()


def _loop_text(word: TranscribedWord) -> str:
    return word.text.strip().casefold()


def find_loops(all_tokens: list[TranscribedWord], limits: AnomalySettings) -> Iterator[Anomaly]:
    """The same few tokens over and over.

    A repeated unit with no letters, such as a number, is never a loop.
    """
    indices = [i for i, word in enumerate(all_tokens) if word.type == "word"]
    texts = [_loop_text(all_tokens[i]) for i in indices]
    position = 0
    while position < len(texts):
        for size in range(1, limits.loop_max_tokens + 1):
            unit = texts[position:position + size]
            if len(unit) < size:
                break
            joined = "".join(unit)
            if not any(unicodedata.category(c)[0] == "L" for c in joined):
                continue
            repeats = 1
            while texts[position + repeats * size:position + (repeats + 1) * size] == unit:
                repeats += 1
            if repeats >= limits.loop_min_repeats:
                first, last = indices[position], indices[position + repeats * size - 1]
                yield Anomaly("loop", all_tokens[first].start, all_tokens[last].end, first, last,
                              f"{joined!r} x{repeats}", unit=size)
                position += repeats * size - 1
                break
        position += 1


def _disorder(all_tokens: list[TranscribedWord], limits: AnomalySettings) -> Iterator[Anomaly]:
    """Missing, backwards or overlapping timestamps."""
    tolerance = limits.disorder_tolerance_sec
    previous: TranscribedWord | None = None
    for index, word in enumerate(all_tokens):
        if word.type == "spacing":
            continue
        start = word.start if word.start is not None else word.end
        end = word.end if word.end is not None else word.start
        problem = ""
        if word.start is None or word.end is None:
            problem = "missing timestamp"
        elif word.end < word.start - tolerance:
            problem = "ends before it starts"
        elif previous is not None and previous.start is not None and word.start < previous.start - tolerance:
            problem = "starts before the previous token"
        elif previous is not None and previous.end is not None and word.start < previous.end - tolerance:
            problem = "overlaps the previous token"
        if problem:
            yield Anomaly("disorder", start or 0.0, end or 0.0, index, index, f"{word.text!r} {problem}")
        if word.start is not None and word.end is not None:
            previous = word


def _low_confidence(all_tokens: list[TranscribedWord], limits: AnomalySettings) -> Iterator[Anomaly]:
    """Runs of unsure words, only where the model's confidence varies at all."""
    if not logprob_is_informative(all_tokens):
        return
    run: list[int] = []
    for index, word in list(_spoken(all_tokens)) + [(len(all_tokens), None)]:
        if word is not None and word.confidence is not None and word.confidence < limits.low_confidence_logprob:
            run.append(index)
            continue
        if len(run) >= limits.low_confidence_min_tokens:
            first, last = all_tokens[run[0]], all_tokens[run[-1]]
            yield Anomaly("low_confidence", first.start, last.end, run[0], run[-1], f"{len(run)} words")
        run = []


# --- Reports ---

def count_by_kind(anomalies: list[Anomaly]) -> dict[str, int]:
    return dict(Counter(anomaly.kind for anomaly in anomalies))


def report(transcript: TranscriptionResult, limits: AnomalySettings) -> dict[str, Any]:
    """Everything health.json holds: the profile, anomalies, and repairs made so far."""
    anomalies = find_anomalies(transcript, limits)
    return {
        "profile": profile(transcript),
        "anomaly_counts": count_by_kind(anomalies),
        "repairs": dict(repairs),
        "anomalies": [asdict(anomaly) for anomaly in anomalies],
    }


def _clock(seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}:{seconds:05.2f}"


def summary(health: dict[str, Any], show: int = 20) -> str:
    """A readable version of a report, for the terminal."""
    lines = []
    for key, value in health["profile"].items():
        lines.append(f"  {key:<28}{value}")
    counts = health["anomaly_counts"]
    lines.append("Anomalies: " + (", ".join(f"{kind} {count}" for kind, count in counts.items()) or "none"))
    for anomaly in sorted(health["anomalies"], key=lambda a: a["end"] - a["start"], reverse=True)[:show]:
        lines.append(f"  {anomaly['kind']:<15}{_clock(anomaly['start'])} to {_clock(anomaly['end'])}  {anomaly['detail']}")
    if health["repairs"]:
        lines.append("Repairs: " + ", ".join(f"{name} {count}" for name, count in health["repairs"].items()))
    return "\n".join(lines)
