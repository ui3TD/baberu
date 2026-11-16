import pytest

from baberu.job import Job
from baberu.providers.transcription.base import TranscribedSegment, TranscribedWord, TranscriptionResult
from baberu.settings import AnomalySettings, ParsingSettings
from baberu.subtitling.types import LineRange
from baberu.transcription import repair, transcript_conversion, transcript_segmented
from baberu.transcription.health import Anomaly, find_anomalies
from conftest import FakeTranscriber, make_subs

LIMITS = AnomalySettings()
KANA = "かきくけこさしすせそたちつてと"


def word(text, start, end):
    return TranscribedWord(text=text, start=start, end=end)


def steady(count, start=0.0, step=0.15):
    return [word(KANA[i % len(KANA)], start + i * step, start + (i + 1) * step) for i in range(count)]


def transcript(*words):
    return TranscriptionResult(segments=[TranscribedSegment(words=list(words))])


# --- smears ---

def test_smear_keeps_the_end_that_touches_the_next_word():
    t = transcript(*steady(20), word("ア", 1195.3, 1345.1), *steady(20, start=1345.1))

    repair.clamp_smears(t, LIMITS)

    smeared = t.segments[0].words[20]
    assert smeared.end == 1345.1
    assert 0 < smeared.end - smeared.start < 1.0
    assert find_anomalies(t, LIMITS) == []


def test_smear_keeps_the_start_when_that_is_closer():
    words = [*steady(20), word("Seamlessly.", 3.1, 141.0), word("Next", 365.0, 365.5)]
    t = transcript(*words)

    repair.clamp_smears(t, LIMITS)

    assert t.segments[0].words[20].start == 3.1
    assert t.segments[0].words[20].end < 10


def test_a_smear_no_longer_makes_a_line_last_minutes():
    t = transcript(*steady(20), word("ア", 3.0, 153.0), *steady(5, start=153.0))

    subs = transcript_conversion.convert_transcript_to_subs(t, ParsingSettings(parsing_model=""))

    assert max(e.end - e.start for e in subs.events) < 10_000


# --- collapses ---

def test_collapsed_words_spread_across_the_gap_before_the_next_word():
    squeezed = [word(k, 4.0, 4.0) for k in "あいうえおや"]
    t = transcript(*steady(20), *squeezed, *steady(5, start=4.0))  # 3.0s to 4.0s is free

    assert repair.spread_collapses(t, LIMITS) == 1

    spread = t.segments[0].words[20:26]
    assert spread[0].start == pytest.approx(3.0) and spread[-1].end == pytest.approx(4.0)
    assert all(w.end > w.start for w in spread)


def test_collapse_with_no_room_is_left_for_retranscribe():
    squeezed = [word(k, 3.0, 3.0) for k in "あいうえおや"]
    t = transcript(*steady(20), *squeezed, *steady(5, start=3.0))

    assert repair.spread_collapses(t, LIMITS) == 0
    assert [a.kind for a in repair.unresolved(t, LIMITS)] == ["collapse"]


def test_unresolved_keeps_smears_but_not_punctuation_held_too_long():
    t = transcript(*steady(20), word("。", 3.0, 14.0), word("ア", 15.0, 165.0), *steady(5, start=165.0))

    assert [a.kind for a in repair.unresolved(t, LIMITS)] == ["smear"]


# --- choosing lines to retranscribe ---

def test_anomalies_map_to_the_lines_they_touch_or_sit_between():
    subs = make_subs([(0, 1000, "a"), (1000, 2000, "b"), (10_000, 11_000, "c"), (11_000, 12_000, "d")])

    inside = transcript_segmented.anomaly_ranges(subs, [Anomaly("smear", 1.5, 1.8, 0, 0)])
    between = transcript_segmented.anomaly_ranges(subs, [Anomaly("smear", 3.0, 9.0, 0, 0)])

    assert inside == [LineRange(1, 1)]
    assert between == [LineRange(1, 2)]


def test_overlapping_and_touching_ranges_merge():
    ranges = [LineRange(5, 8), LineRange(0, 2), LineRange(3, 4), LineRange(7, 10), LineRange(20, 21)]

    assert transcript_segmented.merge_ranges(ranges) == [LineRange(0, 10), LineRange(20, 21)]


# --- accepting a second pass ---

@pytest.fixture
def second_pass(tmp_path, monkeypatch):
    """Audio cutting is skipped; the transcriber returns one 2-second line, "hello"."""
    transcriber = FakeTranscriber()
    monkeypatch.setattr("baberu.providers.factory.AIToolFactory.get_transcription_provider",
                        staticmethod(lambda model, options=None: transcriber))
    monkeypatch.setattr("baberu.tools.av_utils.cut_audio", lambda *a: None)
    job = Job.locate(tmp_path / "show.mp4", tmp_path, "ja", "en")

    def run(subs, line_range, **options):
        return transcript_segmented.transcribe_segments(
            subs, [line_range], tmp_path / "audio.m4a", job, "ja", ParsingSettings(parsing_model=""),
            "scribe_v3", limits=LIMITS, kinds=("smear", "collapse", "loop"), **options)

    run.job = job
    run.transcriber = transcriber
    return run


def test_a_better_second_pass_replaces_the_lines(second_pass):
    subs = make_subs([(0, 10, "x"), (10, 20, "y"), (20, 30, "z"), (5000, 6000, "after")])

    result = second_pass(subs, LineRange(0, 2))

    assert [e.text for e in result.events] == ["hello", "after"]
    assert second_pass.job.segment(0, 30, "scribe_v3").exists()


def test_a_second_pass_that_is_no_better_is_not_used(second_pass):
    subs = make_subs([(0, 1000, "fine"), (1000, 2000, "lines")])

    result = second_pass(subs, LineRange(0, 1))

    assert [e.text for e in result.events] == ["fine", "lines"]


def test_a_chosen_range_is_always_replaced(second_pass):
    subs = make_subs([(0, 1000, "fine"), (1000, 2000, "lines")])

    result = second_pass(subs, LineRange(0, 1), always_accept=True)

    assert [e.text for e in result.events] == ["hello"]


def test_a_saved_second_pass_is_reused_without_paying(second_pass):
    subs = lambda: make_subs([(0, 10, "x"), (10, 20, "y"), (20, 30, "z")])
    second_pass(subs(), LineRange(0, 2))

    second_pass(subs(), LineRange(0, 2))

    assert len(second_pass.transcriber.calls) == 1


def test_a_margin_gives_lead_in_and_drops_words_outside_the_range(second_pass):
    # The fake service always answers "hello" at 0-2 s of the clip it is sent.
    subs = make_subs([(0, 500, "before"), (1000, 1010, "x"), (1010, 1020, "y"), (1020, 3000, "z"), (6000, 7000, "after")])

    result = second_pass(subs, LineRange(1, 3), margin_sec=1.0)

    # The clip is cut from 0 s to 4 s; "hello" (middle at 1 s) falls inside lines 1-3, so it replaces them.
    assert second_pass.job.segment(0, 4000, "scribe_v3").exists()
    assert [e.text for e in result.events] == ["before", "hello", "after"]
