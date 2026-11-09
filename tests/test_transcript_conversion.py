from LLMFactory.transcription.base import TranscribedSegment, TranscribedWord, TranscriptionResult

from baberu.constants import CONTINUE_FLAG
from baberu.transcription.transcript_conversion import convert_transcript_to_subs
from conftest import FakeLLM


def word(text: str, start: float, end: float, kind: str = "word") -> TranscribedWord:
    return TranscribedWord(text=text, start=start, end=end, type=kind)


def transcript(*words: TranscribedWord) -> TranscriptionResult:
    return TranscriptionResult(segments=[TranscribedSegment(words=list(words))])


def texts(subs) -> list[str]:
    return [e.text for e in subs.events]


def test_splits_on_delimiters_and_keeps_timing():
    subs = convert_transcript_to_subs(
        transcript(word("hello", 0.0, 1.0), word("!", 1.0, 1.2), word("bye", 2.0, 3.0)),
        delimiters=["!"],
    )

    assert texts(subs) == ["hello!", "bye"]
    assert (subs.events[1].start, subs.events[1].end) == (2000, 3000)


def test_splits_on_long_time_gaps():
    subs = convert_transcript_to_subs(
        transcript(word("one", 0.0, 1.0), word("two", 5.0, 6.0)),
        max_time_gap_sec=2.0,
    )

    assert texts(subs) == ["one", "two"]


def test_audio_events_are_dropped():
    subs = convert_transcript_to_subs(
        transcript(word("one", 0.0, 1.0), word("(laughs)", 1.0, 2.0, "audio_event"))
    )

    assert texts(subs) == ["one"]


def test_soft_delimiter_only_splits_past_soft_limit():
    subs = convert_transcript_to_subs(
        transcript(word("ab,", 0.0, 1.0), word("cdefgh,", 1.0, 2.0), word("ij", 2.0, 3.0)),
        soft_delimiters=[","],
        soft_max_lines=5,
    )

    assert texts(subs) == ["ab,cdefgh,", "ij"]


def test_hard_limit_carries_words_over_with_continue_flag():
    words = [word(f"{w} ", i, i + 1) for i, w in enumerate(["one", "two", "three", "four"])]

    subs = convert_transcript_to_subs(
        transcript(*words), hard_max_lines=12, hard_max_carryover=6, soft_max_lines=99
    )

    assert texts(subs)[0] == f"one two {CONTINUE_FLAG}"
    assert "".join(t.replace(CONTINUE_FLAG, "") for t in texts(subs)).split() == [
        "one", "two", "three", "four",
    ]


def test_llm_chooses_the_split_point(fake_llms):
    fake_llms["splitter"] = FakeLLM(lambda p: "one two three\nfour")
    words = [word(f"{w} ", i, i + 1) for i, w in enumerate(["one", "two", "three", "four"])]

    subs = convert_transcript_to_subs(
        transcript(*words), hard_max_lines=16, hard_max_carryover=6, soft_max_lines=99, model="splitter"
    )

    assert texts(subs)[0] == f"one two three {CONTINUE_FLAG}"


def test_bad_llm_split_falls_back_to_character_count(fake_llms):
    fake_llms["splitter"] = FakeLLM(lambda p: "not the original text")
    words = [word(f"{w} ", i, i + 1) for i, w in enumerate(["one", "two", "three", "four"])]

    subs = convert_transcript_to_subs(
        transcript(*words), hard_max_lines=12, hard_max_carryover=6, soft_max_lines=99, model="splitter"
    )

    assert texts(subs)[0] == f"one two {CONTINUE_FLAG}"
