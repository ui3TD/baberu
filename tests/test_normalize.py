import json
import random

from baberu.providers.transcription.base import TranscribedSegment, TranscribedWord, TranscriptionResult
from baberu.providers.transcription.elevenlabs import ScribeProvider
from baberu.settings import AnomalySettings, ParsingSettings
from baberu.transcription import transcript_conversion
from baberu.transcription.health import letters
from baberu.transcription.normalize import normalize

JAPANESE = ParsingSettings(parsing_model="")


def word(text, start, end, kind="word"):
    return TranscribedWord(text=text, start=start, end=end, type=kind)


def transcript(*words):
    return TranscriptionResult(segments=[TranscribedSegment(words=list(words))])


def texts(t):
    return [w.text for s in t.segments for w in s.words]


def convert(t, parsing=JAPANESE):
    return transcript_conversion.convert_transcript_to_subs(t, parsing)


# --- normalize ---

def test_closing_punctuation_joins_the_word_before_and_opening_the_word_after():
    t = transcript(word("「", 0, 0), word("今", 0, 0.2), word("日", 0.2, 0.4), word("。", 0.4, 0.4), word("」", 0.4, 0.4))

    assert texts(normalize(t)) == ["「今", "日。」"]


def test_spacing_joins_the_word_before_and_takes_no_time():
    t = transcript(word(" ", 0.0, 0.5, "spacing"), word("Hi", 0.5, 1.0), word(" ", 1.0, 3.0, "spacing"),
                   word("there", 3.0, 3.5))

    result = normalize(t).segments[0].words

    assert [(w.text, w.start, w.end) for w in result] == [("Hi ", 0.5, 1.0), ("there", 3.0, 3.5)]


def test_missing_times_take_a_known_neighbouring_time():
    t = transcript(word("a", 1.0, 2.0), word("b", None, None), word("c", None, 4.0), word("d", 5.0, None))

    result = normalize(t).segments[0].words

    assert [(w.start, w.end) for w in result] == [(1.0, 2.0), (2.0, 2.0), (4.0, 4.0), (5.0, 5.0)]


def test_unknown_types_are_dropped_not_fatal():
    t = transcript(word("a", 0, 1), word("(breath)", 1, 2, "breath"), word("(laughs)", 2, 3, "audio_event"))

    assert texts(normalize(t)) == ["a"]


def test_loops_keep_three_repeats():
    t = transcript(*(word(" thank you", i, i + 0.5) for i in range(8)))

    assert texts(normalize(t)) == [" thank you"] * 3


def test_normalize_leaves_its_input_alone():
    t = transcript(word("今", 0, 0.2), word("。", 0.2, 0.2))
    before = t.model_dump()

    normalize(t)

    assert t.model_dump() == before


# --- parsing ---

def scribe(words, segments=None):
    data = {"language_code": "eng", "language_probability": 1.0, "text": "", "words": words}
    if segments is not None:
        data["additional_formats"] = [{
            "requested_format": "segmented_json", "file_extension": "json", "content_type": "application/json",
            "is_base_64_encoded": False, "content": json.dumps({"segments": segments})}]
    return data


def test_segments_keep_the_spaces_segmented_json_drops():
    words = [{"text": "Thanks.", "start": 0.0, "end": 0.5, "type": "word", "speaker_id": "s0", "logprob": 0.0},
             {"text": " ", "start": 0.5, "end": 0.6, "type": "spacing", "speaker_id": "s0", "logprob": 0.0},
             {"text": "Okay.", "start": 0.6, "end": 1.0, "type": "word", "speaker_id": "s0", "logprob": 0.0},
             {"text": " ", "start": 1.0, "end": 1.1, "type": "spacing", "speaker_id": "s0", "logprob": 0.0},
             {"text": "Next", "start": 4.0, "end": 4.5, "type": "word", "speaker_id": "s1", "logprob": 0.0}]
    segments = [{"text": "Thanks.Okay.", "words": [words[0], words[2]]}, {"text": "Next", "words": [words[4]]}]

    result = ScribeProvider.parse(scribe(words, segments))

    assert ["".join(w.text for w in s.words) for s in result.segments] == ["Thanks. Okay. ", "Next"]
    assert result.language == "eng"
    assert result.segments[0].words[0].confidence == 0.0


def test_unknown_types_and_missing_times_parse():
    words = [{"text": "a", "start": None, "end": 1.0, "type": "word", "speaker_id": "s0", "logprob": -0.1},
             {"text": "(breath)", "start": 1.0, "end": 2.0, "type": "breath", "speaker_id": "s0", "logprob": -0.1}]

    result = ScribeProvider.parse(scribe(words))

    assert [(w.type, w.start) for w in result.segments[0].words] == [("word", None), ("breath", 1.0)]


# --- conversion ---

def test_numbers_survive_conversion():
    one_token = transcript(word("1000000", 0, 1))
    seven_tokens = transcript(*(word(d, i * 0.1, (i + 1) * 0.1) for i, d in enumerate("1000000")))

    assert [e.text for e in convert(one_token).events] == ["1000000"]
    assert [e.text for e in convert(seven_tokens).events] == ["1000000"]


def test_lines_of_only_punctuation_are_dropped():
    t = transcript(word("。」", 0, 0.1), word("...", 3.0, 3.1), word("はい", 6, 6.5))

    assert [e.text for e in convert(t).events] == ["はい"]


def test_a_line_starts_with_its_first_word_not_a_space():
    english = ParsingSettings(delimiters=["."], soft_delimiters=[], remove_text=[], parsing_model="")
    t = transcript(word("Done.", 0.0, 1.0), word(" ", 1.0, 5.0, "spacing"), word("Next", 5.0, 5.5))

    events = convert(t, english).events

    assert [(e.text, e.start) for e in events] == [("Done.", 0), ("Next", 5000)]


def test_converting_twice_gives_the_same_lines():
    t = transcript(*(word(k, i * 0.1, (i + 1) * 0.1) for i, k in enumerate("あいうえおかきくけこさしすせそ" * 5)))
    parsing = ParsingSettings(soft_max_chars=10, hard_max_chars=20, parsing_model="")

    first, second = convert(t, parsing), convert(t, parsing)

    assert [(e.text, e.start, e.end) for e in first.events] == [(e.text, e.start, e.end) for e in second.events]


def test_conversion_keeps_every_letter_and_never_makes_empty_timing():
    """Property: across random transcripts, no spoken letter is lost and no line has end <= start."""
    rng = random.Random(7)
    pool = list("あいうえおかきくけこ漢字ABCxyz") + ["。", "、", "？", "「", "」", " "]
    parsing = ParsingSettings(soft_max_chars=8, hard_max_chars=16, hard_max_carryover=6, max_time_gap_sec=1.0,
                              remove_text=["「", "」"], parsing_model="")
    limits = AnomalySettings(loop_min_repeats=10_000)

    for _ in range(200):
        words, time = [], 0.0
        for _ in range(rng.randint(1, 60)):
            text = rng.choice(pool)
            duration = rng.choice([0.0, 0.0, 0.05, 0.2, 1.5])
            kind = "spacing" if text == " " else "word"
            words.append(word(text, time, time + duration, kind))
            time += duration + rng.choice([0.0, 0.0, 0.1, 2.5])
        t = transcript(*words)

        subs = transcript_conversion.convert_transcript_to_subs(t, parsing, limits=limits)

        spoken_in = "".join(c for w in words for c in w.text if letters(c))
        spoken_out = "".join(c for e in subs.events for c in e.text.replace("%%CONT%%", "") if letters(c))
        assert spoken_out == spoken_in
        assert all(e.end > e.start for e in subs.events)


# --- words cut by a pause ---

def chars(text, start, step=0.1):
    """One token per character, the way Scribe times Japanese."""
    return [word(c, start + i * step, start + (i + 1) * step) for i, c in enumerate(text)]


def lines(t, parsing=JAPANESE):
    return [e.text for e in convert(t, parsing).events]


def test_a_drawn_out_last_syllable_stays_on_its_line():
    t = transcript(*chars("せーの、乾", 0.0), *chars("杯。", 3.3))

    assert lines(t) == ["せーの、乾杯。"]


def test_a_tail_split_into_another_segment_is_joined_too():
    t = TranscriptionResult(segments=[TranscribedSegment(words=chars("舞香が優勝で", 0.0)),
                                      TranscribedSegment(words=chars("す。", 3.4))])

    assert lines(t) == ["舞香が優勝です。"]


def test_brackets_cut_by_a_pause_are_joined():
    t = transcript(*chars("下に下（聞", 0.0), *chars("き取り不能）。", 3.3))

    assert lines(t) == ["下に下（聞き取り不能）。"]


def test_a_longer_line_after_a_pause_stays_separate():
    # A one-letter line before a pause stays a line of its own.
    t = transcript(*chars("特", 0.0), *chars("別二番。", 3.0))

    assert lines(t) == ["特", "別二番。"]


def test_a_pause_after_punctuation_still_starts_a_line():
    t = transcript(*chars("はい、", 0.0), *chars("す。", 3.0))

    assert lines(t) == ["はい、", "す。"]


def test_a_very_long_pause_is_never_joined():
    t = transcript(*chars("乾", 0.0), *chars("杯。", 6.0))

    assert lines(t) == ["乾", "杯。"]


def test_english_words_cut_by_a_pause_never_join():
    english = ParsingSettings(delimiters=["."], soft_delimiters=[" "], remove_text=[], parsing_model="")
    t = transcript(word("So", 0.0, 0.3), word(" ", 0.3, 0.3, "spacing"), word("I", 0.3, 0.5),
                   word(" ", 0.5, 0.5, "spacing"), word("think", 0.5, 0.9), word(" ", 0.9, 0.9, "spacing"),
                   word("no.", 3.5, 3.9))

    assert lines(t, english) == ["So I think", "no."]


# --- second-pass clip margins ---

def test_clip_margin_widens_both_sides_but_not_before_zero():
    from baberu.transcription.transcript_segmented import _clip_bounds

    assert _clip_bounds(1300, 3000, 1.0) == (300, 4000)
    assert _clip_bounds(400, 3000, 1.0) == (0, 4000)
    assert _clip_bounds(1300, 3000, 0.0) == (1300, 3000)


def test_words_in_the_margin_are_dropped():
    from baberu.transcription.transcript_segmented import _within

    t = TranscriptionResult(segments=[
        TranscribedSegment(words=[word("前", 0.2, 0.6), word("優", 0.9, 1.1), word("勝", 1.1, 1.3)]),
        TranscribedSegment(words=[word("後", 3.2, 3.6)]),
    ])

    assert texts(_within(t, 1.0, 3.0)) == ["優", "勝"]   # 優's middle (1.0) is inside; 前 and 後 aren't
