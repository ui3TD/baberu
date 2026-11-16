import json

from baberu.providers.transcription.base import TranscribedSegment, TranscribedWord, TranscriptionResult
from baberu.settings import AnomalySettings
from baberu.transcription import health
from conftest import run_cli

LIMITS = AnomalySettings()


def word(text, start, end, kind="word", confidence=None):
    return TranscribedWord(text=text, start=start, end=end, type=kind, confidence=confidence)


KANA = "かきくけこさしすせそたちつてと"
WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven"]


def steady(count, start=0.0, step=0.15, english=False):
    """Ordinary speech: varied tokens at a normal pace."""
    texts = WORDS if english else KANA
    step = step * 3 if english else step
    return [word(texts[i % len(texts)], start + i * step, start + (i + 1) * step) for i in range(count)]


def transcript(*words):
    return TranscriptionResult(segments=[TranscribedSegment(words=list(words))])


def kinds(t):
    return [a.kind for a in health.find_anomalies(t, LIMITS)]


def test_a_long_single_letter_is_a_smear():
    words = steady(20) + [word("ア", 3.0, 152.9)] + steady(20, start=152.9)

    found = health.find_anomalies(transcript(*words), LIMITS)

    assert [a.kind for a in found] == ["smear"]
    assert (found[0].start, found[0].end) == (3.0, 152.9)


def test_a_long_english_word_is_a_smear_but_a_slow_one_is_not():
    words = [word("Seamlessly.", 10.0, 148.0), word("Okay", 150.0, 151.5)]

    assert kinds(transcript(*steady(10, english=True), *words)) == ["smear"]


def test_zero_length_punctuation_is_normal():
    words = steady(5) + [word("。", 0.75, 0.75), word("、", 0.75, 0.75)] + steady(5, start=0.75)

    assert kinds(transcript(*words)) == []


def test_words_squeezed_into_no_time_are_a_collapse():
    squeezed = [word(kana, 3.0, 3.0) for kana in "あいうえおや"]

    assert kinds(transcript(*steady(20), *squeezed, *steady(20, start=3.0))) == ["collapse"]


def test_short_words_spread_out_are_not_a_collapse():
    quick = [word("は", 3.0 + i * 0.2, 3.0 + i * 0.2 + 0.001) for i in range(6)]

    assert "collapse" not in kinds(transcript(*steady(20), *quick))


def test_repeated_words_are_a_loop():
    loop = [word(" thank you", 3.0 + i, 3.5 + i) for i in range(6)]

    found = health.find_anomalies(transcript(*steady(5, english=True), *loop), LIMITS)

    assert [a.kind for a in found] == ["loop"]
    assert "x6" in found[0].detail


def test_digits_are_never_a_loop():
    digits = [word(d, i * 0.1, (i + 1) * 0.1) for i, d in enumerate("1000000")]

    assert kinds(transcript(*digits)) == []


def test_backwards_and_overlapping_times_are_disorder():
    words = [word("a", 1.0, 2.0), word("b", 1.5, 2.5), word("c", 3.0, 2.0)]

    assert [a.detail.split(" ", 1)[1] for a in health.find_anomalies(transcript(*words), LIMITS)] == [
        "overlaps the previous token", "ends before it starts"]


def test_constant_confidence_is_ignored():
    unsure = [word(WORDS[i], i, i + 0.2, confidence=-5.0) for i in range(5)]

    assert kinds(transcript(*unsure)) == []
    assert health.profile(transcript(*unsure))["logprob_informative"] is False


def test_low_confidence_runs_count_when_confidence_varies():
    sure = [word(WORDS[i], i, i + 0.2, confidence=-0.01) for i in range(5)]
    unsure = [word(WORDS[5 + i], 5 + i, 5.2 + i, confidence=-5.0) for i in range(3)]

    assert kinds(transcript(*sure, *unsure)) == ["low_confidence"]


def test_profile_describes_tokens():
    t = transcript(*steady(4), word("。", 0.6, 0.6), word(" ", 0.6, 0.7, "spacing"))

    p = health.profile(t)

    assert p["token_types"] == {"word": 5, "spacing": 1}
    assert p["punctuation_tokens"] == 1
    assert p["zero_duration_tokens"] == {"all": 1, "punctuation": 1}
    assert p["single_letter_tokens"] == 4


SCRIBE_JSON = {
    "language_code": "jpn", "language_probability": 1.0, "text": "アイ",
    "words": [
        {"text": "ア", "start": 1.0, "end": 151.0, "type": "word", "speaker_id": "speaker_0", "logprob": 0.0},
        *({"text": kana, "start": 151.0 + i * 0.1, "end": 151.1 + i * 0.1, "type": "word",
           "speaker_id": "speaker_0", "logprob": 0.0} for i, kana in enumerate("イドルのおはなしです")),
    ],
}


def test_diagnose_reports_and_writes_health(workspace, monkeypatch, capsys):
    source = workspace / "show.json"
    source.write_text(json.dumps(SCRIBE_JSON), encoding="utf-8")

    run_cli(monkeypatch, source, "--diagnose")

    output = capsys.readouterr()
    saved = json.loads((workspace / "show.baberu" / "health.json").read_text(encoding="utf-8"))
    assert saved["anomaly_counts"] == {"smear": 1}
    assert "smear" in output.err and "150.0s" in output.err
    assert not (workspace / "show.baberu" / "raw.ass").exists()


def test_every_run_saves_health(media, workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "show.mp4", "-c")

    saved = json.loads((workspace / "show.baberu" / "health.json").read_text(encoding="utf-8"))
    assert "profile" in saved and "repairs" in saved
