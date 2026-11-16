"""Records of which model and options made each piece of work, and when old work is reused."""
import json

import pytest

from baberu.job import Job
from baberu.providers.transcription.base import TranscribedSegment, TranscribedWord, TranscriptionResult
from baberu.settings import ParsingSettings
from baberu.setup import config_setup
from baberu.transcription import transcript_conversion
from conftest import FakeLLM, run_cli


def fails(monkeypatch, capsys, *argv) -> str:
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, *argv)
    assert exit_info.value.code == 1
    return capsys.readouterr().err


def test_transcript_records_its_model_and_options(media, workspace, monkeypatch):
    run_cli(monkeypatch, workspace / "show.mp4", "-c")

    manifest = json.loads((workspace / "show.baberu" / "manifest.json").read_text(encoding="utf-8"))
    origin = manifest["made_from"]["transcript.json"]
    assert origin["model"] == "whisper-1"
    assert origin["source"].startswith("audio.")
    assert origin["options"] == {}


def test_changed_transcription_model_stops_before_reusing(media, workspace, monkeypatch, capsys):
    run_cli(monkeypatch, workspace / "show.mp4", "-c")
    capsys.readouterr()
    config_setup.load_config()["transcription"]["elevenlabs_model"] = "scribe_v3"

    error = fails(monkeypatch, capsys, workspace / "show.mp4", "-c")

    assert "--redo transcribe" in error and "scribe_v3" in error
    assert len(media.calls) == 1


def test_redone_transcript_is_converted_again(media, workspace, monkeypatch, capsys):
    run_cli(monkeypatch, workspace / "show.mp4", "-c")
    capsys.readouterr()

    run_cli(monkeypatch, workspace / "show.mp4", "-c", "--redo", "transcribe", "--dry-run")

    plan = capsys.readouterr().err
    assert "convert        run" in plan


def test_second_pass_can_use_another_model(media, workspace, monkeypatch):
    config_setup.load_config()["transcription"]["retranscribe_model"] = "scribe_v3"

    run_cli(monkeypatch, workspace / "show.mp4", "-r")

    manifest = json.loads((workspace / "show.baberu" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["made_from"]["transcript.json"]["model"] == "whisper-1"
    assert manifest["made_from"]["2pass.ass"]["model"] == "scribe_v3"


def test_segment_cache_is_named_by_model(tmp_path):
    job = Job.locate(tmp_path / "show.mp4", tmp_path, "ja", "en")

    assert job.segment(0, 1000, "scribe_v1") != job.segment(0, 1000, "scribe_v3")
    assert "scribe_v3" in job.segment(0, 1000, "scribe_v3").name


def test_split_answers_are_reused(fake_llms):
    fake_llms["splitter"] = FakeLLM(lambda p: "one two three\nfour")
    words = [TranscribedWord(text=f"{w} ", start=i, end=i + 1) for i, w in enumerate(["one", "two", "three", "four"])]
    transcript = TranscriptionResult(segments=[TranscribedSegment(words=words)])
    parsing = ParsingSettings(delimiters=[], soft_delimiters=[], remove_text=[], soft_max_chars=99,
                              hard_max_chars=16, hard_max_carryover=6, parsing_model="splitter")
    splits: dict[str, str] = {}

    first = transcript_conversion.convert_transcript_to_subs(transcript, parsing, splits)
    second = transcript_conversion.convert_transcript_to_subs(transcript, parsing, splits)

    assert len(fake_llms["splitter"].calls) == 1
    assert [e.text for e in first.events] == [e.text for e in second.events]
