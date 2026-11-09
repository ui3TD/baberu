"""Shared test helpers. Nothing here touches the network or ffmpeg."""
import importlib.resources
import re
import sys
from collections.abc import Callable

import pytest
import yaml
from pysubs2 import SSAEvent, SSAFile

from LLMFactory.factory import AIToolFactory
from LLMFactory.transcription.base import TranscriptionProvider
from LLMFactory.transcription.openai import WhisperProvider

BATCH_MARKER = "Maintain exact entry count and order:\n\n"
ENTRY = re.compile(r"^\d+\. (.*)$")


def make_subs(rows: list[tuple[int, int, str]]) -> SSAFile:
    """Builds subtitles from (start_ms, end_ms, text) rows."""
    subs = SSAFile()
    subs.events = [SSAEvent(start=s, end=e, text=t, style="Default") for s, e, t in rows]
    return subs


def numbered_subs(count: int) -> SSAFile:
    return make_subs([(i * 2000, i * 2000 + 1500, f"line{i}") for i in range(count)])


def batch_entries(prompt: str) -> list[str]:
    """Pulls the source entries out of a translation prompt."""
    body = prompt.split(BATCH_MARKER)[-1]
    return [ENTRY.match(line).group(1) for line in body.split("\n") if ENTRY.match(line)]


def translate_all(prompt: str) -> str:
    """A well-behaved translator: one 'T:' line per source entry."""
    return "\n".join(f"T:{entry}" for entry in batch_entries(prompt))


class FakeLLM:
    model = "fake-model"

    def __init__(self, responder: Callable[[str], str]):
        self.responder = responder
        self.calls: list[str] = []
        self.kwargs: list[dict] = []

    def prompt(self, user_prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        self.calls.append(user_prompt)
        self.kwargs.append(kwargs)
        return self.responder(user_prompt)


@pytest.fixture
def fake_llms(monkeypatch):
    """Routes AIToolFactory to FakeLLMs. Set entries by model name before use."""
    llms: dict[str, FakeLLM] = {}

    def get_llm_provider(model_name: str, system_prompt: str = "", **options):
        return llms[model_name]

    monkeypatch.setattr(AIToolFactory, "get_llm_provider", staticmethod(get_llm_provider))
    return llms


WHISPER_JSON = {
    "text": "hello",
    "segments": [{"text": "hello", "start": 0.0, "end": 2.0}],
    "words": [{"word": "hello", "start": 0.0, "end": 2.0}],
}


class FakeTranscriber(TranscriptionProvider):
    parse = staticmethod(WhisperProvider.parse)
    validate = staticmethod(WhisperProvider.validate)

    def __init__(self):
        super().__init__("key", "fake")
        self.calls: list = []

    def transcribe(self, audio_file, **kwargs):
        self.calls.append(audio_file)
        return WHISPER_JSON


@pytest.fixture
def workspace(tmp_path, monkeypatch, fake_llms):
    """A folder holding video.ass and ctx.txt, with baberu's config and logging stubbed."""
    text = importlib.resources.files("baberu.defaults").joinpath("default_config.yaml").read_text("utf-8")
    config = yaml.safe_load(text)
    config["translation"].update(confirm_auto_context=False, websearch_model="main", default_model="main", fallback_model="main")
    config["logging"]["log_to_file"] = False
    config["parsing"]["parsing_model"] = ""
    monkeypatch.setattr("baberu.setup.config_setup.load_config", lambda *a: config)
    monkeypatch.setattr("baberu.setup.logging_setup.setup_logging", lambda **k: None)
    monkeypatch.delenv("BABERU_DIR", raising=False)
    monkeypatch.delenv("BABERU_LOG_DIR", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    fake_llms["main"] = FakeLLM(translate_all)
    (tmp_path / "ctx.txt").write_text("context", encoding="utf-8")
    numbered_subs(6).save(str(tmp_path / "video.ass"), encoding="utf-8")
    return tmp_path


@pytest.fixture
def media(workspace, monkeypatch):
    """Adds show.mp4, and stands in for ffmpeg and the transcription service."""
    transcriber = FakeTranscriber()
    monkeypatch.setattr(AIToolFactory, "get_transcription_provider", staticmethod(lambda model: transcriber))

    def fake_extract(video, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"audio")
        return output

    monkeypatch.setattr("baberu.tools.av_utils.get_audio_codec", lambda path: "aac")
    monkeypatch.setattr("baberu.tools.av_utils.extract_audio", fake_extract)
    monkeypatch.setattr("baberu.tools.av_utils.duration_seconds", lambda path: 60.0)
    monkeypatch.setattr("baberu.transcription.transcript_segmented.find_segments", lambda *a, **k: [])
    (workspace / "show.mp4").touch()
    return transcriber


def run_cli(monkeypatch, *argv) -> None:
    from baberu import cli

    monkeypatch.setattr(sys, "argv", ["baberu", *map(str, argv)])
    cli.main()


def texts(path) -> list[str]:
    return [e.text for e in SSAFile.load(str(path), encoding="utf-8").events]
