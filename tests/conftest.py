"""Shared test helpers. Nothing here touches the network or ffmpeg."""
import importlib.resources
import re
import sys
from collections.abc import Callable

import pytest
import yaml
from pysubs2 import SSAEvent, SSAFile

from baberu.providers.factory import AIToolFactory
from baberu.providers.llm.base import Completion, prompt_text
from baberu.providers.transcription.base import TranscriptionProvider
from baberu.providers.transcription.openai import WhisperProvider

BATCH_MARKER = "Maintain exact entry count and order:\n\n"
RETRY_MARKER = "Translate only these entries, keeping their numbers:\n\n"
ENTRY = re.compile(r"^(\d+)\. (.*)$")


def make_subs(rows: list[tuple[int, int, str]]) -> SSAFile:
    """Builds subtitles from (start_ms, end_ms, text) rows."""
    subs = SSAFile()
    subs.events = [SSAEvent(start=s, end=e, text=t, style="Default") for s, e, t in rows]
    return subs


def numbered_subs(count: int) -> SSAFile:
    return make_subs([(i * 2000, i * 2000 + 1500, f"line{i}") for i in range(count)])


def numbered_entries(prompt) -> list[tuple[int, str]]:
    """Pulls the requested (number, source text) entries out of a translation or retry prompt.

    A prompt with pictures is read from its first part, the text before any frames.
    """
    prompt = prompt if isinstance(prompt, str) else prompt[0]
    body = prompt.split(RETRY_MARKER if RETRY_MARKER in prompt else BATCH_MARKER)[-1]
    return [(int(m.group(1)), m.group(2)) for m in map(ENTRY.match, body.split("\n")) if m]


def batch_entries(prompt: str) -> list[str]:
    """Pulls the requested source entries out of a translation prompt."""
    return [text for _, text in numbered_entries(prompt)]


def is_retry(prompt) -> bool:
    return RETRY_MARKER in prompt_text(prompt)


def translate_all(prompt: str) -> str:
    """A well-behaved translator: one numbered 'T:' line per requested entry."""
    return "\n".join(f"{n}. T:{text}" for n, text in numbered_entries(prompt))


def translate_or_describe(prompt: str) -> str:
    """Translates a batch, or answers a context request with some notes."""
    return translate_all(prompt) if numbered_entries(prompt) else "Generated notes."


class FakeLLM:
    """Answers with what `responder` returns: text, a Completion, or an exception to raise."""
    supports_grounding = True

    def __init__(self, responder: Callable[[str], "str | Completion | Exception"], model: str = "fake-model",
                 images: bool = False):
        self.responder = responder
        self.model = model
        self.images = images
        self.calls: list = []
        self.kwargs: list[dict] = []

    def takes_images(self) -> bool:
        return self.images

    def complete(self, user_prompt: str, system_prompt: str | None = None, **kwargs) -> Completion:
        self.calls.append(user_prompt)
        self.kwargs.append(kwargs)
        outcome = self.responder(user_prompt)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome if isinstance(outcome, Completion) else Completion.of(outcome)

    def prompt(self, user_prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        return self.complete(user_prompt, system_prompt, **kwargs).text


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
    config["frames"]["enabled"] = False
    monkeypatch.setattr("baberu.setup.config_setup.load_config", lambda *a: config)
    monkeypatch.setattr("baberu.setup.logging_setup.setup_logging", lambda **k: None)
    monkeypatch.delenv("BABERU_DIR", raising=False)
    monkeypatch.delenv("BABERU_LOG_DIR", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    fake_llms["main"] = FakeLLM(translate_or_describe)
    (tmp_path / "ctx.txt").write_text("context", encoding="utf-8")
    numbered_subs(6).save(str(tmp_path / "video.ass"), encoding="utf-8")
    return tmp_path


@pytest.fixture
def media(workspace, monkeypatch):
    """Adds show.mp4, and stands in for ffmpeg and the transcription service."""
    transcriber = FakeTranscriber()
    monkeypatch.setattr(AIToolFactory, "get_transcription_provider", staticmethod(lambda model, options=None: transcriber))

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
