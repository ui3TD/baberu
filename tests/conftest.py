"""Shared test helpers. Nothing here touches the network or ffmpeg."""
import re
from collections.abc import Callable

import pytest
from pysubs2 import SSAEvent, SSAFile

from LLMFactory.factory import AIToolFactory

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

    def get_llm_provider(model_name: str, system_prompt: str = ""):
        return llms[model_name]

    monkeypatch.setattr(AIToolFactory, "get_llm_provider", staticmethod(get_llm_provider))
    return llms
