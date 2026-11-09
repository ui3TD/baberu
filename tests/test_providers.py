from types import SimpleNamespace

import pytest

from baberu.providers.factory import AIToolFactory
from baberu.providers.llm import base as llm_base
from baberu.providers.llm.base import LLMProvider, is_retryable
from baberu.providers.llm.claude import ClaudeProvider
from baberu.providers.llm.grok import GrokProvider


class StatusError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


class CodeError(Exception):
    """Google's SDK reports the status as `.code`."""

    def __init__(self, code):
        self.code = code


class APIConnectionError(Exception):
    pass


class ScriptedProvider(LLMProvider):
    def __init__(self, outcomes, **kwargs):
        super().__init__("key", "model", **kwargs)
        self.outcomes = iter(outcomes)
        self.seen_system = None

    def _complete(self, user_prompt, system_prompt, grounding, prefill):
        self.seen_system = system_prompt
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_base.time, "sleep", lambda s: None)


@pytest.mark.parametrize(
    "error,expected",
    [
        (StatusError(429), True),
        (StatusError(503), True),
        (CodeError(500), True),
        (APIConnectionError(), True),
        (TimeoutError(), True),
        (StatusError(400), False),
        (StatusError(401), False),
        (ValueError("x"), False),
    ],
)
def test_is_retryable(error, expected):
    assert is_retryable(error) is expected


def test_temporary_failures_are_retried():
    provider = ScriptedProvider([StatusError(503), APIConnectionError(), "done"])

    assert provider.prompt("hi") == "done"


def test_retries_stop_at_the_limit():
    provider = ScriptedProvider([StatusError(503)] * 3, max_retries=2)

    with pytest.raises(StatusError):
        provider.prompt("hi")


def test_permanent_failures_are_not_retried():
    provider = ScriptedProvider([StatusError(400), "never"])

    with pytest.raises(StatusError):
        provider.prompt("hi")


def test_system_prompt_defaults_and_overrides():
    provider = ScriptedProvider(["a", "b"], system_prompt="default")

    provider.prompt("hi")
    assert provider.seen_system == "default"
    provider.prompt("hi", system_prompt="other")
    assert provider.seen_system == "other"


def test_unsupported_grounding_is_an_error():
    with pytest.raises(ValueError, match="grounding"):
        GrokProvider("k", "grok-4").prompt("hi", grounding=True)


def test_unsupported_prefill_is_an_error_but_empty_prefill_is_fine():
    provider = ScriptedProvider(["ok"])

    assert provider.prompt("hi", prefill="") == "ok"
    with pytest.raises(ValueError, match="prefill"):
        provider.prompt("hi", prefill="Sure:")


def fake_stream(messages_seen, responses):
    class Stream:
        def __init__(self, response):
            self.response = response

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return self.response

    def stream(**kwargs):
        messages_seen.append(kwargs)
        return Stream(next(responses))

    return SimpleNamespace(messages=SimpleNamespace(stream=stream))


def text_message(text, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])


def claude_with(responses):
    seen = []
    provider = ClaudeProvider("k", "claude-opus-5-5", system_prompt="sys")
    provider.client = fake_stream(seen, iter(responses))
    return provider, seen


def test_claude_request_uses_adaptive_thinking_without_prefill():
    provider, seen = claude_with([text_message("hello")])

    assert provider.prompt("hi") == "hello"
    assert seen[0]["thinking"] == {"type": "adaptive"}
    assert seen[0]["output_config"] == {"effort": "high"}
    assert seen[0]["system"] == "sys"
    assert [m["role"] for m in seen[0]["messages"]] == ["user"]
    assert "tools" not in seen[0]


def test_claude_grounding_uses_current_web_search():
    provider, seen = claude_with([text_message("hello")])

    provider.prompt("hi", grounding=True)

    assert seen[0]["tools"] == [{"type": "web_search_20260209", "name": "web_search"}]


def test_claude_joins_text_blocks_once():
    message = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="a"),
            SimpleNamespace(type="text", text="b"),
        ],
    )
    provider, _ = claude_with([message])

    assert provider.prompt("hi") == "ab"


def test_claude_resumes_a_paused_turn():
    paused = text_message("", stop_reason="pause_turn")
    provider, seen = claude_with([paused, text_message("finished")])

    assert provider.prompt("hi", grounding=True) == "finished"
    assert seen[1]["messages"][1] == {"role": "assistant", "content": paused.content}


def test_claude_refusal_returns_empty_text():
    provider, _ = claude_with([text_message("", stop_reason="refusal")])

    assert provider.prompt("hi") == ""


def test_factory_passes_options_to_the_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    provider = AIToolFactory.get_llm_provider("claude-opus-5-5", max_retries=9, effort="low")

    assert (provider.max_retries, provider.effort) == (9, "low")


@pytest.mark.parametrize("model", ["deepseek-chat", "whisper-v3"])
def test_removed_providers_are_gone(model, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    if model == "whisper-v3":
        with pytest.raises(ValueError):
            AIToolFactory.get_transcription_provider(model)
    else:
        with pytest.raises(ValueError):
            AIToolFactory.get_llm_provider(model)


def test_transcription_type_needs_no_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    assert AIToolFactory.get_transcription_provider_type("scribe_v1").__name__ == "ScribeProvider"
