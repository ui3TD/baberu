from types import SimpleNamespace

import pytest

from baberu.providers.factory import AIToolFactory
from baberu.providers.llm import base as llm_base
from baberu.providers.llm.base import Completion, LLMProvider, is_retryable
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
        return Completion.of(outcome)


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


def test_complete_reports_status_and_prompt_returns_text():
    provider = ScriptedProvider(["", "hi"])

    assert provider.complete("x").status == "empty"
    assert provider.prompt("x") == "hi"


@pytest.mark.parametrize(
    "stop_reason,status",
    [("end_turn", "complete"), ("max_tokens", "truncated"), ("refusal", "refused")],
)
def test_claude_maps_stop_reason_to_status(stop_reason, status):
    provider, _ = claude_with([text_message("partial", stop_reason=stop_reason)])

    assert provider.complete("hi").status == status


def gemini_response(text, finish_reason=None, block_reason=None):
    candidates = [SimpleNamespace(finish_reason=SimpleNamespace(name=finish_reason))] if finish_reason else []
    return SimpleNamespace(text=text, candidates=candidates,
                           prompt_feedback=SimpleNamespace(block_reason=block_reason))


@pytest.mark.parametrize(
    "response,status",
    [
        (gemini_response("ok", "STOP"), "complete"),
        (gemini_response("cut", "MAX_TOKENS"), "truncated"),
        (gemini_response(None, "SAFETY"), "refused"),
        (gemini_response(None, "PROHIBITED_CONTENT"), "refused"),
        (gemini_response(None, block_reason="SAFETY"), "refused"),
        (gemini_response(None, "STOP"), "empty"),
    ],
)
def test_gemini_maps_finish_reason_to_status(response, status):
    from baberu.providers.llm.gemini import gemini_completion

    assert gemini_completion(response).status == status


def responses_reply(text, status="completed", reason=None, refusal=False):
    content = [SimpleNamespace(type="refusal", refusal="no")] if refusal else [SimpleNamespace(type="output_text")]
    return SimpleNamespace(output_text=text, status=status,
                           incomplete_details=SimpleNamespace(reason=reason) if reason else None,
                           output=[SimpleNamespace(type="message", content=content)])


@pytest.mark.parametrize(
    "response,status",
    [
        (responses_reply("ok"), "complete"),
        (responses_reply("cut", "incomplete", "max_output_tokens"), "truncated"),
        (responses_reply("", "incomplete", "content_filter"), "refused"),
        (responses_reply("", refusal=True), "refused"),
        (responses_reply(""), "empty"),
    ],
)
def test_responses_api_maps_status(response, status):
    from baberu.providers.llm.base import responses_completion

    assert responses_completion(response).status == status


@pytest.mark.parametrize("module,cls", [("openai", "GPTProvider"), ("openai", "OProvider"),
                                        ("grok", "GrokProvider")])
def test_responses_providers_report_truncation(module, cls):
    import importlib

    provider = getattr(importlib.import_module(f"baberu.providers.llm.{module}"), cls)("k", "m")
    reply = responses_reply("cut", "incomplete", "max_output_tokens")
    provider.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: reply))

    assert provider.complete("hi").status == "truncated"


def test_openrouter_reports_the_providers_cost_for_byok():
    from baberu.providers.llm.base import responses_usage, usage_summary

    def reply(usage):
        return SimpleNamespace(usage=SimpleNamespace(model_dump=lambda: usage))

    byok = responses_usage("m", reply({"input_tokens": 16, "output_tokens": 10, "cost": 0, "is_byok": True,
                                       "output_tokens_details": {"reasoning_tokens": 4},
                                       "cost_details": {"upstream_inference_cost": 6.6e-06}}))
    credits = responses_usage("m", reply({"input_tokens": 1, "output_tokens": 1, "cost": 0.5, "is_byok": False}))

    assert (byok.input_tokens, byok.output_tokens, byok.reasoning_tokens, byok.cost) == (16, 10, 4, 6.6e-06)
    assert credits.cost == 0.5
    assert usage_summary([byok, credits]) == "m: 2 requests, 17 in, 11 out, $0.500007"


# --- OpenRouter (chat completions, for caching) ---

def chat_reply(text, finish_reason="stop", refusal=None):
    message = SimpleNamespace(content=text, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)], usage=None)


@pytest.mark.parametrize("reply,status", [
    (chat_reply("ok"), "complete"),
    (chat_reply("cut", "length"), "truncated"),
    (chat_reply("", "content_filter"), "refused"),
    (chat_reply("", refusal="no"), "refused"),
    (chat_reply(""), "empty"),
])
def test_openrouter_reads_how_a_reply_ended(reply, status):
    from baberu.providers.llm.openrouter import OpenRouterProvider

    provider = OpenRouterProvider("k", "m")
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: reply)))

    assert provider.complete("hi").status == status


def test_openrouter_marks_the_cacheable_prefix():
    from baberu.providers.llm.openrouter import OpenRouterProvider

    sent = {}
    provider = OpenRouterProvider("k", "m", system_prompt="sys")
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: sent.update(kw) or chat_reply("ok"))))

    provider.complete("Context:\nnotes\n\nTranslate 1. x", cache_prefix="Context:\nnotes\n\n")

    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["messages"][1]["content"] == [
        {"type": "text", "text": "Context:\nnotes\n\n", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Translate 1. x"},
    ]

    provider.complete("plain")   # without a prefix, the prompt goes as plain text
    assert sent["messages"][1]["content"] == "plain"


def test_providers_that_cannot_cache_ignore_the_prefix():
    provider = ScriptedProvider(["ok"])

    assert provider.complete("Context: x", cache_prefix="Context:").text == "ok"
