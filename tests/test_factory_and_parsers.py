import json

import pytest

from baberu.providers.factory import AIToolFactory
from baberu.providers.transcription.elevenlabs import ScribeProvider
from baberu.providers.transcription.openai import WhisperProvider


@pytest.mark.parametrize(
    "model,env,cls_name",
    [
        ("google/gemini-pro", "OPENROUTER_API_KEY", "OpenRouterProvider"),
        ("gemini-2.5-pro", "GEMINI_API_KEY", "GeminiProvider"),
        ("claude-opus-5-5", "ANTHROPIC_API_KEY", "ClaudeProvider"),
        ("grok-4", "XAI_API_KEY", "GrokProvider"),
        ("o3", "OPENAI_API_KEY", "OProvider"),
        ("gpt-4.1-mini", "OPENAI_API_KEY", "GPTProvider"),
    ],
)
def test_llm_routing(monkeypatch, model, env, cls_name):
    monkeypatch.setenv(env, "test-key")

    assert type(AIToolFactory.get_llm_provider(model)).__name__ == cls_name


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key"):
        AIToolFactory.get_llm_provider("gemini-2.5-pro")


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Could not determine"):
        AIToolFactory.get_llm_provider("mystery-model")


def scribe_response(speaker="speaker_0") -> dict:
    words = [{"text": "hello", "start": 0.0, "end": 1.0, "type": "word", "speaker_id": speaker, "logprob": -0.1}]
    return {
        "language_code": "en",
        "language_probability": 1.0,
        "text": "hello",
        "words": words,
        "additional_formats": [
            {
                "requested_format": "segmented_json",
                "file_extension": "json",
                "content_type": "application/json",
                "is_base_64_encoded": False,
                "content": json.dumps({"segments": [{"text": "hello", "words": words}]}),
            }
        ],
    }


def test_scribe_parses_segmented_json():
    result = ScribeProvider.parse(scribe_response())

    assert [w.text for w in result.segments[0].words] == ["hello"]
    assert result.segments[0].words[0].speaker == "speaker_0"


def whisper_response() -> dict:
    return {
        "text": "Hello, world.",
        "segments": [{"text": "Hello, world.", "start": 0.0, "end": 2.0}],
        "words": [{"word": "Hello", "start": 0.0, "end": 1.0}, {"word": "world", "start": 1.0, "end": 2.0}],
    }


def test_whisper_keeps_punctuation_between_words():
    result = WhisperProvider.parse(whisper_response())

    assert "".join(w.text for w in result.segments[0].words) == "Hello, world."


def test_whisper_round_trips_through_provider_format():
    first = WhisperProvider.parse(whisper_response())

    second = WhisperProvider.parse(WhisperProvider.to_provider_format(first))

    assert "".join(w.text for w in second.segments[0].words) == "Hello, world."
