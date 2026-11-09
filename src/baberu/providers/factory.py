from dataclasses import dataclass
from importlib import import_module
from os import environ
from typing import Any

from .llm import base as llm_base
from .transcription import base as transcription_base


@dataclass(frozen=True)
class _Route:
    """Connects model names containing any of `tokens` to a provider class."""

    tokens: tuple[str, ...]
    module: str
    class_name: str
    env_var: str

    def matches(self, model_name: str) -> bool:
        return any(token in model_name.lower() for token in self.tokens)

    def provider_class(self) -> type:
        # Imported on demand so an unused provider's SDK is never loaded.
        return getattr(import_module(f".{self.module}", __package__), self.class_name)


# Order matters: the first match wins. Names like "google/gemini-pro" must
# reach OpenRouter before "gemini" is considered.
_LLM_ROUTES = (
    _Route(("/",), "llm.openrouter", "OpenRouterProvider", "OPENROUTER_API_KEY"),
    _Route(("gemini",), "llm.gemini", "GeminiProvider", "GEMINI_API_KEY"),
    _Route(("claude",), "llm.claude", "ClaudeProvider", "ANTHROPIC_API_KEY"),
    _Route(("grok",), "llm.grok", "GrokProvider", "XAI_API_KEY"),
    _Route(("o1", "o3", "o4"), "llm.openai", "OProvider", "OPENAI_API_KEY"),
    _Route(("gpt",), "llm.openai", "GPTProvider", "OPENAI_API_KEY"),
)

_TRANSCRIPTION_ROUTES = (
    _Route(("scribe",), "transcription.elevenlabs", "ScribeProvider", "ELEVENLABS_API_KEY"),
    _Route(("whisper-1",), "transcription.openai", "WhisperProvider", "OPENAI_API_KEY"),
)


def _find_route(routes: tuple[_Route, ...], model_name: str, kind: str) -> _Route:
    for route in routes:
        if route.matches(model_name):
            return route
    raise ValueError(f"Could not determine {kind} provider for model: {model_name}")


def _api_key(route: _Route) -> str:
    api_key = environ.get(route.env_var, "")
    if not api_key:
        raise ValueError(f"API key not found. Set the {route.env_var} environment variable.")
    return api_key


class AIToolFactory:
    """Creates AI tool providers from a model name."""

    @staticmethod
    def get_llm_provider(model_name: str, system_prompt: str = "", **options: Any) -> llm_base.LLMProvider:
        """Creates the LLM provider for a model.

        Args:
            model_name: The name of the language model.
            system_prompt: An optional default system prompt.
            **options: Passed to the provider, e.g. `max_retries`, or `effort` for Claude.
        """
        route = _find_route(_LLM_ROUTES, model_name, "LLM")
        return route.provider_class()(api_key=_api_key(route), model=model_name, system_prompt=system_prompt, **options)

    @staticmethod
    def get_transcription_provider(model_name: str) -> transcription_base.TranscriptionProvider:
        """Creates the audio transcription provider for a model."""
        route = _find_route(_TRANSCRIPTION_ROUTES, model_name, "transcription")
        return route.provider_class()(api_key=_api_key(route), model=model_name)

    @staticmethod
    def get_transcription_provider_type(model_name: str) -> type[transcription_base.TranscriptionProvider]:
        """Returns the provider class without needing an API key, for reading saved transcripts."""
        return _find_route(_TRANSCRIPTION_ROUTES, model_name, "transcription").provider_class()
