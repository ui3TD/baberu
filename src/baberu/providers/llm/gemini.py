from google.genai import Client
from google.genai.types import Content, GenerateContentConfig, GoogleSearch, Part, SafetySetting, Tool

from .base import COMPLETE, REFUSED, TRUNCATED, Completion, Image, LLMProvider, Prompt, prompt_parts

_UNFILTERED = [
    SafetySetting(category=category, threshold="BLOCK_NONE")
    for category in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]


class GeminiProvider(LLMProvider):
    supports_grounding = True
    supports_images = True

    def __init__(self, api_key: str, model: str, system_prompt: str | None = None, max_retries: int = 5,
                 effort: str | None = None):
        super().__init__(api_key, model, system_prompt, max_retries, effort)
        self.client = Client(api_key=self.api_key)

    def _complete(self, user_prompt: Prompt, system_prompt: str, grounding: bool, prefill: str) -> Completion:
        config = {"system_instruction": system_prompt, "safety_settings": _UNFILTERED}
        if grounding:
            config["tools"] = [Tool(google_search=GoogleSearch())]

        response = self.client.models.generate_content(
            model=self.model,
            contents=[Content(parts=user_parts(user_prompt), role="user")],
            config=GenerateContentConfig(**config),
        )

        return gemini_completion(response)


def user_parts(prompt: Prompt) -> list[Part]:
    """The prompt as Gemini parts: text, and images as inline bytes."""
    return [Part.from_bytes(data=part.data, mime_type=part.media_type) if isinstance(part, Image) else Part(text=part)
            for part in prompt_parts(prompt) if part != ""]


_REFUSAL_REASONS = {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "MODEL_ARMOR"}


def gemini_completion(response) -> Completion:
    """Reads how a Gemini reply ended from its finish reason, or from a blocked prompt."""
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        return Completion("", REFUSED)

    candidates = getattr(response, "candidates", None) or []
    reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    reason = getattr(reason, "name", reason)
    text = response.text if candidates else ""
    if reason == "MAX_TOKENS":
        return Completion.of(text, TRUNCATED)
    if reason in _REFUSAL_REASONS:
        return Completion(text or "", REFUSED)
    return Completion.of(text, COMPLETE)
