from google.genai import Client
from google.genai.types import Content, GenerateContentConfig, GoogleSearch, Part, SafetySetting, Tool

from .base import LLMProvider

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

    def __init__(self, api_key: str, model: str, system_prompt: str | None = None, max_retries: int = 5):
        super().__init__(api_key, model, system_prompt, max_retries)
        self.client = Client(api_key=self.api_key)

    def _complete(self, user_prompt: str, system_prompt: str, grounding: bool, prefill: str) -> str:
        config = {"system_instruction": system_prompt, "safety_settings": _UNFILTERED}
        if grounding:
            config["tools"] = [Tool(google_search=GoogleSearch())]

        response = self.client.models.generate_content(
            model=self.model,
            contents=[Content(parts=[Part(text=user_prompt)], role="user")],
            config=GenerateContentConfig(**config),
        )

        if not response.text:
            self.logger.warning("Gemini returned empty or invalid response")
            return ""

        return response.text
