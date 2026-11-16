from openai import OpenAI

from .base import Completion, LLMProvider, responses_completion


class OProvider(LLMProvider):
    """OpenAI reasoning models (o-series)."""

    def __init__(self, api_key: str, model: str, system_prompt: str | None = None, max_retries: int = 5):
        super().__init__(api_key, model, system_prompt, max_retries)
        self.client = OpenAI(api_key=self.api_key, max_retries=0)

    def _complete(self, user_prompt: str, system_prompt: str, grounding: bool, prefill: str) -> Completion:
        completion = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}],
            reasoning={"effort": "high"},
        )
        return responses_completion(completion)


class GPTProvider(LLMProvider):
    supports_grounding = True

    def __init__(self, api_key: str, model: str, system_prompt: str | None = None, max_retries: int = 5):
        super().__init__(api_key, model, system_prompt, max_retries)
        self.client = OpenAI(api_key=self.api_key, max_retries=0)

    def _complete(self, user_prompt: str, system_prompt: str, grounding: bool, prefill: str) -> Completion:
        completion = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=[{"type": "web_search_preview"}] if grounding else [],
        )
        return responses_completion(completion)
