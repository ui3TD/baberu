from openai import OpenAI

from .base import Completion, LLMProvider, responses_completion


class GrokProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, system_prompt: str | None = None, max_retries: int = 5,
                 effort: str | None = None):
        super().__init__(api_key, model, system_prompt, max_retries, effort)
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1", max_retries=0)

    def _complete(self, user_prompt: str, system_prompt: str, grounding: bool, prefill: str) -> Completion:
        completion = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return responses_completion(completion)
