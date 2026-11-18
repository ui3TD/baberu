from openai import OpenAI

from .base import Completion, Image, LLMProvider, Prompt, prompt_parts, responses_completion


class OProvider(LLMProvider):
    """OpenAI reasoning models (o-series)."""

    def __init__(self, api_key: str, model: str, system_prompt: str | None = None, max_retries: int = 5,
                 effort: str | None = None):
        super().__init__(api_key, model, system_prompt, max_retries, effort)
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
    supports_images = True

    def __init__(self, api_key: str, model: str, system_prompt: str | None = None, max_retries: int = 5,
                 effort: str | None = None):
        super().__init__(api_key, model, system_prompt, max_retries, effort)
        self.client = OpenAI(api_key=self.api_key, max_retries=0)

    def _complete(self, user_prompt: Prompt, system_prompt: str, grounding: bool, prefill: str) -> Completion:
        completion = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": user_content(user_prompt)},
            ],
            tools=[{"type": "web_search_preview"}] if grounding else [],
        )
        return responses_completion(completion)


def user_content(prompt: Prompt) -> str | list[dict]:
    """The user turn's content for the Responses API: the text as it is, or text and image parts in order."""
    if isinstance(prompt, str):
        return prompt
    return [{"type": "input_image", "image_url": part.data_url} if isinstance(part, Image)
            else {"type": "input_text", "text": part}
            for part in prompt_parts(prompt) if part != ""]
