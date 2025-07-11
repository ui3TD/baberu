try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "The 'openai' package is required to use xAI models. "
        "Install it with: pip install openai"
    )

from .base import LLMProvider
import json

class GrokProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, system_prompt: str = None):
        super().__init__(api_key, model, system_prompt)
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")

    def prompt(self, user_prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        """
        Sends a prompt to Gemini and returns the response.
        
        Supported kwargs:
        - None
        """
        if system_prompt is None:
            system_prompt = self.system_prompt

        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        self.logger.debug(f"Grok prompt messages:\n{json.dumps(prompt_messages, indent=2)}")
        completion = self.client.responses.create(
            model=self.model,
            input=prompt_messages
        )
        self.logger.debug(f"Grok response:\n{completion.model_dump_json()}")
        response = completion.output_text

        return response