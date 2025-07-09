from .base import LLMProvider
import json
from openai import OpenAI

class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, system_prompt: str = None):
        super().__init__(api_key, model, system_prompt)
        self.client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")

    def prompt(self, user_prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        """
        Sends a prompt to Gemini and returns the response.
        
        Supported kwargs:
        - prefill (str): Enable prefill text (default: "")
        - return_prefill (bool): Return prefill text prepended to response (default: False)
        """
        if system_prompt is None:
            system_prompt = self.system_prompt

        prefill = kwargs.get("prefill", "")
        return_prefill = kwargs.get("return_prefill", False)

        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if prefill:
            prompt_messages.append({"role": "assistant", "content": prefill})
            
        self.logger.debug(f"Openrouter prompt messages:\n{json.dumps(prompt_messages, indent=2)}")

        completion = self.client.responses.create(
            model=self.model,
            input=prompt_messages
        )
        self.logger.debug(f"Openrouter response:\n{json.dumps(completion, indent=2)}")

        response = ""
        if return_prefill:
            response += prefill
        response += completion.output_text

        return response