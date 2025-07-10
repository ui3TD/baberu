try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "The 'openai' package is required to use Deepseek models. "
        "Install it with: pip install openai"
    )

from .base import LLMProvider
import json

class DeepseekProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, system_prompt: str = None):
        super().__init__(api_key, model, system_prompt)
        self.client = OpenAI(api_key=self.DEEP_API_KEY, base_url="https://api.deepseek.com/beta")

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
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": prefill, "prefix": True}
        ]
        self.logger.debug(f"Deepseek prompt messages:\n{json.dumps(prompt_messages, indent=2)}")
        completion = self.client.responses.create(
            model=self.model,
            input=prompt_messages
        )
        self.logger.debug(f"Deepseek response:\n{json.dumps(completion, indent=2)}")

        response = ""
        if return_prefill:
            response += prefill

        response += completion.output_text

        return response
