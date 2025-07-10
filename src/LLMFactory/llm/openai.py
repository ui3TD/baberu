try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "The 'openai' package is required to use OpenAI models. "
        "Install it with: pip install openai"
    )

from .base import LLMProvider
import json

class OProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, system_prompt: str = None):
        super().__init__(api_key, model, system_prompt)
        self.client = OpenAI(api_key=self.api_key)

    def prompt(self, user_prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        """
        Sends a prompt to Gemini and returns the response.
        
        Supported kwargs:
        - None
        """
        if system_prompt is None:
            system_prompt = self.system_prompt

        prompt_messages = [
            {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
        ]
        self.logger.debug(f"OpenAI prompt messages:\n{json.dumps(prompt_messages, indent=2)}")
        completion = self.client.responses.create(
            model=self.model,
            input=prompt_messages,
            reasoning={"effort": "high"}
        )
        self.logger.debug(f"OpenAI response:\n{json.dumps(completion, indent=2)}")
        response = completion.output_text

        return response

class GPTProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, system_prompt: str = None):
        super().__init__(api_key, model, system_prompt)
        self.client = OpenAI(api_key=self.api_key)

    def prompt(self, user_prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        """
        Sends a prompt to Gemini and returns the response.
        
        Supported kwargs:
        - grounding (bool): Enable Web Search grounding (default: False)
        """
        if system_prompt is None:
            system_prompt = self.system_prompt

        grounding = kwargs.get("grounding", False)

        search_tool = []
        if grounding:
            search_tool={"type": "web_search_preview"}
            
        prompt_messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        self.logger.debug(f"Prompt messages:\n{json.dumps(prompt_messages, indent=2)}")
        completion = self.client.responses.create(
            model=self.model,
            input=prompt_messages,
            tools=search_tool
        )
        response = completion.output_text

        return response