from .base import LLMProvider
import json
from anthropic import Anthropic

class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, system_prompt: str = None):
        super().__init__(api_key, model, system_prompt)
        self.client = Anthropic(api_key=self.api_key)

    def prompt(self, user_prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        """
        Sends a prompt to Gemini and returns the response.
        
        Supported kwargs:
        - prefill (str): Enable prefill text (default: "")
        - return_prefill (bool): Return prefill text prepended to response (default: False)
        - return_thinking (bool): Return thinking (default: False)
        - grounding (bool): Enable Web Search grounding (default: False)
        """
        if system_prompt is None:
            system_prompt = self.system_prompt

        prefill = kwargs.get("prefill", "")
        return_prefill = kwargs.get("return_prefill", False)
        return_thinking = kwargs.get("return_thinking", False)
        grounding = kwargs.get("grounding", False)

        search_tool = []
        if grounding:
            search_tool = [{
                "type": "web_search_20250305",
                "name": "web_search"
            }]
        
        
        prompt_messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": prefill}
                ]
        self.logger.info(f"Prompt messages:\n{json.dumps(prompt_messages, indent=2)}")

        completion = self.client.messages.create(
            model=self.model,
            max_tokens=30000,
            system=system_prompt,
            thinking= {
                "type": "enabled",
                "budget_tokens": 25000
            },
            stream=False,
            timeout=600.0,
            tools=search_tool,
            messages=prompt_messages,
        )
        response = ""

        for block in completion.content:
            if return_thinking:
                if block.type == "thinking":
                    response += block.thinking + "\n\n"
            if block.type == "text":
                if return_prefill:
                    response += prefill
                response += prefill + block.text

        return response