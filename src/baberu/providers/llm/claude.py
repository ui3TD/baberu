from anthropic import Anthropic

from .base import LLMProvider

MAX_OUTPUT_TOKENS = 64000
MAX_CONTINUATIONS = 5


class ClaudeProvider(LLMProvider):
    """Claude 4.6 and later, which use adaptive thinking and don't accept prefill."""

    supports_grounding = True

    def __init__(self, api_key: str, model: str, system_prompt: str | None = None,
                 max_retries: int = 5, effort: str = "high"):
        super().__init__(api_key, model, system_prompt, max_retries)
        self.effort = effort
        # Retries are handled once, in LLMProvider.prompt.
        self.client = Anthropic(api_key=self.api_key, max_retries=0)

    def _complete(self, user_prompt: str, system_prompt: str, grounding: bool, prefill: str) -> str:
        request = {
            "model": self.model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
        }
        if system_prompt:
            request["system"] = system_prompt
        if grounding:
            request["tools"] = [{"type": "web_search_20260209", "name": "web_search"}]

        messages = [{"role": "user", "content": user_prompt}]
        message = self._send(request, messages)

        # A long web search can pause; sending the turn back resumes it.
        continuations = 0
        while message.stop_reason == "pause_turn" and continuations < MAX_CONTINUATIONS:
            messages = [messages[0], {"role": "assistant", "content": message.content}]
            message = self._send(request, messages)
            continuations += 1

        if message.stop_reason == "refusal":
            self.logger.warning(f"{self.model} declined the request.")
            return ""
        if message.stop_reason == "max_tokens":
            self.logger.warning(f"{self.model} response was cut off at the output limit.")

        return "".join(block.text for block in message.content if block.type == "text")

    def _send(self, request: dict, messages: list[dict]):
        with self.client.messages.stream(messages=messages, **request) as stream:
            return stream.get_final_message()
