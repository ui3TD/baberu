from anthropic import Anthropic

from .base import COMPLETE, REFUSED, TRUNCATED, Completion, Image, LLMProvider, Prompt, prompt_parts

MAX_OUTPUT_TOKENS = 64000
MAX_CONTINUATIONS = 5


class ClaudeProvider(LLMProvider):
    """Claude 4.6 and later, which use adaptive thinking and don't accept prefill."""

    supports_grounding = True
    supports_images = True

    def __init__(self, api_key: str, model: str, system_prompt: str | None = None,
                 max_retries: int = 5, effort: str | None = None):
        super().__init__(api_key, model, system_prompt, max_retries, effort or "high")
        # SDK retries are off; LLMProvider.complete retries.
        self.client = Anthropic(api_key=self.api_key, max_retries=0)

    def _complete(self, user_prompt: Prompt, system_prompt: str, grounding: bool, prefill: str) -> Completion:
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

        messages = [{"role": "user", "content": user_content(user_prompt)}]
        message = self._send(request, messages)

        # A long web search can pause; sending the turn back resumes it.
        continuations = 0
        while message.stop_reason == "pause_turn" and continuations < MAX_CONTINUATIONS:
            messages = [messages[0], {"role": "assistant", "content": message.content}]
            message = self._send(request, messages)
            continuations += 1

        text = "".join(block.text for block in message.content if block.type == "text")
        if message.stop_reason == "refusal":
            return Completion("", REFUSED)
        if message.stop_reason in ("max_tokens", "model_context_window_exceeded", "pause_turn"):
            return Completion.of(text, TRUNCATED)
        return Completion.of(text, COMPLETE)

    def _send(self, request: dict, messages: list[dict]):
        with self.client.messages.stream(messages=messages, **request) as stream:
            return stream.get_final_message()


def user_content(prompt: Prompt) -> str | list[dict]:
    """The user turn's content: the text as it is, or text and image blocks in order."""
    if isinstance(prompt, str):
        return prompt
    return [{"type": "image", "source": {"type": "base64", "media_type": part.media_type, "data": part.base64}}
            if isinstance(part, Image) else {"type": "text", "text": part}
            for part in prompt_parts(prompt) if part != ""]
