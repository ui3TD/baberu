from openai import OpenAI

from .base import COMPLETE, REFUSED, TRUNCATED, Completion, LLMProvider, record_usage, responses_usage


class OpenRouterProvider(LLMProvider):
    """Any model on OpenRouter, named as vendor/model (e.g. anthropic/claude-sonnet-5.5).

    Uses the chat completions API: OpenRouter passes cache markers on to the
    provider there, but ignores them in its Responses API.
    """

    supports_grounding = True  # through OpenRouter's web plugin, whatever the model
    supports_prefill = True
    supports_caching = True

    def __init__(self, api_key: str, model: str, system_prompt: str | None = None, max_retries: int = 5,
                 effort: str | None = None):
        super().__init__(api_key, model, system_prompt, max_retries, effort)
        self.client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1", max_retries=0)

    def request_body(self, grounding: bool) -> dict:
        """OpenRouter's own fields, beyond the chat completions API."""
        body: dict = {"usage": {"include": True}}  # adds the cost to the reply
        if self.effort:
            body["reasoning"] = {"effort": self.effort}
        if grounding:
            body["plugins"] = [{"id": "web"}]
        return body

    @staticmethod
    def messages(user_prompt: str, system_prompt: str, prefill: str = "", cache_prefix: str = "") -> list[dict]:
        """The request's messages. A cache marker after `cache_prefix` caches it with the system prompt."""
        user: str | list[dict] = user_prompt
        if cache_prefix:
            user = [{"type": "text", "text": cache_prefix, "cache_control": {"type": "ephemeral"}}]
            rest = user_prompt[len(cache_prefix):]
            if rest:
                user.append({"type": "text", "text": rest})
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
        return messages

    def _complete(self, user_prompt: str, system_prompt: str, grounding: bool, prefill: str,
                  cache_prefix: str = "") -> Completion:
        reply = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages(user_prompt, system_prompt, prefill, cache_prefix),
            extra_body=self.request_body(grounding),
        )
        usage = responses_usage(self.model, reply)
        if usage:
            record_usage(usage)
        return chat_completion(reply)


def chat_completion(reply) -> Completion:
    """Reads how a chat completions reply ended."""
    choice = (getattr(reply, "choices", None) or [None])[0]
    if choice is None:
        return Completion.of("")
    message = choice.message
    text = getattr(message, "content", None) or ""
    if getattr(message, "refusal", None) or choice.finish_reason == "content_filter":
        return Completion(text, REFUSED)
    if choice.finish_reason == "length":
        return Completion.of(text, TRUNCATED)
    return Completion.of(text, COMPLETE)
