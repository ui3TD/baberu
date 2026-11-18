import functools
import logging

import httpx
from openai import OpenAI

from .base import (COMPLETE, REFUSED, TRUNCATED, Completion, Image, LLMProvider, Prompt, record_usage,
                   responses_usage, split_prefix)

MODELS_URL = "https://openrouter.ai/api/v1/models"

logger = logging.getLogger(__name__)


@functools.cache
def input_modalities() -> dict[str, frozenset[str]]:
    """What each OpenRouter model takes as input (text, image, ...), by model id. Read once per run."""
    reply = httpx.get(MODELS_URL, timeout=30)
    reply.raise_for_status()
    return {model["id"]: frozenset((model.get("architecture") or {}).get("input_modalities") or [])
            for model in reply.json().get("data", [])}


class OpenRouterProvider(LLMProvider):
    """Any model on OpenRouter, named as vendor/model (e.g. anthropic/claude-sonnet-5.5).

    Uses the chat completions API. OpenRouter passes cache markers on to the
    provider there, and ignores them in its Responses API.
    """

    supports_grounding = True  # through OpenRouter's web plugin, whatever the model
    supports_prefill = True
    supports_caching = True
    supports_images = True     # for models whose OpenRouter listing includes image input

    def __init__(self, api_key: str, model: str, system_prompt: str | None = None, max_retries: int = 5,
                 effort: str | None = None):
        super().__init__(api_key, model, system_prompt, max_retries, effort)
        self.client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1", max_retries=0)

    def takes_images(self) -> bool:
        """Whether OpenRouter lists image input for this model. False, with a warning, when the list can't be read."""
        try:
            modalities = input_modalities()
        except (httpx.HTTPError, ValueError) as error:
            logger.warning(f"Can't read OpenRouter's model list ({error}); sending {self.model} no images.")
            return False
        return "image" in modalities.get(self.model.split(":")[0], frozenset())

    def request_body(self, grounding: bool) -> dict:
        """OpenRouter's own fields, beyond the chat completions API."""
        body: dict = {"usage": {"include": True}}  # adds the cost to the reply
        if self.effort:
            body["reasoning"] = {"effort": self.effort}
        if grounding:
            body["plugins"] = [{"id": "web"}]
        return body

    @staticmethod
    def messages(user_prompt: Prompt, system_prompt: str, prefill: str = "", cache_prefix: str = "") -> list[dict]:
        """The request's messages. A cache marker after `cache_prefix` caches it with the system prompt.

        Plain text with no prefix is sent as a string; anything else as a list of parts.
        """
        user: str | list[dict] = user_prompt if isinstance(user_prompt, str) and not cache_prefix else []
        if not isinstance(user, str):
            parts = split_prefix(user_prompt, cache_prefix)
            for index, part in enumerate(parts):
                if isinstance(part, Image):
                    user.append({"type": "image_url", "image_url": {"url": part.data_url}})
                    continue
                block: dict = {"type": "text", "text": part}
                if index == 0 and cache_prefix and part == cache_prefix:
                    block["cache_control"] = {"type": "ephemeral"}
                user.append(block)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
        return messages

    def _complete(self, user_prompt: Prompt, system_prompt: str, grounding: bool, prefill: str,
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
