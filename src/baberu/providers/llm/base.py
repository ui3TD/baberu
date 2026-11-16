from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import time

RETRYABLE_STATUS_CODES = {408, 409, 429}
RETRYABLE_EXCEPTION_NAMES = {
    "APIConnectionError",
    "ConnectionError",
    "TimeoutError",
    "TransportError",
}
MAX_RETRY_DELAY_SECONDS = 60


def is_retryable(error: Exception) -> bool:
    """Tells whether trying the same request again could succeed.

    Goes by the HTTP status the error reports, or by the name of the network
    error class it is.
    """
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if isinstance(status, int):
        return status in RETRYABLE_STATUS_CODES or status >= 500
    return any(cls.__name__ in RETRYABLE_EXCEPTION_NAMES for cls in type(error).__mro__)


COMPLETE = "complete"
TRUNCATED = "truncated"
REFUSED = "refused"
EMPTY = "empty"


@dataclass(frozen=True)
class Completion:
    """A response, and how it ended: complete, truncated, refused or empty."""
    text: str
    status: str = COMPLETE

    @classmethod
    def of(cls, text: str | None, status: str = COMPLETE) -> "Completion":
        """Marks a response with no text as empty, unless a stronger reason is known."""
        text = text or ""
        if status == COMPLETE and not text.strip():
            status = EMPTY
        return cls(text, status)


@dataclass(frozen=True)
class Usage:
    """What one request used. `cost` is in US dollars, or None if the provider didn't say."""
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost: float | None = None


# Every request made in this run.
ledger: list[Usage] = []
_usage_logger = logging.getLogger("baberu.usage")


def record_usage(usage: Usage) -> None:
    ledger.append(usage)
    cost = f", ${usage.cost:.6f}" if usage.cost is not None else ""
    thinking = f" ({usage.reasoning_tokens:,} thinking)" if usage.reasoning_tokens else ""
    cached = f" ({usage.cached_tokens:,} cached)" if usage.cached_tokens else ""
    _usage_logger.info(f"{usage.model}: {usage.input_tokens:,} in{cached}, {usage.output_tokens:,} out{thinking}{cost}")


def usage_summary(entries: list[Usage] | None = None) -> str:
    """One line per model: requests, tokens and cost."""
    entries = ledger if entries is None else entries
    lines = []
    for model in dict.fromkeys(u.model for u in entries):
        mine = [u for u in entries if u.model == model]
        costs = [u.cost for u in mine if u.cost is not None]
        cost = f", ${sum(costs):.6f}" + ("" if len(costs) == len(mine) else " (some unknown)") if costs else ""
        lines.append(f"{model}: {len(mine)} requests, {sum(u.input_tokens for u in mine):,} in, "
                     f"{sum(u.output_tokens for u in mine):,} out{cost}")
    return "; ".join(lines)


def responses_usage(model: str, response) -> Usage | None:
    """Reads the usage block of a Responses API or chat completions reply.

    OpenRouter adds the cost: with your own provider key (BYOK), the provider's
    charge is in cost_details, and OpenRouter's own `cost` is its fee.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    data = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    details = data.get("cost_details") or {}
    cost = details.get("upstream_inference_cost") if data.get("is_byok") else data.get("cost")
    output_details = data.get("output_tokens_details") or data.get("completion_tokens_details") or {}
    input_details = data.get("input_tokens_details") or data.get("prompt_tokens_details") or {}
    return Usage(
        model=model,
        input_tokens=data.get("input_tokens") or data.get("prompt_tokens") or 0,
        output_tokens=data.get("output_tokens") or data.get("completion_tokens") or 0,
        reasoning_tokens=output_details.get("reasoning_tokens") or 0,
        cached_tokens=input_details.get("cached_tokens") or 0,
        cost=float(cost) if cost is not None else None,
    )


def responses_completion(response) -> Completion:
    """Reads how an OpenAI-style Responses API reply ended.

    OpenAI, Grok and OpenRouter all answer through this API.
    """
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                return Completion(response.output_text or "", REFUSED)
    if getattr(response, "status", None) == "incomplete":
        reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
        return Completion.of(response.output_text, REFUSED if reason == "content_filter" else TRUNCATED)
    return Completion.of(response.output_text)


class LLMProvider(ABC):
    """Abstract base class for all Large Language Model providers."""

    supports_grounding: bool = False
    supports_prefill: bool = False
    supports_caching: bool = False  # takes `cache_prefix` in _complete

    def __init__(self, api_key: str, model: str, system_prompt: str | None = None, max_retries: int = 5,
                 effort: str | None = None):
        """Initializes the LLM provider.

        Args:
            api_key: The API key for the LLM provider.
            model: The specific model to be used.
            system_prompt: An optional default system prompt for all requests.
            max_retries: How many times to retry temporary failures.
            effort: How hard the model should think (e.g. low, high, xhigh). Providers
                that can't set it ignore it; None leaves the provider's own default.
        """
        self.api_key = api_key
        self.effort = effort
        self.model = model
        self.system_prompt = system_prompt or ""
        self.max_retries = max_retries
        self.logger = logging.getLogger(type(self).__module__)

    def prompt(self,
               user_prompt: str,
               system_prompt: str | None = None,
               *,
               grounding: bool = False,
               prefill: str = "",
               cache_prefix: str = "") -> str:
        """Sends a prompt to the LLM and returns the response text. See `complete`."""
        return self.complete(user_prompt, system_prompt, grounding=grounding, prefill=prefill,
                             cache_prefix=cache_prefix).text

    def complete(self,
                 user_prompt: str,
                 system_prompt: str | None = None,
                 *,
                 grounding: bool = False,
                 prefill: str = "",
                 cache_prefix: str = "") -> Completion:
        """Sends a prompt to the LLM and returns the response, with how it ended.

        Args:
            user_prompt: The user's input prompt.
            system_prompt: A system instruction to override the instance default.
            grounding: Let the model search the web. Raises ValueError if unsupported.
            prefill: Text to start the response with. Raises ValueError if unsupported.
            cache_prefix: The start of user_prompt that repeats across requests. Providers
                that can cache it (with the system prompt) do; the rest ignore it.
        """
        if grounding and not self.supports_grounding:
            raise ValueError(f"{self.model} does not support grounding through {type(self).__name__}.")
        if prefill and not self.supports_prefill:
            raise ValueError(f"{self.model} does not support prefill through {type(self).__name__}.")

        system = self.system_prompt if system_prompt is None else system_prompt
        self.logger.debug(f"{self.model} prompt:\nSYSTEM: {system}\nUSER: {user_prompt}")

        attempt = 0
        while True:
            try:
                extra = {"cache_prefix": cache_prefix} \
                    if cache_prefix and self.supports_caching and user_prompt.startswith(cache_prefix) else {}
                response = self._complete(user_prompt, system, grounding, prefill, **extra)
                self.logger.debug(f"{self.model} response ({response.status}):\n{response.text}")
                if response.status != COMPLETE:
                    self.logger.warning(f"{self.model} response was {response.status}.")
                return response
            except Exception as error:
                if attempt >= self.max_retries or not is_retryable(error):
                    raise
                delay = min(2 ** attempt, MAX_RETRY_DELAY_SECONDS)
                attempt += 1
                self.logger.warning(
                    f"{self.model} request failed ({type(error).__name__}). "
                    f"Retry {attempt}/{self.max_retries} in {delay}s.")
                time.sleep(delay)

    @abstractmethod
    def _complete(self, user_prompt: str, system_prompt: str, grounding: bool, prefill: str) -> Completion:
        """Makes one request to the provider and returns the response."""
