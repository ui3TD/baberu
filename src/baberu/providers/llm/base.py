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

    Provider SDKs share no base exception class, so this goes by the HTTP
    status they report, or by the name of the network error they raise.
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

    def __init__(self, api_key: str, model: str, system_prompt: str | None = None, max_retries: int = 5):
        """Initializes the LLM provider.

        Args:
            api_key: The API key for the LLM provider.
            model: The specific model to be used.
            system_prompt: An optional default system prompt for all requests.
            max_retries: How many times to retry temporary failures.
        """
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt or ""
        self.max_retries = max_retries
        self.logger = logging.getLogger(type(self).__module__)

    def prompt(self,
               user_prompt: str,
               system_prompt: str | None = None,
               *,
               grounding: bool = False,
               prefill: str = "") -> str:
        """Sends a prompt to the LLM and returns the response text. See `complete`."""
        return self.complete(user_prompt, system_prompt, grounding=grounding, prefill=prefill).text

    def complete(self,
                 user_prompt: str,
                 system_prompt: str | None = None,
                 *,
                 grounding: bool = False,
                 prefill: str = "") -> Completion:
        """Sends a prompt to the LLM and returns the response, with how it ended.

        Args:
            user_prompt: The user's input prompt.
            system_prompt: A system instruction to override the instance default.
            grounding: Let the model search the web. Raises ValueError if unsupported.
            prefill: Text to start the response with. Raises ValueError if unsupported.
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
                response = self._complete(user_prompt, system, grounding, prefill)
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
