from abc import ABC, abstractmethod
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
        """Sends a prompt to the LLM and returns the response as a string.

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
                self.logger.debug(f"{self.model} response:\n{response}")
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
    def _complete(self, user_prompt: str, system_prompt: str, grounding: bool, prefill: str) -> str:
        """Makes one request to the provider and returns the response text."""
