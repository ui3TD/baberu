from abc import ABC, abstractmethod
import logging

class LLMProvider(ABC):
    """Abstract base class for all Large Language Model providers."""
    
    def __init__(self, api_key: str, model: str, system_prompt: str = None):
        """Initializes the LLM provider.

        Args:
            api_key: The API key for the LLM provider.
            model: The specific model to be used.
            system_prompt: An optional default system prompt for all requests.
        """
        if system_prompt is None:
            system_prompt = ""

        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def prompt(self, user_prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        """Sends a prompt to the LLM and returns the response as a string.

        Args:
            user_prompt: The user's input prompt.
            system_prompt: A system instruction to override the instance default.
            **kwargs: Provider-specific parameters (e.g., prefill, grounding).
        """
        pass