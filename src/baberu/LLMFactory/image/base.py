from abc import ABC, abstractmethod
import logging
from pathlib import Path

class ImageProvider(ABC):
    """Abstract base class for all Large Language Model providers."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def prompt(self, prompt: str, file_path: Path, **kwargs) -> None:
        """Sends a prompt to the LLM and returns an image.

        Args:
            prompt: The user's input prompt
            **kwargs: Provider-specific parameters (e.g., prefill, grounding, etc.)
        """
        pass