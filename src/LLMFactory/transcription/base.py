from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Literal
import pydantic
import logging
from typing import Any

class TranscribedWord(pydantic.BaseModel):
    text: str
    start: float
    end: float
    type: Literal["word", "audio_event", "spacing"] = "word"
    speaker: str = "Default"
    confidence: float | None = None
    
class TranscribedSegment(pydantic.BaseModel):
    words: List[TranscribedWord]

class TranscriptionResult(pydantic.BaseModel):
    segments: List[TranscribedSegment]
    language: str | None = None

class TranscriptionProvider(ABC):
    """Abstract base class for all transcription providers."""
    def __init__(self, api_key: str, model: str):
        """Initializes the LLM provider.

        Args:
            api_key: The API key for the LLM provider.
            model: The specific model to be used.
            system_prompt: An optional default system prompt for all requests.
        """
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.max_size_bytes: int | None = None
    
    @abstractmethod
    def transcribe(self, audio_file: Path, **kwargs) -> dict[str, Any]:
        """
        Transcribes an audio file and returns a Pydantic dict of the json received directly from the provider.
        """
        pass

    @staticmethod
    @abstractmethod
    def parse(json_data: dict[str, Any]) -> TranscriptionResult:
        """
        Parses a transcription of an audio file as a Pydantic dict and returns a standardized TranscriptionResult object.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def validate(json_data: dict[str, Any]) -> dict[str, Any]:
        """
        Validates a transcript's json data as having the expected structure of the provider
        """
        pass


class WritableTranscriptionProvider(TranscriptionProvider):
    """
    An ABC for providers that ALSO support converting a standard transcript
    back into the provider's specific format.
    """
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)

    @staticmethod
    @abstractmethod
    def to_provider_format(transcript: TranscriptionResult) -> dict[str, Any]:
        pass