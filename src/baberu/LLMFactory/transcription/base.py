from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
import pydantic
import logging

class TranscribedWord(pydantic.BaseModel):
    text: str
    start: float
    end: float
    type: str = "text"
    speaker: str | None = None
    confidence: float | None = None
    
class TranscribedSegment(pydantic.BaseModel):
    words: List[TranscribedWord]
    speaker: str | None = None

class TranscriptionResult(pydantic.BaseModel):
    segments: List[TranscribedSegment]
    # You can add other metadata here if needed

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
    
    @abstractmethod
    def transcribe(self, audio_file: Path, **kwargs) -> TranscriptionResult:
        """
        Transcribes an audio file and returns a standardized TranscriptionResult object.
        """
        pass