from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
import pydantic # Or dataclasses

# Define a canonical (standard) data model for a word
class TranscribedWord(pydantic.BaseModel):
    text: str
    start: float
    end: float
    speaker: int | None = None
    confidence: float | None = None

# Define the canonical output for any transcription provider
class TranscriptionResult(pydantic.BaseModel):
    words: List[TranscribedWord]
    # You can add other metadata here if needed

class TranscriptionProvider(ABC):
    """Abstract base class for all transcription providers."""
    
    @abstractmethod
    def transcribe(self, audio_file: Path, num_speakers: int, lang: str) -> TranscriptionResult:
        """
        Transcribes an audio file and returns a standardized TranscriptionResult object.
        """
        pass