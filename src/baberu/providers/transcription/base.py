from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
import pydantic
import logging
from typing import Any

class TranscribedWord(pydantic.BaseModel):
    """One token as the service sent it. Times can be missing; normalize fills them in.

    `type` is usually "word", "spacing" or "audio_event", but any other type a
    service adds is kept rather than failing the whole transcript.
    """
    text: str
    start: float | None
    end: float | None
    type: str = "word"
    speaker: str = "Default"
    confidence: float | None = None
    
class TranscribedSegment(pydantic.BaseModel):
    words: List[TranscribedWord]

class TranscriptionResult(pydantic.BaseModel):
    segments: List[TranscribedSegment]
    language: str | None = None

class TranscriptionProvider(ABC):
    """Abstract base class for all transcription providers."""
    # Whether the service takes a `keyterms` list of words to recognise.
    supports_keyterms: bool = False

    def __init__(self, api_key: str, model: str, options: dict[str, Any] | None = None):
        """Initializes the transcription provider.

        Args:
            api_key: The API key for the service.
            model: The specific model to be used.
            options: Request parameters passed to the service as they are, over the provider's defaults.
        """
        self.api_key = api_key
        self.model = model
        self.options: dict[str, Any] = dict(options or {})
        self.logger = logging.getLogger(__name__)
        self.max_size_bytes: int | None = None
    
    def request_options(self) -> dict[str, Any]:
        """The settings sent with each request, as plain JSON, for the job's records."""
        return dict(self.options)

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
    def __init__(self, api_key: str, model: str, options: dict[str, Any] | None = None):
        super().__init__(api_key, model, options)

    @staticmethod
    @abstractmethod
    def to_provider_format(transcript: TranscriptionResult) -> dict[str, Any]:
        pass