from openai import OpenAI
from openai.types.audio.transcription_verbose import TranscriptionVerbose
from openai.types.audio.transcription_word import TranscriptionWord

from .base import TranscriptionProvider, TranscriptionResult, TranscribedWord, TranscribedSegment

from pathlib import Path
from typing import Any
import logging

class WhisperProvider(TranscriptionProvider):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.client: OpenAI = OpenAI(
            api_key=self.api_key,
        )

    def transcribe(self, audio_file: Path, **kwargs) -> dict[str, Any]:

        max_size = 25 * 1024 * 1024  # 25MB in bytes
        file_size = audio_file.stat().st_size
        
        self.logger.debug(f"File size: {file_size / (1024 * 1024):.2f} MB")
        
        if file_size > max_size:
            raise ValueError(f"Audio file size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed size of 25MB")

        self.logger.info(f"Transcribing audio from {audio_file}...")
        
        lang = kwargs.get("lang", None)
        
        audio_data= open(audio_file, "rb")

        transcription: TranscriptionVerbose = self.client.audio.transcriptions.create(
            file=audio_data,
            model=self.model,
            language=lang,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            timeout = 3600
        )
        self.logger.debug(f"API response: {transcription.model_dump()}")
        return transcription.model_dump()

    @staticmethod
    def parse(json_data: dict[str, Any]) -> TranscriptionResult:
        logger = logging.getLogger(__name__)
        if not "words" in json_data:
            logger.error("OpenAI JSON validation failed. Key 'words' does not exist.")
            raise ValueError
        
        words_list: list[TranscriptionWord] = json_data["words"]
        
        # Convert to TranscribedWord objects
        transcribed_words = [
            TranscribedWord(
                text=word.word,
                start=word.start or 0.0,
                end=word.end or 0.0
            )
            for word in words_list
        ]
        
        segment = TranscribedSegment(
            words=transcribed_words
        )
        
        return TranscriptionResult(segments=[segment])
    
    @staticmethod
    def validate(json_data: dict[str, Any]) -> dict[str, Any]:
        logger = logging.getLogger(__name__)
        if not "words" in json_data:
            logger.error("OpenAI JSON validation failed. Key 'words' does not exist.")
            raise ValueError
        
        logger.warning("OpenAI JSON validation bypassed due to low reliability.")
        return json_data