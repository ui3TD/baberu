from openai import OpenAI
from openai.types.audio.transcription_verbose import TranscriptionVerbose
from openai.types.audio.transcription_word import TranscriptionWord

from .base import TranscriptionProvider, TranscriptionResult, TranscribedWord, TranscribedSegment

from pathlib import Path
import json

class WhisperProvider(TranscriptionProvider):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.client: OpenAI = OpenAI(
            api_key=self.api_key,
        )

    def transcribe(self, audio_file: Path, **kwargs) -> str:

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
            timestamps_granularities=["word"],
            timeout = 3600
        )
        self.logger.debug(f"API response: {transcription.model_dump_json()}")
        return transcription.model_dump_json()

    def parse(self, json_str: str) -> TranscriptionResult:
        json_data = json.load(json_str)
        transcription = TranscriptionVerbose.model_validate(json_data)
        
        words_list: list[TranscriptionWord] = transcription.words
        
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