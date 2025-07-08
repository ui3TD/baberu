import requests

from .base import TranscriptionProvider, TranscriptionResult, TranscribedWord, TranscribedSegment

from pathlib import Path
from typing import Any
import logging

class FireworksProvider(TranscriptionProvider):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)

    def transcribe(self, audio_file: Path, **kwargs) -> dict[str, Any]:
        self.logger.info(f"Transcribing audio from {audio_file}...")
        
        lang = kwargs.get("lang", None)

        with open(audio_file, "rb") as f:
            transcription = requests.post(
                "https://audio-prod.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": f},
                data={
                    "model": self.model,
                    "vad_model": "whisperx-pyannet",
                    "alignment_model": "tdnn_ffn",
                    "preprocessing": "none",
                    "language": lang,
                    "temperature": "0",
                    "timestamp_granularities": "word,segment",
                    "audio_window_seconds": "5",
                    "speculation_window_words": "4",
                    "response_format": "verbose_json",
                    "diarize": "true"
                },
            )

        if transcription.status_code == 200:
            print(transcription.json())
        else:
            print(f"Error: {transcription.status_code}", transcription.text)
            raise ConnectionAbortedError
        
        self.logger.debug(f"API response: {transcription.json()}")

        return transcription.json()
    
    @staticmethod
    def parse(json_data: dict[str, Any]) -> TranscriptionResult:
        logger = logging.getLogger(__name__)
        logger.error("Parsing not supported for Fireworks yet.")
        raise RuntimeError
    
    @staticmethod
    def validate(json_data: dict[str, Any]) -> dict[str, Any]:
        return json_data