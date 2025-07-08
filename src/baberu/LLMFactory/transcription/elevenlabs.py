from elevenlabs.client import ElevenLabs
from elevenlabs.types import SpeechToTextChunkResponseModel, SpeechToTextWordResponseModel

from .base import TranscriptionProvider, TranscriptionResult, TranscribedWord, TranscribedSegment

from pathlib import Path
from io import BytesIO
from itertools import groupby
from typing import Any
import json

class ScribeProvider(TranscriptionProvider):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.client: ElevenLabs = ElevenLabs(
            api_key=self.api_key,
        )

    def transcribe(self, audio_file: Path, **kwargs) -> dict[str, Any]:
        self.logger.info(f"Transcribing audio from {audio_file}...")
        
        lang = kwargs.get("lang", None)
        
        with open(audio_file, 'rb') as f:
            audio_data: BytesIO = BytesIO(f.read())

        segmented_json = {
            "format": "segmented_json",
            "segment_on_silence_longer_than_s": 3
        }

        transcription: SpeechToTextChunkResponseModel = self.client.speech_to_text.convert(
            file=audio_data,
            model_id=self.model,
            tag_audio_events=False,
            language_code=lang,
            diarize=True,
            diarization_threshold=0.1,
            timestamps_granularity="word",
            additional_formats= json.dumps([segmented_json]),
            request_options = {"timeout_in_seconds": 3600}
        )
        self.logger.debug(f"API response: {transcription.model_dump_json()}")

        return transcription.model_dump_json()
    
    def parse(self, json_data: dict[str, Any]) -> TranscriptionResult:
        transcription = SpeechToTextChunkResponseModel.model_validate(json_data)
        grouped_words = groupby(transcription.words, key=lambda w: w.speaker_id)
        segments: list[TranscribedSegment] = []
        for speaker_id, words_group in grouped_words:
            words_list: list[SpeechToTextWordResponseModel] = list(words_group)
            
            # Convert to TranscribedWord objects
            transcribed_words = [
                TranscribedWord(
                    text=word.text,
                    start=word.start or 0.0,
                    end=word.end or 0.0,
                    type=word.type,
                    speaker=speaker_id or None,
                    confidence=word.logprob or None
                )
                for word in words_list
            ]
            
            segment = TranscribedSegment(
                words=transcribed_words,
                speaker=speaker_id
            )
            segments.append(segment)
        
        return TranscriptionResult(segments=segments)