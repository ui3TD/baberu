try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs.types import (
        SpeechToTextChunkResponseModel, 
        SpeechToTextWordResponseModel, 
        ExportOptions_SegmentedJson, 
        AdditionalFormatResponseModel
    )
except ImportError:
    raise ImportError(
        "The 'elevenlabs' package is required to use ElevenLabs models. "
        "Install it with: pip install elevenlabs"
    )

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
        self.logger.debug(f"Transcribing audio from {audio_file}...")
        
        lang = kwargs.get("lang", None)
        
        with open(audio_file, 'rb') as f:
            audio_data: BytesIO = BytesIO(f.read())

        transcription: SpeechToTextChunkResponseModel = self.client.speech_to_text.convert(
            file=audio_data,
            model_id=self.model,
            tag_audio_events=False,
            language_code=lang,
            diarize=True,
            diarization_threshold=0.1,
            timestamps_granularity="word",
            additional_formats= [ExportOptions_SegmentedJson(
                include_speakers=True,
                include_timestamps=True,
                segment_on_silence_longer_than_s=2.0
            )],
            request_options = {"timeout_in_seconds": 3600}
        )
        self.logger.debug(f"ElevenLabs API response: {transcription.model_dump_json()}")

        return transcription.model_dump()
    
    @staticmethod
    def parse(json_data: dict[str, Any]) -> TranscriptionResult:
        transcription = SpeechToTextChunkResponseModel.model_validate(json_data)

        is_segmented_json = False
        formats_list = transcription.additional_formats or []
        segmented_format = next((f for f in formats_list if f.requested_format == "segmented_json"), None)
        if segmented_format:
            content_str = segmented_format.content
            if not content_str or not isinstance(content_str, str):
                is_segmented_json = False
            else:
                is_segmented_json = True

        segments: list[TranscribedSegment] = []

        if is_segmented_json:
            segmented_data = json.loads(content_str)
            
            for segment_data in segmented_data.get("segments", []):
                words_data_list = segment_data.get("words", [])
                
                transcribed_words = [
                    TranscribedWord(
                        text=word_data.get("text", ""),
                        start=word_data.get("start", 0.0),
                        end=word_data.get("end", 0.0),
                        type=word_data.get("type", "word"),
                        speaker=word_data.get("speaker_id"),
                        confidence=word_data.get("logprob")
                    )
                    for word_data in words_data_list
                ]
                
                segment = TranscribedSegment(
                    words=transcribed_words
                )
                segments.append(segment)
        else:   
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
                    words=transcribed_words
                )
                segments.append(segment)
        
        return TranscriptionResult(segments=segments)
    
    @staticmethod
    def validate(json_data: dict[str, Any]) -> dict[str, Any]:
        SpeechToTextChunkResponseModel.model_validate(json_data)
        return json_data