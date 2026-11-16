from elevenlabs.client import ElevenLabs
from elevenlabs.types import (
    SpeechToTextChunkResponseModel, 
    ExportOptions_SegmentedJson, 
)

from .base import TranscriptionProvider, TranscriptionResult, TranscribedWord, TranscribedSegment

from pathlib import Path
from io import BytesIO
from itertools import groupby
from typing import Any
import json
import logging

logger = logging.getLogger(__name__)

class ScribeProvider(TranscriptionProvider):
    supports_keyterms = True

    # What baberu has always asked for. `options` from config.yaml go on top, so any
    # parameter of the speech-to-text API (num_speakers, no_verbatim, keyterms,
    # numbers_format...) can be tried without a code change.
    DEFAULTS: dict[str, Any] = {
        "tag_audio_events": False,
        "diarize": True,
        "timestamps_granularity": "word",
        "segment_on_silence_longer_than_s": 2.0,  # for segmented_json, which marks line breaks
    }

    def __init__(self, api_key: str, model: str, options: dict[str, Any] | None = None):
        super().__init__(api_key, model, options)
        self.client: ElevenLabs = ElevenLabs(
            api_key=self.api_key,
        )

    def request_options(self) -> dict[str, Any]:
        return {**self.DEFAULTS, **self.options}

    def transcribe(self, audio_file: Path, **kwargs) -> dict[str, Any]:
        self.logger.debug(f"Transcribing audio from {audio_file}...")
        
        lang = kwargs.get("lang", None)
        options = self.request_options()
        silence = options.pop("segment_on_silence_longer_than_s")
        
        with open(audio_file, 'rb') as f:
            audio_data: BytesIO = BytesIO(f.read())

        transcription: SpeechToTextChunkResponseModel = self.client.speech_to_text.convert(
            file=audio_data,
            model_id=self.model,
            language_code=lang,
            additional_formats= [ExportOptions_SegmentedJson(
                include_speakers=True,
                include_timestamps=True,
                segment_on_silence_longer_than_s=silence
            )],
            request_options = {"timeout_in_seconds": 3600},
            **options,
        )
        self.logger.debug(f"ElevenLabs API response: {transcription.model_dump_json()}")

        return transcription.model_dump()
    
    @staticmethod
    def parse(json_data: dict[str, Any]) -> TranscriptionResult:
        """Reads a response. Tokens come from `words`, which keeps spacing; segments only mark the breaks.

        segmented_json drops spacing tokens, so its own words would run English
        together ("Thanks.Okay.So,"). When it doesn't line up with `words`, the
        words are grouped by speaker instead.
        """
        transcription = SpeechToTextChunkResponseModel.model_validate(json_data)
        words = [ScribeProvider._word(word.model_dump()) for word in transcription.words]

        segment_sizes = ScribeProvider._segment_sizes(transcription)
        segments = ScribeProvider._split_by_sizes(words, segment_sizes) if segment_sizes else None
        if segments is None:
            if segment_sizes:
                logger.warning("segmented_json doesn't match the word list; grouping words by speaker instead.")
            segments = [TranscribedSegment(words=list(group)) for _, group in groupby(words, key=lambda w: w.speaker)]

        return TranscriptionResult(segments=segments, language=transcription.language_code)

    @staticmethod
    def _word(data: dict[str, Any]) -> TranscribedWord:
        return TranscribedWord(
            text=data.get("text") or "",
            start=data.get("start"),
            end=data.get("end"),
            type=data.get("type") or "word",
            speaker=data.get("speaker_id") or "Default",
            confidence=data.get("logprob"),  # 0.0 is a real value; only a missing one is None
        )

    @staticmethod
    def _segment_sizes(transcription: SpeechToTextChunkResponseModel) -> list[tuple[int, list[str]]]:
        """The non-spacing token texts of each segment in segmented_json, if it was returned."""
        formats_list = transcription.additional_formats or []
        segmented = next((f for f in formats_list if f and f.requested_format == "segmented_json"), None)
        if not segmented or not isinstance(segmented.content, str) or not segmented.content:
            return []
        sizes = []
        for segment in json.loads(segmented.content).get("segments", []):
            texts = [w.get("text") or "" for w in segment.get("words", []) if w.get("type") != "spacing"]
            sizes.append((len(texts), texts))
        return sizes

    @staticmethod
    def _split_by_sizes(words: list[TranscribedWord], sizes: list[tuple[int, list[str]]]) -> list[TranscribedSegment] | None:
        """Cuts `words` where segmented_json breaks. Spacing stays with the segment before it."""
        spoken = [w for w in words if w.type != "spacing"]
        if [w.text for w in spoken] != [text for _, texts in sizes for text in texts]:
            return None

        segments: list[TranscribedSegment] = []
        current: list[TranscribedWord] = []
        remaining = iter(size for size, _ in sizes if size)
        target = next(remaining, 0)
        counted = 0
        for word in words:
            if word.type != "spacing" and counted == target and current:
                segments.append(TranscribedSegment(words=current))
                current, counted, target = [], 0, next(remaining, 0)
            current.append(word)
            counted += word.type != "spacing"
        if current:
            segments.append(TranscribedSegment(words=current))
        return segments
    
    @staticmethod
    def validate(json_data: dict[str, Any]) -> dict[str, Any]:
        SpeechToTextChunkResponseModel.model_validate(json_data)
        return json_data