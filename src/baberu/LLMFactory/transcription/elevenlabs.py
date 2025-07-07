from elevenlabs.client import ElevenLabs
from elevenlabs.types import SpeechToTextChunkResponseModel

from .base import TranscriptionProvider, TranscriptionResult, TranscribedWord

from pathlib import Path
from io import BytesIO
# ... other imports

class ScribeProvider(TranscriptionProvider):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.client: ElevenLabs = ElevenLabs(
            api_key=self.api_key,
        )

    def transcribe(self, audio_file: Path, **kwargs) -> TranscriptionResult:
        self.logger.info(f"Transcribing audio from {audio_file}...")
        
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
            request_options = {"timeout_in_seconds": 3600}
        )

        transcription_json = transcription.model_dump()

        # Step 2: Parse the service-specific JSON into our canonical format
        standardized_words = []
        for segment in transcription_json.get('segments', []):
            for word_data in segment.get('words', []):
                 if word_data.get('type') == 'word':
                    standardized_words.append(
                        TranscribedWord(
                            text=word_data['text'],
                            start=word_data['start'],
                            end=word_data['end'],
                            speaker=segment.get('speaker_id')
                        )
                    )
        
        return TranscriptionResult(words=standardized_words)