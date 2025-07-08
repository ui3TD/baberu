from openai import OpenAI
from openai.types.audio.transcription_verbose import TranscriptionVerbose
from openai.types.audio.transcription_word import TranscriptionWord

from .base import TranscriptionProvider, TranscriptionResult, TranscribedWord, TranscribedSegment

from pathlib import Path
from typing import Any, Generator
import tempfile
import logging
import math
from pydub import AudioSegment

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

        self.logger.info(f"Transcribing audio from {audio_file}...")
        
        lang = kwargs.get("lang", None)
        
        if file_size < max_size:
            audio_data= open(audio_file, "rb")
            transcription: TranscriptionVerbose = self.client.audio.transcriptions.create(
                file=audio_data,
                model=self.model,
                language=lang,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                timeout = 3600
            )
            self.logger.debug(f"API response: {transcription.model_dump()}")
            return transcription.model_dump()
        
        self.logger.info(f"File size exceeds 25MB. Splitting and processing in chunks...")

        try:
            audio = AudioSegment.from_file(audio_file)
            total_duration_s = len(audio) / 1000.0
        except Exception as e:
            raise ValueError(
                f"Could not load audio file: {audio_file}. "
                "Ensure it is a valid audio format and ffmpeg is installed."
            ) from e

        # Calculate average bytes per second to estimate duration for max_size
        # Add a 5% safety margin to stay safely under the limit.
        safe_max_size = max_size * 0.95 
        avg_bytes_per_second = file_size / total_duration_s
        
        # This is the maximum duration a chunk can be to not exceed the size limit
        max_duration_per_chunk_s = safe_max_size / avg_bytes_per_second
        
        # Now, create evenly distributed chunks
        num_chunks = math.ceil(total_duration_s / max_duration_per_chunk_s)
        chunk_duration_ms = math.ceil((total_duration_s / num_chunks) * 1000)
        
        self.logger.info(
            f"Audio duration: {total_duration_s:.2f}s. "
            f"Splitting into {num_chunks} chunks of ~{chunk_duration_ms / 1000:.2f}s each."
        )

        full_text, all_segments, all_words = [], [], []
        segment_id_counter = 0
        detected_language = lang 
        
        # The generator handles creating and cleaning up temporary chunk files
        chunk_generator = self._chunk_audio(
            audio, 
            audio_file_format=audio_file.suffix.lstrip('.'), 
            chunk_length_ms=chunk_duration_ms
        )

        for chunk_path, time_offset_s in chunk_generator:
            self.logger.info(f"Transcribing chunk starting at {time_offset_s:.2f}s...")
            with open(chunk_path, "rb") as chunk_data:
                chunk_transcription = self.client.audio.transcriptions.create(
                    file=chunk_data,
                    model=self.model,
                    language=lang,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                    timeout=3600
                )
            
            response_data = chunk_transcription.model_dump()
            self.logger.debug(f"Chunk API response processed.")
            
            # If language is not specified, use the one from the first chunk
            if detected_language is None:
                detected_language = response_data.get('language')

            full_text.append(response_data['text'])
            
            for segment in response_data.get('segments', []):
                segment['start'] += time_offset_s
                segment['end'] += time_offset_s
                segment['id'] = segment_id_counter
                all_segments.append(segment)
                segment_id_counter += 1
            
            for word in response_data.get('words', []):
                word['start'] += time_offset_s
                word['end'] += time_offset_s
                all_words.append(word)
        
        # Combine all results into a single dictionary
        total_duration = all_segments[-1]['end'] if all_segments else 0
        combined_result = {
            "text": " ".join(full_text).strip(),
            "segments": all_segments,
            "words": all_words,
            "language": detected_language,
            "duration": total_duration,
            "task": "transcribe",
            "usage": {
                "duration": total_duration,
                "type": "duration",
                "seconds": int(round(total_duration)) # schema requires integer
            }
        }

        self.logger.info("Successfully combined transcriptions from all chunks.")
        return combined_result
    
    def _chunk_audio(
        self,
        audio: AudioSegment,
        audio_file_format: str,
        chunk_length_ms: int
    ) -> Generator[tuple[Path, float], None, None]:
        """
        Yields temporary chunk files from a pydub AudioSegment object.

        This is a generator function that creates and yields one chunk at a time
        within a temporary directory, which is cleaned up automatically after
        the generator is exhausted.

        Args:
            audio: The loaded pydub.AudioSegment object.
            audio_file_format: The original file format (e.g., "mp3", "wav").
            chunk_length_ms: The desired length of each chunk in milliseconds.

        Yields:
            A tuple containing the Path to the temporary chunk file and its
            time offset in seconds from the start of the original audio.
        """
        total_duration_ms = len(audio)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            for i, start_ms in enumerate(range(0, total_duration_ms, chunk_length_ms)):
                end_ms = start_ms + chunk_length_ms
                chunk = audio[start_ms:end_ms]
                
                chunk_file_path = temp_dir_path / f"chunk_{i}.{audio_file_format or 'mp3'}"
                
                self.logger.debug(f"Exporting chunk {i+1} to {chunk_file_path}")
                chunk.export(chunk_file_path, format=audio_file_format)
                
                time_offset_s = start_ms / 1000.0
                yield chunk_file_path, time_offset_s

    @staticmethod
    def parse(json_data: dict[str, Any]) -> TranscriptionResult:
        logger = logging.getLogger(__name__)
        if "text" not in json_data:
            logger.error("OpenAI JSON validation failed. Key 'text' does not exist.")
            raise ValueError
        if "segments" not in json_data:
            logger.error("OpenAI JSON validation failed. Key 'segments' does not exist.")
            raise ValueError
        if "words" not in json_data:
            logger.error("OpenAI JSON validation failed. Key 'words' does not exist.")
            raise ValueError
        
        # Get the master lists of segments and all words
        segments_data = json_data.get("segments", [])
        all_words_data = json_data.get("words", [])
        
        final_segments: list[TranscribedSegment] = []

        # 1. Main loop: Iterate through each segment from the JSON
        for segment_data in segments_data:
            segment_text = segment_data.get("text","").strip() # Use the segment's specific text
            segment_start = segment_data.get("start", 0.0)
            segment_end = segment_data.get("end", 0.0)

            # Filter the master word list to get words belonging to the current segment
            # A word belongs to a segment if its start time is within the segment's time range.
            words_for_segment = [
                word for word in all_words_data 
                if word.get('start', -1) >= segment_start and word.get('start', -1) <= segment_end
            ]

            reconstructed_words: list[TranscribedWord] = []
            
            text_cursor = 0

            # 2. Inner loop
            for word_data in words_for_segment:
                word_text = word_data.get('word', "")
                word_start = word_data.get('start', 0.0)
                word_end = word_data.get('end', 0.0)

                try:
                    # Find the word's position in the current segment's text
                    word_start_pos = segment_text.index(word_text, text_cursor)
                except ValueError:
                    # It's possible for whisper to have slight transcription mismatches.
                    # We'll log it and skip the punctuation handling for this gap.
                    logger.debug(
                        f"Word overlaps segment timing: Word '{word_text}' overlaps with Segment '{segment_text}' at position {text_cursor}. Skipping."
                    )
                    continue

                # 3. Capture any leading punctuation and whitespace
                if word_start_pos > text_cursor:
                    # The time of the previous word, defaulting to the segment's start time
                    prev_end_time = reconstructed_words[-1].end if reconstructed_words else segment_start
                    
                    interstitial_text = segment_text[text_cursor:word_start_pos]
                    
                    punctuation_to_append = ""
                    space_present = False

                    # Separate punctuation from whitespace
                    for char in interstitial_text:
                        if char.isspace():
                            space_present = True
                            if punctuation_to_append:
                                leading_word = TranscribedWord(
                                    text=punctuation_to_append,
                                    start=prev_end_time,
                                    end=prev_end_time
                                )
                                reconstructed_words.append(leading_word)
                            
                            leading_word = TranscribedWord(
                                text=char, start=prev_end_time, end=prev_end_time, type="spacing"
                            )
                            reconstructed_words.append(leading_word)
                            punctuation_to_append = ""
                        else:
                            punctuation_to_append += char

                    punctuation_time = word_start if space_present else prev_end_time
                    if punctuation_to_append:
                        leading_word = TranscribedWord(
                            text=punctuation_to_append, start=punctuation_time, end=punctuation_time
                        )
                        reconstructed_words.append(leading_word)

                # 4. Add the actual timed word
                timed_word = TranscribedWord(
                    text=word_text, start=word_start, end=word_end
                )
                reconstructed_words.append(timed_word)

                # 5. Update the cursor
                text_cursor = word_start_pos + len(word_text)

            # 6. After the loop, capture any remaining trailing characters
            if text_cursor < len(segment_text):
                trailing_text = segment_text[text_cursor:]
                
                last_word_end_time = reconstructed_words[-1].end if reconstructed_words else segment_end
                
                trailing_word = TranscribedWord(
                    text=trailing_text, start=last_word_end_time, end=last_word_end_time
                )
                reconstructed_words.append(trailing_word)
            
            # 7. Create a TranscribedSegment with the words from this iteration and add to our final list
            segment = TranscribedSegment(words=reconstructed_words)
            final_segments.append(segment)
            
        return TranscriptionResult(segments=final_segments)
    
    @staticmethod
    def validate(json_data: dict[str, Any]) -> dict[str, Any]:
        logger = logging.getLogger(__name__)
        if "text" not in json_data:
            logger.error("OpenAI JSON validation failed. Key 'text' does not exist.")
            raise ValueError
        if "segments" not in json_data:
            logger.error("OpenAI JSON validation failed. Key 'segments' does not exist.")
            raise ValueError
        if "words" not in json_data:
            logger.error("OpenAI JSON validation failed. Key 'words' does not exist.")
            raise ValueError
        
        logger.warning("OpenAI JSON validation bypassed due to low reliability.")
        return json_data