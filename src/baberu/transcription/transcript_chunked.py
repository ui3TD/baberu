from typing import Any, Generator
from pathlib import Path
import logging
import tempfile

import math
from pydub import AudioSegment

from baberu.LLMFactory.factory import AIToolFactory
from baberu.LLMFactory.transcription.base import TranscriptionResult, TranscriptionProvider

logger = logging.getLogger(__name__)

def transcribe_in_chunks(audio_file: Path, model: str, lang: str | None) -> dict[str, Any]:
    
    transcription_provider = AIToolFactory.get_transcription_provider(model)
    max_size = transcription_provider.max_size_bytes
    file_size = audio_file.stat().st_size
    
    logger.debug(f"File size: {file_size / (1024 * 1024):.2f} MB")

    if not max_size:
        logger.error("Chunking is not applicable. Model has no max size.")
        raise ValueError
    elif file_size <= max_size:
        logger.error(f"Chunking is not applicable. File size {file_size / (1024 * 1024):.2f} is less than model's max size {max_size / (1024 * 1024):.2f}.")
        raise ValueError

    logger.info(f"Transcribing audio from {audio_file}...")

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
    
    logger.info(
        f"Audio duration: {total_duration_s:.2f}s. "
        f"Splitting into {num_chunks} chunks of ~{chunk_duration_ms / 1000:.2f}s each."
    )

    full_text, all_segments, all_words = [], [], []
    segment_id_counter = 0
    detected_language = lang 
    
    # The generator handles creating and cleaning up temporary chunk files
    chunk_generator = _chunk_audio(
        audio, 
        audio_file_format=audio_file.suffix.lstrip('.'), 
        chunk_length_ms=chunk_duration_ms
    )

    for chunk_path, time_offset_s in chunk_generator:
        logger.info(f"Transcribing chunk starting at {time_offset_s:.2f}s...")
        with open(chunk_path, "rb") as chunk_data:
            chunk_transcription = transcription_provider.transcribe(chunk_data, lang=lang)
        
        response_data = chunk_transcription.model_dump()
        
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

    logger.info("Successfully combined transcriptions from all chunks.")
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