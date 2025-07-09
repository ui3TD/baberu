from typing import Any, Generator
from pathlib import Path
import logging
import tempfile

import math
from pydub import AudioSegment

from baberu.LLMFactory.transcription.base import TranscriptionResult, TranscriptionProvider, TranscribedSegment, TranscribedWord

logger = logging.getLogger(__name__)

def transcribe_in_chunks(audio_file: Path, provider: TranscriptionProvider, lang: str | None) -> TranscriptionResult:
    """
    Transcribes a large audio file by splitting it into chunks and processing each one.

    Args:
        audio_file: Path to the audio file.
        provider: The transcription provider instance.
        lang: The language of the audio. If None, it will be auto-detected.

    Returns:
        A structured TranscriptionResult object.
    """
    max_size = provider.max_size_bytes
    file_size = audio_file.stat().st_size
    
    logger.debug(f"File size: {file_size / (1024 * 1024):.2f} MB")

    if not max_size or file_size <= max_size:
        reason = "Model has no max size" if not max_size else f"File size {file_size / (1024 * 1024):.2f} MB is not greater than model's max size {max_size / (1024 * 1024):.2f} MB"
        logger.error(f"Chunking is not applicable. {reason}.")
        raise ValueError("Chunking is not applicable for this file and provider.")

    logger.info(f"Audio file is large, transcribing in chunks: {audio_file}")

    try:
        audio = AudioSegment.from_file(audio_file)
        total_duration_s = len(audio) / 1000.0
    except Exception as e:
        raise ValueError(
            f"Could not load audio file: {audio_file}. "
            "Ensure it is a valid audio format and ffmpeg is installed."
        ) from e

    chunk_duration_ms = _get_chunk_duration(max_size, file_size, total_duration_s)
    
    # This list will store the final, time-adjusted TranscribedSegment objects from all chunks.
    all_segments: list[TranscribedSegment] = []
    detected_language = lang 
    
    chunk_generator = _chunk_audio(
        audio, 
        audio_file_format=audio_file.suffix.lstrip('.'), 
        chunk_length_ms=chunk_duration_ms
    )

    for chunk_path, time_offset_s in chunk_generator:
        logger.info(f"Transcribing chunk starting at {time_offset_s:.2f}s...")
        
        # The provider returns raw data, which we immediately parse into our standard model.
        with open(chunk_path, "rb") as chunk_data:
            response_data: dict[str, Any] = provider.transcribe(chunk_data, lang=lang)

        transcript: TranscriptionResult = provider.parse(response_data)
        
        if detected_language is None:
            detected_language = transcript.language

        for segment in transcript.segments:
            adjusted_words = []
            for word in segment.words:
                # Create a new word with the global timestamp.
                adjusted_word = word.model_copy(
                    update={
                        "start": word.start + time_offset_s,
                        "end": word.end + time_offset_s,
                    }
                )
                adjusted_words.append(adjusted_word)
            
            # Only add segments that contain words after processing.
            if adjusted_words:
                all_segments.append(TranscribedSegment(words=adjusted_words))
            
    # --- Assemble the final result from the collected segments ---
    logger.info("Successfully combined transcriptions from all chunks.")
    
    return TranscriptionResult(
        segments=all_segments,
        language=detected_language
    )

def _get_chunk_duration(max_size: int, file_size: int, total_duration_s: float) -> int:
    
    # Calculate average bytes per second to estimate duration for max_size
    # Add a 5% safety margin to stay safely under the limit.
    safe_max_size = max_size * 0.95 
    avg_bytes_per_second = file_size / total_duration_s
    
    # This is the maximum duration a chunk can be to not exceed the size limit
    max_duration_per_chunk_s = safe_max_size / avg_bytes_per_second
    
    # Now, create evenly distributed chunks
    num_chunks = math.ceil(total_duration_s / max_duration_per_chunk_s)
    chunk_duration_ms = math.ceil((total_duration_s / num_chunks) * 1000)
    
    logger.debug(
        f"Audio duration: {total_duration_s:.2f}s. "
        f"Splitting into {num_chunks} chunks of ~{chunk_duration_ms / 1000:.2f}s each."
    )
    return chunk_duration_ms

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

