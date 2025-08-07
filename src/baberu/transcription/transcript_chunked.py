from typing import Any, Generator
from pathlib import Path
import logging
import tempfile

import math
import ffmpeg

from LLMFactory.transcription.base import TranscriptionResult, TranscriptionProvider, TranscribedSegment

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
        probe = ffmpeg.probe(str(audio_file))
        total_duration_s = float(probe['format']['duration'])
    except Exception as e:
        raise ValueError(
            f"Could not load audio file: {audio_file}. "
            "Ensure it is a valid audio format and ffmpeg is installed."
        ) from e

    chunk_duration_ms = _get_chunk_duration(max_size, file_size, total_duration_s)
    
    # This list will store the final, time-adjusted TranscribedSegment objects from all chunks.
    all_segments: list[TranscribedSegment] = []
    detected_language = lang 
    
    with tempfile.TemporaryDirectory() as temp_dir:
        chunk_generator = _chunk_audio(
            audio_file=audio_file, 
            chunk_length_ms=chunk_duration_ms,
            output_dir=Path(temp_dir) 
        )

        for chunk_path, time_offset_s in chunk_generator:
            logger.info(f"Transcribing chunk starting at {time_offset_s:.2f}s...")
            
            response_data: dict[str, Any] = provider.transcribe(chunk_path, lang=lang)

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
    audio_file: Path,
    chunk_length_ms: int,
    output_dir: Path
) -> Generator[tuple[Path, float], None, None]:
    """
    Splits an audio file into chunks using ffmpeg's lossless stream copy and yields each chunk file.

    Args:
        audio_file: The path to the source audio file.
        chunk_length_ms: The desired length of each chunk in milliseconds.
        output_dir: The directory to save temporary chunk files.

    Yields:
        A tuple containing the Path to the chunk audio file and its
        time offset in seconds from the start of the original audio.
    """
    try:
        probe = ffmpeg.probe(str(audio_file))
        total_duration_s = float(probe['format']['duration'])
        total_duration_ms = int(total_duration_s * 1000)
        file_suffix = audio_file.suffix

        if not file_suffix:
            format_name = probe['format'].get('format_name', '').split(',')[0]
            if not format_name:
                 raise ValueError("Could not determine audio format for chunking.")
            file_suffix = f".{format_name}"
            logger.warning(
                f"Audio file has no extension. Inferred format '{format_name}'. "
                f"Using '{file_suffix}' for chunks."
            )
            
    except (ffmpeg.Error, KeyError, ValueError) as e:
        logger.error(f"Failed to probe audio file for chunking: {audio_file}")
        raise e

    for i, start_ms in enumerate(range(0, total_duration_ms, chunk_length_ms)):
        start_s = start_ms / 1000.0
        chunk_duration_s = chunk_length_ms / 1000.0
        
        chunk_file_path = output_dir / f"chunk_{i}{file_suffix}"
        
        logger.debug(f"Exporting chunk {i+1} to {chunk_file_path} (start: {start_s:.2f}s, duration: {chunk_duration_s:.2f}s)")
        
        try:
            (
                ffmpeg
                .input(str(audio_file), ss=start_s)
                .output(
                    str(chunk_file_path),
                    t=chunk_duration_s,  # Specify duration for the chunk
                    c='copy',              # Copy stream to avoid re-encoding (lossless and fast)
                    map_metadata=-1,       # Do not copy metadata to chunks
                )
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            stderr = e.stderr.decode()
            logger.error(f"ffmpeg error while creating chunk {i+1}: {stderr}")
            if "codec copy not supported" in stderr:
                raise IOError(
                    f"Lossless chunking is not supported for this audio format: {file_suffix}. "
                    "You may need to convert it to a different format like M4A or MP3 first."
                ) from e
            raise e
        
        time_offset_s = start_s
        yield chunk_file_path, time_offset_s