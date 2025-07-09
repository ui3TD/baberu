from typing import Any
from pathlib import Path
import logging

import yt_dlp
import ffmpeg
from pydub import AudioSegment

logger = logging.getLogger(__name__)

def download(url: str, 
             output_filename: Path | None = None, 
             download_directory: Path | None = None) -> Path:
    """Downloads an audio file or video from a given URL using yt-dlp.

    Args:
        url: The URL of the audio or video to download.
        output_filename: Optional name for the output file. If not provided,
            the source title is used.
        download_directory: Optional directory to save the file. Defaults to
            the current working directory.

    Returns:
        The path to the downloaded file.
    """
    logger.debug(f"Downloading from {url}...")
    
    # Configure output template
    if download_directory:
        if output_filename:
            output_template = str(download_directory / output_filename.name)
        else:
            output_template = str(download_directory / '%(title)s.%(ext)s')
    else:
        output_template = str(output_filename) if output_filename else '%(title)s.%(ext)s'

    # Configure yt-dlp options
    ydl_opts: dict[str, Any] = {
        'format': 'bestaudio+worstvideo/best',
        'outtmpl': output_template,
        'retries': 500,
        'quiet': False,
        'no_warnings': False,
        'ignoreerrors': False
    }
    
    # Download the video
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info: dict[str, Any] = ydl.extract_info(url, download=True)
        video_file: Path = Path(ydl.prepare_filename(info))
    
    logger.debug(f"Video downloaded successfully to {video_file}")
    return video_file

def extract_audio(video_file: Path,
                  output_file: Path | None = None) -> Path:
    """Extracts the audio stream from a video file without re-encoding.

    Args:
        video_file: Path to the input video file.
        output_file: Optional path for the output audio file. If not provided,
            it is generated from the video name and original audio codec.

    Returns:
        The path to the extracted audio file.
    """
    logger.debug(f"Extracting audio from {video_file}...")

    if not output_file:
        codec_name = get_audio_codec(video_file)
        if not codec_name:
            raise ValueError(
                f"No audio codec found for direct copy extraction. "
            )
        base_path = video_file.parent / video_file.stem
        output_file = base_path.with_suffix("." + codec_name)
    
    # Extract audio using ffmpeg
    try:
        (
            ffmpeg
            .input(str(video_file))
            .output(str(output_file), acodec='copy', vn=None)
            .run(quiet=False, overwrite_output=False)
        )
        logger.debug(f"Audio extracted successfully to {output_file}")
        return output_file
    except ffmpeg.Error as e:
        logger.error(f"Error extracting audio: {str(e)}")
        raise

def get_audio_codec(media_file: Path) -> str | None:
    """Probes a media file to determine its audio codec.

    Args:
        media_file: Path to the input media file.

    Returns:
        The codec name as a string, or None if no audio stream is found.
    """
    logger.debug(f"Probing {media_file} for audio stream...")
    try:
        # Use ffmpeg.probe to get media information as a dictionary
        probe_data = ffmpeg.probe(str(media_file), select_streams='a:0')
        logger.debug(f"Probe data for {media_file}: {probe_data}")

        # The 'streams' list should contain the first audio stream
        if probe_data and 'streams' in probe_data and len(probe_data['streams']) > 0:
            codec_name = probe_data['streams'][0].get('codec_name')
            return codec_name
        else:
            return None
    except ffmpeg.Error as e:
        logger.error(f"Error probing audio stream for {media_file}: {e.stderr.decode('utf-8')}")
        raise

def cut_audio(audio_file: Path, 
              start_time_sec: float, 
              duration_sec: float, 
              output_path: Path) -> Path:
    """Cuts a segment from an audio file and re-encodes it to the Opus codec.

    Args:
        audio_file: Path to the input audio file.
        start_time_sec: The start time of the segment in seconds.
        duration_sec: The duration of the segment in seconds.
        output_path: The path for the output audio segment.

    Returns:
        The path to the created audio segment.
    """
    try:
        audio = AudioSegment.from_file(str(audio_file))

        start_ms = start_time_sec * 1000
        end_ms = (start_time_sec + duration_sec) * 1000

        audio_segment = audio[start_ms:end_ms]

        # Export the segment to Opus format
        audio_segment.export(
            str(output_path),
            format="opus", 
            codec="libopus"
        )
        
        return output_path
    except Exception as e:
        logger.error(f"Error extracting audio segment with pydub: {e}")
        raise

def hardcode_subtitles(video_file: Path,
                       subtitle_file: Path,
                       output_file: Path) -> Path:
    """Burns subtitles into a video, re-encoding to H.264 video and AAC audio.

    Args:
        video_file: Path to the input video file.
        subtitle_file: Path to the subtitle file (e.g., .srt, .ass).
        output_file: Path for the output video.

    Returns:
        The path to the video with hardcoded subtitles.
    """
    logger.debug(f"Hardcoding subtitles from {subtitle_file} into {video_file}...")

    # Hardcode subtitles using ffmpeg with h264 encoding
    try:
        in_file = ffmpeg.input(str(video_file))
        # Explicitly select video (stream 0) and audio (stream 1)
        video_stream = in_file['0']
        audio_stream = in_file['1']

        # Apply the filter ONLY to the video stream
        filtered_video = video_stream.filter('subtitles', subtitle_file.relative_to(Path.cwd(), walk_up=True).as_posix())

        # Output the filtered video AND the original audio
        (
            ffmpeg
            .output(
                filtered_video,      # Use the filtered video stream
                audio_stream,      # Use the original audio stream
                str(output_file),
                vcodec='libx264',
                crf=24,
                acodec='aac',     
                preset='veryfast'
            )
            .run(quiet=False, overwrite_output=False)
        )
        logger.info(f"Subtitles hardcoded successfully to {output_file}")
        return output_file
    except ffmpeg.Error as e:
        logger.error(f"Error hardcoding subtitles: {str(e)}")
        raise

def audio_to_video(image_file: Path,
                   audio_file: Path,
                   output_file: Path) -> Path:
    """Creates a video from a static image and an audio track.

    Args:
        image_file: Path to the input static image file.
        audio_file: Path to the input audio file.
        output_file: Path for the output video file.

    Returns:
        The path to the created video file.
    """
    logger.debug(f"Creating video '{output_file.name}' from image '{image_file.name}' and audio '{audio_file.name}'...")

    try:
        # Define the input streams
        image_stream = ffmpeg.input(str(image_file), loop=1, framerate=1)
        audio_stream = ffmpeg.input(str(audio_file))

        # Build the output command
        (
            ffmpeg
            .output(
                image_stream,
                audio_stream,
                str(output_file),
                vcodec='libx264',
                acodec='copy',
                shortest=None,
                preset='ultrafast',
                tune='stillimage'
            )
            .run(quiet=False, overwrite_output=True)
        )
        logger.info(f"Video created successfully at {output_file}")
        return output_file
    except ffmpeg.Error as e:
        logger.error(f"Error creating video: {e.stderr}")
        raise