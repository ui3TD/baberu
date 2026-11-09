from typing import Any, Tuple
from pathlib import Path
import logging
import shutil
import tempfile

import yt_dlp
import ffmpeg

from baberu.tools.file_utils import atomic_path

logger = logging.getLogger(__name__)

# yt-dlp's usual naming, with a cap so long titles fit within file name limits.
DOWNLOAD_NAME = "%(title).120B [%(id)s].%(ext)s"

def _download_options(directory: Path | None = None) -> dict[str, Any]:
    options: dict[str, Any] = {
        'format': 'bestaudio+worstvideo/best',
        'outtmpl': DOWNLOAD_NAME,
        'retries': 500,
        'quiet': True,
        'no_warnings': False,
        'ignoreerrors': False,
        'logger': logger,
    }
    if directory:
        options['paths'] = {'home': str(directory)}
    return options


def download(url: str, directory: Path) -> Path:
    """Downloads audio or video from a URL into a folder, returning the file's path."""
    with yt_dlp.YoutubeDL(_download_options(directory)) as ydl:
        info: dict[str, Any] = ydl.extract_info(url, download=True) or {}
        if not info:
            raise RuntimeError(f"Failed to download: {url}")
        downloads = info.get("requested_downloads") or []
        return Path(downloads[0]["filepath"]) if downloads else Path(ydl.prepare_filename(info))


def probe_url(url: str) -> dict[str, Any]:
    """Reads a URL's title and duration without downloading it."""
    with yt_dlp.YoutubeDL(_download_options()) as ydl:
        return ydl.extract_info(url, download=False) or {}


def duration_seconds(media_file: Path) -> float | None:
    """The length of a media file, or None if it can't be read."""
    try:
        return float(ffmpeg.probe(str(media_file))["format"]["duration"])
    except (ffmpeg.Error, KeyError, ValueError, OSError):
        return None


def extract_audio(video_file: Path, output_file: Path) -> Path:
    """Copies the audio stream out of a video without re-encoding."""
    with atomic_path(output_file) as temporary:
        run_output: Tuple[bytes, bytes] = (
            ffmpeg
            .input(str(video_file))
            .output(str(temporary), acodec='copy', vn=None)
            .run(quiet=True, overwrite_output=True)
        )
    _, err = run_output
    if err:
        logger.debug(err.decode())
    return output_file

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
        probe_data: dict[str, Any] = ffmpeg.probe(str(media_file), select_streams='a:0')
        logger.debug(f"Probe data for {media_file}: {probe_data}")

        # The 'streams' list should contain the first audio stream
        if probe_data and 'streams' in probe_data and len(probe_data['streams']) > 0:
            codec_name = probe_data['streams'][0].get('codec_name')
            return codec_name
        else:
            return None
    except ffmpeg.Error as e:
        logger.debug(f"Error probing audio stream for {media_file}: {e.stderr.decode('utf-8')}")
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
        run_output: Tuple[bytes, bytes] = (
            ffmpeg
            .input(str(audio_file), ss=start_time_sec, t=duration_sec)
            .output(str(output_path), acodec="libopus", loglevel="error")
            .run(quiet=True, overwrite_output=True)
        )
        _, err = run_output
        if err:
            logger.debug(err.decode())
        return output_path
    except ffmpeg.Error as e:
        logger.debug(f"Error extracting audio segment with ffmpeg-python: {e.stderr.decode()}")
        raise

def hardcode_subtitles(video_file: Path,
                       subtitle_file: Path,
                       output_file: Path) -> Path:
    """Burns subtitles into a video, re-encoding to H.264 video and AAC audio.

    ffmpeg's subtitle filter can't reliably read paths containing drive
    letters or special characters, so it is given a plainly named temporary
    copy under the current directory.

    Args:
        video_file: Path to the input video file.
        subtitle_file: Path to the subtitle file (e.g., .srt, .ass).
        output_file: Path for the output video.

    Returns:
        The path to the video with hardcoded subtitles.
    """
    with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix="baberu_subs_") as temp_dir:
        temp_copy = Path(temp_dir) / f"subs{subtitle_file.suffix}"
        shutil.copy(subtitle_file, temp_copy)
        with atomic_path(output_file) as temporary:
            _burn_subtitles(video_file, temp_copy.relative_to(Path.cwd()), temporary)
    return output_file


def _burn_subtitles(video_file: Path,
                    subtitle_file: Path,
                    output_file: Path) -> Path:
    """Runs ffmpeg with a subtitle path that needs no escaping."""
    logger.debug(f"Hardcoding subtitles from {subtitle_file} into {video_file}...")

    # Hardcode subtitles using ffmpeg with h264 encoding
    try:
        in_file = ffmpeg.input(str(video_file))
        video_stream = in_file.video
        audio_stream = in_file.audio

        # Apply the filter ONLY to the video stream
        filtered_video = video_stream.filter('subtitles', subtitle_file.as_posix())

        # Output the filtered video AND the original audio
        run_output: Tuple[bytes, bytes] = (
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
            .run(quiet=True, overwrite_output=True)
        )
        _, err = run_output
        if err:
            logger.debug(err.decode())
        return output_file
    except ffmpeg.Error as e:
        logger.debug(f"Error hardcoding subtitles: {e.stderr.decode()}")
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
        with atomic_path(output_file) as temporary:
            run_output: Tuple[bytes, bytes] = (
                ffmpeg
                .output(
                    image_stream,
                    audio_stream,
                    str(temporary),
                    vcodec='libx264',
                    acodec='copy',
                    shortest=None,
                    preset='ultrafast',
                    tune='stillimage'
                )
                .run(quiet=True, overwrite_output=True)
            )
        _, err = run_output
        if err:
            logger.debug(err.decode())
        return output_file
    except ffmpeg.Error as e:
        logger.debug(f"Error creating video: {e.stderr.decode()}")
        raise