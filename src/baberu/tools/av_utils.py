from typing import Any
from pathlib import Path
import logging

import yt_dlp
import ffmpeg

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
    print(f"Downloading from {url}...")
    
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
        'format': 'bestaudio*+worstvideo/best',
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
    
    print(f"Video downloaded successfully to {video_file}")
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
    print(f"Extracting audio from {video_file}...")

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
        print(f"Audio extracted successfully to {output_file}")
        return output_file
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {str(e)}")
        raise

def get_audio_codec(media_file: Path) -> str | None:
    """Probes a media file to determine its audio codec.

    Args:
        media_file: Path to the input media file.

    Returns:
        The codec name as a string, or None if no audio stream is found.
    """
    probe_result = ffmpeg.probe(str(media_file))
    audio_stream = next((stream for stream in probe_result.get('streams', [])
                         if stream.get('codec_type') == 'audio'), None)

    if not audio_stream:
        return None
    
    codec_name = audio_stream.get('codec_name')
    return codec_name

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
        (
            ffmpeg
            .input(str(audio_file), ss=start_time_sec, t=duration_sec)
            .output(str(output_path), acodec='libopus')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        start_time_fmt = f"{int(start_time_sec/60)}:{start_time_sec%60:.3f}"
        print(f"Extracted audio segment from {start_time_fmt} for {duration_sec:.2f}s")
        return output_path
    except ffmpeg.Error as e:
        print(f"Error extracting audio segment: {e}")
        raise

def hardcode_subtitles(video_file: Path,
                       subtitle_file: Path,
                       output_file: Path | None = None) -> Path:
    """Burns subtitles into a video, re-encoding to H.264 video and AAC audio.

    Args:
        video_file: Path to the input video file.
        subtitle_file: Path to the subtitle file (e.g., .srt, .ass).
        output_file: Optional path for the output video. If not provided,
            a '_subbed' suffix is added to the original video name.

    Returns:
        The path to the video with hardcoded subtitles.
    """
    print(f"Hardcoding subtitles from {subtitle_file} into {video_file}...")

    # If no output file is specified, create one with '_subbed' suffix
    if output_file is None:
        output_file = video_file.with_stem(f"{video_file.stem}_subbed")

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
        print(f"Subtitles hardcoded successfully to {output_file}")
        return output_file
    except ffmpeg.Error as e:
        print(f"Error hardcoding subtitles: {str(e)}")
        raise

def audio_to_video(image_file: Path,
                   audio_file: Path,
                   output_file: Path | None = None) -> Path:
    """Creates a video from a static image and an audio track.

    Args:
        image_file: Path to the input static image file.
        audio_file: Path to the input audio file.
        output_file: Optional path for the output video file.

    Returns:
        The path to the created video file.
    """
    print(f"Creating video '{output_file.name}' from image '{image_file.name}' and audio '{audio_file.name}'...")

    # If no output file is specified, create one with '_subbed' suffix
    if output_file is None:
        output_file = audio_file.with_name(f"{audio_file.stem}_template.mp4")

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
        print(f"Video created successfully at {output_file}")
        return output_file
    except ffmpeg.Error as e:
        print(f"Error creating video: {e.stderr}")
        raise