from pathlib import Path
from urllib import parse
import logging

logger = logging.getLogger(__name__)

class formats:
    _video_suffixes = [".mp4", ".mkv", ".webm", ".ts", ".flv", ".mov", ".avi", ".wmv"]
    _audio_suffixes = [".oga", ".mp3", ".aac", ".m4a", ".wav", ".flac", ".opus", ".ogg", ".ac3", ".eac3"]
    _json_suffixes = [".json"]
    _subtitle_suffixes = [".srt", ".ass", ".ssa", ".vtt", ".sbv"]
    _text_suffixes = [".txt"]
    _image_suffixes = [".jpg", ".bmp", ".png", ".jpeg", ".webp"]

    def is_url(input_str: str) -> bool:
        """Checks if the input string is a URL.
        
        Args:
            input_str: The input string to check.
        
        Returns:
            True if input is a URL, False otherwise.
        """
        parsed = parse.urlparse(input_str)
        return all([parsed.scheme, parsed.netloc])
    
    def is_video(file: Path | None) -> bool:
        """Checks if the file is a video.
        
        Args:
            file: The file path to check.
        
        Returns:
            True if the file is a video, False otherwise.
        """
        return file and file.suffix in formats._video_suffixes
    
    def is_audio(file: Path | None) -> bool:
        """Checks if the file is an audio file.
        
        Args:
            file: The file path to check.
        
        Returns:
            True if the file is an audio file, False otherwise.
        """
        return file and file.suffix in formats._audio_suffixes
    
    def is_json(file: Path | None) -> bool:
        """Checks if the file is a JSON file.
        
        Args:
            file: The file path to check.
        
        Returns:
            True if the file is a JSON file, False otherwise.
        """
        return file and file.suffix in formats._json_suffixes
    
    def is_sub(file: Path | None) -> bool:
        """Checks if the file is a subtitle file.
        
        Args:
            file: The file path to check.
        
        Returns:
            True if the file is a subtitle file, False otherwise.
        """
        return file and file.suffix in formats._subtitle_suffixes
    
    def is_text(file: Path | None) -> bool:
        """Checks if the file is a text file.
        
        Args:
            file: The file path to check.
        
        Returns:
            True if the file is a text file, False otherwise.
        """
        return file and file.suffix in formats._text_suffixes
    
    def is_image(file: Path | None) -> bool:
        """Checks if the file is an image file.
        
        Args:
            file: The file path to check.
        
        Returns:
            True if the file is an image file, False otherwise.
        """
        return file and file.suffix in formats._image_suffixes

def get_file_root(file_path: Path | None = None) -> str:
    """Gets the file's root name by stripping known suffixes."""
    if not file_path:
        return None
    suffixes_to_remove: list[str] = ['.padded', '.en', '.padded_custom', '.tr_custom', '.fixed_custom', '.partial', '.2pass', '.2pass_custom', '.context', '.fixed', '.raw'] 

    path_without_suffix = file_path.with_suffix('')

    current_name = path_without_suffix.name
    final_path = path_without_suffix

    for code in suffixes_to_remove:
        if current_name.endswith(code):
            new_name = current_name[:-len(code)]
            final_path = path_without_suffix.with_name(new_name)
    
    return str(final_path)


def get_project_dir(marker: str = "pyproject.toml") -> Path:
    """Find the project root by looking for a marker file."""
    current_path = Path(__file__).parent
    for parent in [current_path] + list(current_path.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root dir with marker '{marker}'.")
