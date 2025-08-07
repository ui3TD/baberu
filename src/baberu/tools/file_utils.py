from pathlib import Path
from urllib import parse
import logging

from baberu.constants import VIDEO_SUFFIXES, AUDIO_SUFFIXES, SUBTITLE_SUFFIXES, JSON_SUFFIXES, TEXT_SUFFIXES, IMAGE_SUFFIXES, BABERU_SUFFIXES

logger = logging.getLogger(__name__)

class formats:
    @staticmethod
    def is_url(input_str: str) -> bool:
        """Checks if the input string is a URL.
        
        Args:
            input_str: The input string to check.
        
        Returns:
            True if input is a URL, False otherwise.
        """
        try:
            parsed = parse.urlparse(input_str)
            return all([parsed.scheme, parsed.netloc])
        except (ValueError, AttributeError):
            return False

    @staticmethod
    def _is_of_type(file: Path | None, suffixes: frozenset[str]) -> bool:
        """
        Check if a file's suffix is in a given set.
        """
        return file is not None and file.suffix.lower() in suffixes

    @staticmethod
    def is_video(file: Path | None) -> bool:
        """Checks if the file is a video."""
        return formats._is_of_type(file, VIDEO_SUFFIXES)

    @staticmethod
    def is_audio(file: Path | None) -> bool:
        """Checks if the file is an audio file."""
        return formats._is_of_type(file, AUDIO_SUFFIXES)

    @staticmethod
    def is_json(file: Path | None) -> bool:
        """Checks if the file is a JSON file."""
        return formats._is_of_type(file, JSON_SUFFIXES)

    @staticmethod
    def is_sub(file: Path | None) -> bool:
        """Checks if the file is a subtitle file."""
        return formats._is_of_type(file, SUBTITLE_SUFFIXES)

    @staticmethod
    def is_text(file: Path | None) -> bool:
        """Checks if the file is a text file."""
        return formats._is_of_type(file, TEXT_SUFFIXES)

    @staticmethod
    def is_image(file: Path | None) -> bool:
        """Checks if the file is an image file."""
        return formats._is_of_type(file, IMAGE_SUFFIXES)

def get_file_root(file_path: Path | None = None) -> str:
    """Gets the file's root name by stripping known suffixes."""
    if not file_path:
        return ""

    path_without_suffix = file_path.with_suffix('')

    current_name = path_without_suffix.name
    final_path = path_without_suffix

    for code in BABERU_SUFFIXES:
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
