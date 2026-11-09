from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum, auto
import hashlib
import os
from pathlib import Path
import time
from urllib import parse


class FileKind(Enum):
    VIDEO = auto()
    AUDIO = auto()
    TRANSCRIPT = auto()
    SUBTITLE = auto()
    TEXT = auto()
    IMAGE = auto()
    UNKNOWN = auto()


_EXTENSIONS: dict[FileKind, frozenset[str]] = {
    FileKind.VIDEO: frozenset({".mp4", ".mkv", ".webm", ".ts", ".flv", ".mov", ".avi", ".wmv",
                               ".m4v", ".mpg", ".mpeg", ".m2ts", ".mts", ".3gp", ".ogv"}),
    FileKind.AUDIO: frozenset({".oga", ".mp3", ".aac", ".m4a", ".wav", ".flac", ".opus", ".ogg", ".ac3", ".eac3",
                               ".mka", ".wma", ".aiff", ".aif", ".weba"}),
    FileKind.TRANSCRIPT: frozenset({".json"}),
    FileKind.SUBTITLE: frozenset({".srt", ".ass", ".ssa", ".vtt", ".sbv"}),
    FileKind.TEXT: frozenset({".txt"}),
    FileKind.IMAGE: frozenset({".jpg", ".bmp", ".png", ".jpeg", ".webp"}),
}

_KIND_BY_EXTENSION = {extension: kind for kind, extensions in _EXTENSIONS.items() for extension in extensions}


def kind_of(path: Path) -> FileKind:
    """Classifies a file by its extension."""
    return _KIND_BY_EXTENSION.get(path.suffix.lower(), FileKind.UNKNOWN)


def supported_inputs() -> str:
    """Lists the extensions baberu accepts as input, for error messages."""
    kinds = (FileKind.VIDEO, FileKind.AUDIO, FileKind.TRANSCRIPT, FileKind.SUBTITLE)
    return " ".join(sorted(extension for kind in kinds for extension in _EXTENSIONS[kind]))


def is_url(text: str) -> bool:
    """Checks whether the text is a URL rather than a file path."""
    try:
        parsed = parse.urlparse(text)
    except ValueError:
        return False
    return bool(parsed.scheme and parsed.netloc)


@contextmanager
def atomic_path(target: Path) -> Iterator[Path]:
    """Gives a temporary path that becomes `target` only if the block succeeds.

    An interrupted write therefore can't leave a half-written file that a
    later run would trust. The temporary name keeps the extension because
    ffmpeg and pysubs2 choose their format from it.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.tmp-{os.getpid()}{target.suffix}")
    try:
        yield temporary
        _replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _replace(source: Path, target: Path) -> None:
    # Sync clients and antivirus can hold a new file open for a moment on Windows.
    for delay in (0.1, 0.3, 1.0):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            time.sleep(delay)
    os.replace(source, target)


def atomic_write_text(target: Path, text: str) -> None:
    with atomic_path(target) as temporary:
        temporary.write_text(text, encoding="utf-8")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_project_dir(marker: str = "pyproject.toml") -> Path:
    """Find the project root by looking for a marker file."""
    current_path = Path(__file__).parent
    for parent in [current_path] + list(current_path.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root dir with marker '{marker}'.")
