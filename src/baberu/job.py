"""Where everything for one piece of media lives.

A job is a work folder beside the media. Structure comes from folders and
fixed names, and facts live in manifest.json, so nothing is ever worked out
by reading a file's name back.

    show.baberu/
      manifest.json
      audio.ogg  transcript.json  raw.ass  2pass.ass  fixed.ass  padded.ass
      segments/<start>-<end>.ass
      en/  context.txt  partial.txt  translated.ass  padded.ass
      lines-5-10/   (the same structure, for a --lines run)
"""
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
import shutil
from typing import Any

from baberu.errors import BaberuError
from baberu.subtitling.types import LineRange
from baberu.tools.file_utils import FileKind, atomic_write_text, kind_of

logger = logging.getLogger(__name__)

JOB_SUFFIX = ".baberu"
MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1


class Stage(Enum):
    TRANSCRIPT = "transcript.json"
    RAW = "raw.ass"
    TWOPASS = "2pass.ass"
    FIXED = "fixed.ass"
    CONTEXT = "context.txt"
    TRANSLATED = "translated.ass"
    PARTIAL = "partial.txt"
    PADDED = "padded.ass"


_ONE_PER_JOB = {Stage.TRANSCRIPT, Stage.RAW}
_ONE_PER_LANGUAGE = {Stage.CONTEXT}
_PER_RANGE_AND_LANGUAGE = {Stage.TRANSLATED, Stage.PARTIAL}


def enclosing_job(path: Path) -> Path | None:
    """Finds the work folder a file is inside, if any."""
    for parent in path.absolute().parents:
        if parent.name.endswith(JOB_SUFFIX):
            return parent
    return None


def _job_of_published_result(path: Path, folders: list[Path]) -> tuple[str, Path] | None:
    """Finds the job that recorded this file as one of its results."""
    candidate = path.stem
    while "." in candidate:
        candidate = candidate.rsplit(".", 1)[0]
        for folder in folders:
            manifest = folder / f"{candidate}{JOB_SUFFIX}" / MANIFEST_NAME
            if manifest.is_file() and path.name in _read_manifest(manifest).get("published", {}):
                return candidate, manifest.parent
    return None


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BaberuError(f"Can't read {path}: {error}. Delete the file to start this job's records again.") from error


@dataclass
class Job:
    name: str
    folder: Path
    result_dir: Path
    lang_from: str
    lang_to: str
    line_range: LineRange | None = None
    translating: bool = False
    # Files a dry run has counted as adopted, without having copied them.
    adopted: set[Path] = field(default_factory=set)
    manifest: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def locate(cls,
               input_file: Path,
               output_dir: Path,
               lang_from: str,
               lang_to: str,
               line_range: LineRange | None = None,
               translating: bool = False,
               legacy_name: str | None = None) -> "Job":
        """Works out which job a file belongs to."""
        name = input_file.stem
        folder = output_dir / f"{name}{JOB_SUFFIX}"

        enclosing = enclosing_job(input_file)
        if enclosing:
            name, folder = enclosing.name.removesuffix(JOB_SUFFIX), enclosing
        elif kind_of(input_file) in (FileKind.SUBTITLE, FileKind.TRANSCRIPT):
            published = _job_of_published_result(input_file, list(dict.fromkeys([input_file.parent, output_dir])))
            if published:
                name, folder = published
            elif legacy_name:
                name, folder = legacy_name, output_dir / f"{legacy_name}{JOB_SUFFIX}"

        job = cls(name, folder, output_dir, lang_from, lang_to, line_range, translating)
        job.manifest = _read_manifest(job.manifest_path) if job.manifest_path.is_file() else {}
        return job

    # --- Paths: the only place baberu decides where a file goes ---

    @property
    def manifest_path(self) -> Path:
        return self.folder / MANIFEST_NAME

    def path(self, stage: Stage, whole_file: bool = False) -> Path:
        """The file for a stage. `whole_file` ignores any --lines range."""
        base = self.folder
        if self.line_range and not whole_file:
            base = base / f"lines-{self.line_range.start + 1}-{self.line_range.end + 1}"

        if stage in _ONE_PER_JOB:
            return self.folder / stage.value
        if stage in _ONE_PER_LANGUAGE:
            return self.folder / self.lang_to / stage.value
        if stage in _PER_RANGE_AND_LANGUAGE or (stage is Stage.PADDED and self.translating):
            return base / self.lang_to / stage.value
        return base / stage.value

    def audio_path(self, extension: str) -> Path:
        return self.folder / f"audio.{extension}"

    def find_audio(self) -> Path | None:
        if self.folder.is_dir():
            for entry in self.folder.iterdir():
                if entry.stem == "audio" and kind_of(entry) is FileKind.AUDIO:
                    return entry
        return None

    def segment(self, start_ms: int, end_ms: int) -> Path:
        return self.folder / "segments" / f"{start_ms}-{end_ms}.ass"

    def result_path(self, input_file: Path, last_stage: str) -> Path:
        """The name of the file handed back to the user."""
        range_tag = f".{self.line_range.label}" if self.line_range else ""

        if self.translating:
            name = f"{self.name}.{self.lang_to}{range_tag}.ass"
        elif kind_of(input_file) is FileKind.SUBTITLE:
            base = self.name if enclosing_job(input_file) else input_file.stem
            name = f"{base}.{last_stage}{range_tag}.ass"
        else:
            name = f"{self.name}.{self.lang_from}{range_tag}.ass"

        path = self.result_dir / name
        if path.absolute() == input_file.absolute():
            path = self.result_dir / f"{input_file.stem}.{last_stage}{range_tag}.ass"
        return path

    def earlier_result(self) -> Path | None:
        """A subtitle result published by a previous run."""
        for language in (self.lang_to, self.lang_from):
            path = self.result_dir / f"{self.name}.{language}.ass"
            if path.exists() and path.name in self.manifest.get("published", {}):
                return path
        return None

    # --- Existence and clearing ---

    def has(self, path: Path) -> bool:
        return path.exists() or path in self.adopted

    def clear(self, *paths: Path) -> None:
        for path in paths:
            path.unlink(missing_ok=True)
            self.manifest.get("made_from", {}).pop(self._key(path), None)

    def clear_segments(self) -> None:
        shutil.rmtree(self.folder / "segments", ignore_errors=True)

    # --- Manifest ---

    def _key(self, path: Path) -> str:
        return path.relative_to(self.folder).as_posix()

    def save(self) -> None:
        self.manifest["version"] = MANIFEST_VERSION
        atomic_write_text(self.manifest_path, json.dumps(self.manifest, ensure_ascii=False, indent=2))

    def check_source(self, source_name: str) -> None:
        """Stops two different media files from sharing one work folder."""
        recorded = self.manifest.get("source")
        if recorded and recorded != source_name:
            raise BaberuError(
                f"{self.folder.name} holds the work for {recorded}, not {source_name}. "
                "Use -d to work somewhere else, or rename one of the files.")

    def set_source(self, source_name: str) -> None:
        self.manifest["source"] = source_name

    def origin_of(self, path: Path) -> dict[str, str] | None:
        """What a paid stage's file was made from, if that was recorded."""
        return self.manifest.get("made_from", {}).get(self._key(path))

    def record_origin(self, path: Path, source: str, model: str | None = None) -> None:
        self.manifest.setdefault("made_from", {})[self._key(path)] = {"source": source, "model": model}

    def published_hash(self, name: str) -> str | None:
        return self.manifest.get("published", {}).get(name)

    def record_published(self, name: str, content_hash: str) -> None:
        self.manifest.setdefault("published", {})[name] = content_hash
