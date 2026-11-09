"""Every file name baberu reads or writes for a job is decided here."""
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Artifacts:
    """Names the files produced for one input.

    Attributes:
        root: Output path without any extension, e.g. "out/video".
        lang_to: Target language, which is part of translated file names.
        lines_label: "L5-10" when working on part of a file, otherwise "".
            A partial result is only valid for the range it was made from.
        previous_root: Where files were written before `--directory` was
            honoured. Finished work found there is reused rather than redone.
    """

    root: str
    lang_to: str = "en"
    lines_label: str = ""
    previous_root: str | None = None

    def _resolve(self, suffix: str) -> Path:
        path = Path(self.root + suffix)
        if path.exists() or not self.previous_root:
            return path
        previous = Path(self.previous_root + suffix)
        if previous.exists():
            logger.info(f"Reusing existing file: {previous}")
            return previous
        return path

    def _stage(self, full_name: str, custom_name: str) -> Path:
        if self.lines_label:
            return self._resolve(f".{custom_name}.{self.lines_label}.ass")
        return self._resolve(f".{full_name}.ass")

    @property
    def _translation_tag(self) -> str:
        return "tr_custom" if self.lang_to == "en" else f"tr_custom.{self.lang_to}"

    def audio(self, extension: str) -> Path:
        return self._resolve(f".{extension}")

    def transcript(self) -> Path:
        return self._resolve(".json")

    def raw(self) -> Path:
        return self._resolve(".raw.ass")

    def context(self) -> Path:
        return self._resolve(".context.txt")

    def twopass(self) -> Path:
        return self._stage("2pass", "2pass_custom")

    def fixed(self) -> Path:
        return self._stage("fixed", "fixed_custom")

    def translated(self) -> Path:
        return self._stage(self.lang_to, self._translation_tag)

    def padded(self) -> Path:
        return self._stage("padded", "padded_custom")

    def segment_transcript(self, audio_file: Path, start_ms: int, end_ms: int) -> Path:
        """Saved result of re-transcribing one stretch of audio, kept beside the audio."""
        return audio_file.with_suffix(f".{start_ms}-{end_ms}.ass")

    def partial_translation(self) -> Path:
        if self.lines_label:
            return self._resolve(f".partial.{self._translation_tag}.{self.lines_label}.txt")
        return self._resolve(f".partial.{self.lang_to}.txt")
