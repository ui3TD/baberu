from dataclasses import dataclass

import pydantic


@dataclass(frozen=True)
class LineRange:
    """A run of subtitle lines. Both ends are zero-based and included."""

    start: int
    end: int

    @classmethod
    def parse(cls, text: str) -> "LineRange":
        """Reads the 1-based form used on the command line: "5-10" or "5"."""
        first, _, last = text.partition("-")
        start = int(first)
        end = int(last) if last else start
        return cls(start - 1, end - 1)

    @property
    def label(self) -> str:
        """The 1-based range as it appears in file names, e.g. "L5-10"."""
        return f"L{self.start + 1}-{self.end + 1}"

    @property
    def count(self) -> int:
        return self.end - self.start + 1

    def resized(self, delta: int) -> "LineRange":
        """The same start, after `delta` lines were added to or removed from the range."""
        return LineRange(self.start, max(self.start, self.end + delta))


class SubtitleLine(pydantic.BaseModel):
    text: str
    start: float
    end: float
    speaker: str = "Unknown"