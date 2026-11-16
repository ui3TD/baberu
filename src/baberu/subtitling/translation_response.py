"""Cleans up and checks what the translation model sends back. Everything here is a pure function."""
import re

from pysubs2 import SSAEvent

from baberu import constants

MISSING = "missing"
EMPTY = "empty"
UNTRANSLATED = "untranslated"

_THINKING = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
# "12. text", "12) text" or "12: text". The separator must be followed by a space
# or the end of the line; a line starting "1.5 million" is not entry 1.
_ENTRY = re.compile(r"^\s*(\d+)\s*[.):](?:\s+|$)(.*)$")


def parse_response(text: str, expected: int) -> list[str | None]:
    """Places each numbered entry of a response at its number.

    Code fences, thinking blocks and blank lines are dropped. A number that is
    missing is None, and a repeated number keeps its first entry. Unnumbered
    text before the first entry is a preamble and is dropped; unnumbered text
    after an entry is a second line of it. A response with no numbers at all is
    read by position, but only if it has exactly `expected` lines.
    """
    lines = [line for line in _THINKING.sub("", text).splitlines()
             if line.strip() and not line.strip().startswith("```")]

    entries: list[str | None] = [None] * expected
    current: int | None = None
    numbered = False
    for line in lines:
        match = _ENTRY.match(line)
        if match:
            numbered = True
            number = int(match.group(1))
            if 1 <= number <= expected and entries[number - 1] is None:
                current = number - 1
                entries[current] = match.group(2)
            else:
                current = None  # Out of range or repeated: ignore it and anything that follows it.
        elif current is not None:
            entries[current] = f"{entries[current]}\\N{line.strip()}"

    if not numbered:
        entries = list(lines) if len(lines) == expected else [None] * expected

    return _clean_ellipses([_clean(entry) for entry in entries])


def _clean(entry: str | None) -> str | None:
    if entry is None:
        return None
    return entry.replace(constants.CONTINUE_FLAG, "").strip()


def _clean_ellipses(lines: list[str | None]) -> list[str | None]:
    """Removes the ellipses that mark one sentence running across two lines.

    Removes a trailing '...' and the leading '...' of the next line only as a
    pair. Any other ellipsis stays.
    """
    cleaned = list(lines)
    for i in range(1, len(cleaned)):
        before, current = cleaned[i - 1], cleaned[i]
        if before is None or current is None:
            continue
        if before.endswith("...") and current.startswith("...") and before != "..." and current != "...":
            cleaned[i - 1] = before[:-3].rstrip()
            cleaned[i] = current[3:].lstrip()
    return cleaned


def problems(lines: list[str | None],
             batch: list[SSAEvent],
             kept: int,
             same_language: bool = False) -> dict[int, str]:
    """Finds the kept lines that need another try, by their index in the batch.

    Only the first `kept` lines count; the lookahead after them is context.
    """
    found: dict[int, str] = {}
    for i in range(min(kept, len(batch))):
        line = lines[i] if i < len(lines) else None
        source = _clean(batch[i].text) or ""
        if line is None:
            found[i] = MISSING
        elif not line and source:
            found[i] = EMPTY
        elif not same_language and line == source and any(c.isalpha() for c in source):
            found[i] = UNTRANSLATED
    return found
