"""Cleans up what the translation model sends back. Everything here is a pure function."""
from pysubs2 import SSAEvent


def parse_response(translated_text: str) -> list[str]:
    """Splits a model response into cleaned subtitle lines."""
    new_lines = translated_text.strip().split('\n')
    if _is_numbered(new_lines):
        new_lines = _remove_numbering(new_lines)
    return _clean_ellipses(new_lines)


def force_line_count(translated_lines: list[str], 
                    target_lines: list[SSAEvent]) -> list[str]:
    """Forces the translated line count to match the target count as a last resort."""
    if len(translated_lines) > len(target_lines):
        # Combine excess lines
        translated_lines = translated_lines[:len(target_lines)-1] + \
            ["\\N".join(translated_lines[len(target_lines)-1:])]
    elif len(translated_lines) < len(target_lines):
        # Add placeholder lines
        translated_lines = translated_lines + ["[Translation missing]" for _ in range(len(target_lines) - len(translated_lines))]
    return translated_lines


def _is_numbered(lines: list[str]) -> bool:
    """Checks if a list of strings is formatted as a numbered list."""
    # Need at least 2 lines to confirm a pattern
    if len(lines) < 2:
        return False
        
    numbered_count = 0
    for i, line in enumerate(lines[:2]):
        if '.' in line:
            prefix, _, rest = line.partition('.')
            if prefix.strip().isdigit() and rest.startswith(' '):
                if int(prefix.strip()) == i+1:
                    numbered_count += 1
                    
    # Only consider it numbered if both first two lines have numbers
    return numbered_count == 2


def _remove_numbering(lines: list[str]) -> list[str]:
    """Removes leading 'N. ' numbering from a list of strings."""
    stripped_lines: list[str] = []
    for line in lines:
        if '.' in line:
            prefix, _, rest = line.partition('.')
            if prefix.strip().isdigit() and rest.startswith(' '):
                line = rest.strip()
        stripped_lines.append(line)
    return stripped_lines


def _clean_ellipses(lines: list[str]) -> list[str]:
    """Removes redundant ellipses at the start/end of adjacent lines."""
    modified_lines = list(lines)

    for i in range(len(modified_lines) - 1):
        current_line = modified_lines[i]
        next_line = modified_lines[i+1]
        if current_line.endswith("...") and next_line.startswith("..."):
            modified_lines[i] = current_line[:-3]
            modified_lines[i+1] = next_line[3:]
        if modified_lines[i].startswith("..."):
            modified_lines[i] = modified_lines[i][3:]
        if next_line.startswith("..."):
            modified_lines[i+1] = next_line[3:]

    return modified_lines
