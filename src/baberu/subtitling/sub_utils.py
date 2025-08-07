import re
from pathlib import Path
import logging

from pysubs2 import SSAFile

logger = logging.getLogger(__name__)

def load(file: Path) -> SSAFile:
    """Loads a subtitle file into an SSAFile object.

    Args:
        file: The path to the subtitle file.
    
    Returns:
        The loaded subtitle object.
    """
    return SSAFile.load(str(file), encoding="utf-8")

def write(sub_file: SSAFile, 
          output_file: Path) -> Path:
    """Saves an SSAFile object to a specified file path.

    Handles both standard subtitle formats and plain text (.txt) output.
    
    Args:
        sub_file: The SSAFile object to save.
        output_file: Path where the subtitle file will be saved.

    Returns:
        The path to the saved file.
    """
    if output_file.suffix == ".txt":
        with open(output_file, 'w', encoding='utf-8') as f:
            for event in sub_file:
                f.write(event.text + '\n')
    else:
        sub_file.save(str(output_file), encoding='utf-8')
    logger.debug(f"Saved file to {output_file}")
    return output_file

def splice(subtitles: SSAFile, 
           segment: list[int], 
           new_subtitles: SSAFile) -> SSAFile:
    """Replaces a segment of subtitle events with new events.
    
    Args:
        subtitles: The original SSAFile object to modify.
        segment: A list with the start and end indices of the slice to replace.
        new_subtitles: An SSAFile object containing the new events to insert.
        
    Returns:
        The modified SSAFile object.
    """
    if not segment:
        logger.error(f"Segment must be a list of two indices [start, end]. Received: {segment}")
        raise ValueError
    
    start_idx, end_idx = min(segment), max(segment)
    num_events = len(subtitles.events)

    if not (0 <= start_idx < num_events and 0 <= end_idx < num_events and start_idx <= end_idx):
        logger.warning(f"Warning: Segment [{start_idx}, {end_idx}] is invalid or out of bounds for {num_events} events.")
        return subtitles
    
    num_to_remove = end_idx - start_idx + 1

    subtitles.events[start_idx : end_idx + 1] = new_subtitles.events
    
    logger.info(f"Removed {num_to_remove} original subtitles and inserted {len(new_subtitles.events)} new lines.")
    return subtitles

def md_to_ass(subtitles: SSAFile) -> SSAFile:
    """Converts markdown-style formatting in subtitles to ASS override tags.
    
    Conversion rules:
    - ***text*** becomes {\\b1}{\\i1}text{\\i0}{\\b0} (bold italic)
    - **text** becomes {\\b1}text{\\b0} (bold)
    - *text* becomes {\\i1}text{\\i0} (italic)
    
    Args:
        subtitles: The SSAFile object to process.
        
    Returns:
        The SSAFile object with ASS formatting applied.
    """
    
    # Counters for tracking replacements
    bold_count = 0
    italic_count = 0
    bold_italic_count= 0

    bold_italic_pattern = re.compile(r'\*\*\*(.+?)\*\*\*')
    bold_pattern = re.compile(r'\*\*(.+?)\*\*')
    italic_pattern = re.compile(r'(?<!\*)\*([^*]+?)\*(?!\*)')
    
    # Process each subtitle event
    for event in subtitles.events:
        if r"{\i1}" in event.text or r"{\b1}" in event.text:
            continue  # Skip processing this event

        # Replace bold+italic formatting (triple asterisks) - must process first
        bold_italic_result = bold_italic_pattern.subn(r'{\\b1}{\\i1}\1{\\i0}{\\b0}', event.text)
        event.text = bold_italic_result[0]
        bold_italic_count += bold_italic_result[1]
        
        # Replace bold formatting (double asterisks)
        bold_result = bold_pattern.subn(r'{\\b1}\1{\\b0}', event.text)
        event.text = bold_result[0]
        bold_count += bold_result[1]
        
        # Replace italic formatting (single asterisks)
        italic_result = italic_pattern.subn(r'{\\i1}\1{\\i0}', event.text)
        event.text = italic_result[0]
        italic_count += italic_result[1]
        
    logger.info(f"Applied formatting: {bold_count} bold, {italic_count} italic, {bold_italic_count} bold-italic elements")
    return subtitles

def replace_lines(text_lines: list[str], source_subs: SSAFile, idx: int = 0) -> SSAFile:
    """Replaces text in existing subtitle events, preserving original timing.
    
    Args:
        text_lines: A list of new text strings for the subtitle lines.
        source_subs: The source SSAFile object to be modified.
        idx: The starting event index in `source_subs` for text replacement.
        
    Returns:
        The modified SSAFile object.
    """
    # Match number of lines to use
    available_lines = len(source_subs.events) - idx
    if len(text_lines) > available_lines:
        raise ValueError(f"Line mismatch: {len(text_lines)} new lines provided but {available_lines} lines available in source subtitles")
    else:
        line_count = min(len(text_lines), available_lines)
    
    # Update text in source subtitle events
    for i in range(line_count):
        source_subs.events[idx + i].text = text_lines[i]
    
    return source_subs

