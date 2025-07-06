import pysubs2
from pysubs2 import SSAFile

import logging

logger = logging.getLogger(__name__)

def find_mistimed_lines(subtitles: SSAFile, 
                        threshold: float,
                        grp_min_lines: int = 4,
                        grp_backtrace_limit: int = 20,
                        grp_foretrace_limit: int = 5,
                        grp_min_delay_sec: float = 10,
                        grp_max_gap: int = 4) -> set[int]:
    """Identifies subtitle lines that are likely mistimed.

    This function finds lines with a duration below a given threshold,
    then expands the selection to include surrounding lines and merge
    proximate groups to form coherent segments of mistimed content.

    Args:
        subtitles (SSAFile): The subtitle file to process.
        threshold (float): Duration in seconds below which a line is considered short.
        grp_min_lines (int): Minimum number of consecutive short lines to form a group.
        grp_backtrace_limit (int): Max number of lines to search backward from a group.
        grp_foretrace_limit (int): Max number of lines to search forward from a group.
        grp_min_delay_sec (float): Minimum duration for a line to be considered a valid boundary.
        grp_max_gap (int): Maximum number of lines between two groups to be merged.

    Returns:
        set[int]: A set of zero-based indices corresponding to mistimed subtitle lines.
    """
    # Initial identification of short subtitles
    mistimed_subtitles: set[int] = set()
    
    for i, subtitle in enumerate(subtitles.events, 0):
        duration_seconds = (subtitle.end - subtitle.start) / 1000.0
            
        if duration_seconds <= threshold:
            mistimed_subtitles.add(i)
    
    logger.info(f"Found {len(mistimed_subtitles)} subtitle(s) with duration less than {threshold} seconds")
    if not mistimed_subtitles:
        return mistimed_subtitles

    # Include gaps between short subtitles
    mistimed_subtitles = _fill_gaps(mistimed_subtitles)

    # Group consecutive short subtitles
    groups = find_mistimed_groups(mistimed_subtitles, grp_min_lines)

    logger.info(f"Found {len(groups)} large mistimed groups with duration less than {threshold} seconds")
    if not groups:
        return mistimed_subtitles
        
    # For large groups with consecutive indices, attempt expand
    for group in groups:
        # Expand backward
        indices_backward = _expand_mistimed_group(
            subtitles, group[0] - 1, -1, grp_backtrace_limit, grp_min_delay_sec
        )
        
        if indices_backward:
            mistimed_subtitles.update(indices_backward)
            group.extend(indices_backward)
            group.sort()
            logger.info(f"Extending group ({group[0] + 1}-{group[-1] + 1}) backward to line {min(indices_backward) + 1}")
    
        # Expand forward
        indices_forward = _expand_mistimed_group(
            subtitles, group[-1] + 1, 1, grp_foretrace_limit, grp_min_delay_sec
        )
        
        if indices_forward:
            mistimed_subtitles.update(indices_forward)
            group.extend(indices_forward)
            group.sort()
            logger.info(f"Extending group ({group[0] + 1}-{group[-1] + 1}) forward to line {max(indices_forward) + 1}")
        
        if not indices_backward and not indices_forward:
            logger.warning(f"Failed to extend group ({group[0] + 1}-{group[-1] + 1})")

    # Merge large groups that are close
    mistimed_subtitles = _merge_nearby_groups(mistimed_subtitles, groups, grp_max_gap)

    return mistimed_subtitles

def find_mistimed_groups(mistimed_lines: set[int], 
                         group_min_lines: int) -> list[list[int]]:
    """Identifies consecutive groups of mistimed subtitle indices.

    Args:
        mistimed_lines (set[int]): A set of indices for mistimed lines.
        group_min_lines (int): The minimum number of lines to constitute a group.

    Returns:
        list[list[int]]: A list of groups, where each group is a list of consecutive indices.
    """
    groups = []
    current_group = []
    sorted_indices = sorted(mistimed_lines)
    for idx in sorted_indices:
        if not current_group or idx == current_group[-1] + 1:
            current_group.append(idx)
        else:
            if len(current_group) >= group_min_lines:
                groups.append(current_group)
            current_group = [idx]
                
    if current_group and len(current_group) >= group_min_lines:
        groups.append(current_group)
    return groups

def fix_mistimed_lines(subtitles: SSAFile, 
                       threshold: float,
                        grp_min_lines: int = 4,
                        grp_backtrace_limit: int = 20,
                        grp_foretrace_limit: int = 5,
                        grp_min_delay_sec: float = 10,
                        grp_max_gap: int = 4,
                        segment: list[int] = []) -> SSAFile:
    """Finds and corrects mistimed subtitles by redistributing their timing.

    This function identifies groups of short-duration subtitles and merges them
    with an adjacent valid subtitle, adjusting timings proportionally.

    Args:
        subtitles (SSAFile): The subtitle file to correct.
        threshold (float): Duration in seconds below which a line is considered mistimed.
        grp_min_lines (int): Minimum number of consecutive short lines to form a group.
        grp_backtrace_limit (int): Max number of lines to search backward from a group.
        grp_foretrace_limit (int): Max number of lines to search forward from a group.
        grp_min_delay_sec (float): Minimum duration for a line to be considered a valid boundary.
        grp_max_gap (int): Maximum number of lines between two groups to be merged.
        segment (list[int]): Optional line range to restrict the operation.

    Returns:
        SSAFile: The subtitle object with corrected timings.
    """
    mistimed_indices: set[int] = find_mistimed_lines(subtitles, threshold, grp_min_lines, grp_backtrace_limit, grp_foretrace_limit, grp_min_delay_sec, grp_max_gap)
    
    if not mistimed_indices:
        return subtitles
    
    groups: list[list[int]] = find_mistimed_groups(mistimed_indices, 1)

    # Filter for segment
    if segment:
        segment_start, segment_end = min(segment), max(segment)
        groups = [
            group for group in groups 
            if min(group) <= segment_end and max(group) >= segment_start
        ]

    # Process each group of consecutive short subtitles
    for group in groups:            
        # Find adjacent non-short subtitles for regular groups
        prev_idx = group[0] - 1
        next_idx = group[-1] + 1
        
        # Handle group extension
        if prev_idx < 0:
            # Concatenate with next index
            subtitles = _merge_forward(subtitles, next_idx, group)
        else:
            # Redistribute time with previous index
            subtitles = _redistribute_backward(subtitles, prev_idx, group)

    logger.info(f"Extended subtitle durations using threshold of {threshold} s")
    return subtitles

def remove_empty(subtitles: SSAFile,
                 segment: list[int] = []) -> SSAFile:
    """Removes all subtitle events that contain no text.

    Args:
        subtitles (SSAFile): The subtitle object to process.
        segment (list[int]): Optional line range to restrict the operation.

    Returns:
        SSAFile: The subtitle object with empty lines removed.
    """
    original_count = len(subtitles.events)

    if segment:
        segment_start, segment_end = min(segment), max(segment)
        
        subtitles.events = [
            event for i, event in enumerate(subtitles.events)
            if not (segment_start <= i <= segment_end and not event.text.strip())
        ]
    else:
        subtitles.events = [event for event in subtitles.events if event.text.strip()]
    
    removed_count = original_count - len(subtitles.events)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} empty subtitle(s).")
    return subtitles

def apply_timing_standards(subtitles: SSAFile, 
               max_lead_out_sec: float = 1.0,
               max_lead_in_sec: float = 0.25,
               max_cps: float = 20.0,
               min_sec: float = 1.0,
               segment: list[int] = []) -> SSAFile:
    """Extends subtitle durations to meet minimum duration and CPS standards.

    This function adjusts the end time (and start time, if necessary) of each
    subtitle line to ensure it is displayed for a sufficient duration based on
    its text length, without overlapping adjacent lines of the same style.

    Args:
        subtitles (SSAFile): The subtitle object to process.
        max_lead_out_sec (float): Maximum seconds to extend a line's end time.
        max_lead_in_sec (float): Maximum seconds to extend a line's start time.
        max_cps (float): Maximum characters per second to enforce.
        min_sec (float): Minimum duration in seconds for any subtitle line.
        segment (list[int]): Optional line range to restrict the operation.

    Returns:
        SSAFile: The subtitle object with extended durations.
    """
    num_events = len(subtitles.events)
    if num_events == 0:
        return subtitles

    modified_count = 0

    # Determine the range of indices to process based on the segment
    process_range = range(num_events)
    if segment:
        start_idx = min(segment)
        end_idx = max(segment)
        process_range = range(start_idx, end_idx + 1)
    
    for i in process_range:
        current_subtitle = subtitles.events[i]
        text_length = len(current_subtitle.text)
        
        if text_length == 0:
            continue # Skip empty lines

        # Calculate the minimum duration required
        target_duration_ms = max((text_length / max_cps) * 1000.0 if max_cps > 0 else 0, min_sec * 1000)

        # Calculate the target end time
        target_end_time = current_subtitle.start + target_duration_ms
        max_target_end_time = current_subtitle.end + max_lead_out_sec * 1000
        target_end_time = min(target_end_time, max_target_end_time)
        
        # Only extend if the current duration is shorter than required
        if target_end_time <= current_subtitle.end:
            continue

        # Find the start time of the next subtitle with the same style
        limit_start_time = None
        for j in range(i + 1, num_events):
            next_subtitle = subtitles.events[j]
            if next_subtitle.style == current_subtitle.style:
                limit_start_time = next_subtitle.start
                break
        


        # Extend up to the target time or the start of the next same-style line, whichever is earlier
        if limit_start_time is not None:
            new_end_time = min(target_end_time, limit_start_time)
        else:
            new_end_time = target_end_time

        # Calculate the duration after forward extension
        new_duration = new_end_time - current_subtitle.start
        
        # If still too short, try backward extension
        if target_duration_ms > new_duration:    
            # Calculate how much more time we need
            remaining_ms_needed = target_duration_ms - new_duration
            target_backward_ms = min(remaining_ms_needed, max_lead_in_sec * 1000)
            
            # Find the end time of the previous subtitle with the same style
            prev_end_time = 0
            for j in range(i - 1, -1, -1):
                prev_subtitle = subtitles.events[j]
                if prev_subtitle.style == current_subtitle.style:
                    prev_end_time = prev_subtitle.end
                    break
            
            # Calculate how far back we can go
            new_start_time = max(
                current_subtitle.start - target_backward_ms,
                prev_end_time
            )
            
            # Apply backward extension if possible
            if new_start_time < current_subtitle.start:
                current_subtitle.start = new_start_time
                modified_count += 1

        # Avoid rounding errors
        final_new_end_time = max(current_subtitle.end, new_end_time)
        final_new_end_time = max(final_new_end_time, current_subtitle.start + 1)

        if final_new_end_time > current_subtitle.end:
                current_subtitle.end = pysubs2.time.times_to_ms(ms=final_new_end_time)
                modified_count += 1

    if modified_count > 0:
        logger.info(f"Extended {modified_count} subtitle(s) to meet max CPS of {max_cps}")
    else:
        logger.info(f"No subtitles required extension to meet max CPS of {max_cps}.")
        
    return subtitles

def _fill_gaps(mistimed_subtitles: set[int]) -> set[int]:
    """Fills single-line gaps between two mistimed subtitles."""
    sorted_indices: list[int] = sorted(mistimed_subtitles)
    for i in range(len(sorted_indices) - 1):
        current = sorted_indices[i]
        next_idx = sorted_indices[i + 1]
        if next_idx - current == 2:
            mistimed_subtitles.add(current + 1)
    return mistimed_subtitles

def _merge_nearby_groups(mistimed_subtitles: set[int], 
                           groups: list[list[int]], 
                           gap: int) -> set[int]:
    """Merges nearby groups of mistimed subtitles into a single continuous group."""
    i = 0
    while i < len(groups) - 1:
        groups[i].sort()
        groups[i+1].sort()

        current_group_start = groups[i][0]
        current_group_end = groups[i][-1]
        next_group_start = groups[i+1][0]
        next_group_end = groups[i+1][-1]
        
        if next_group_start - current_group_end <= gap + 1:
            # Determine the full range covered by the merged group
            merged_start = current_group_start # Starts with the first group
            merged_end = max(current_group_end, next_group_end) # Ends at the furthest point of either group

            logger.info(f"Merging groups  {current_group_start + 1}-{current_group_end + 1} and {next_group_start + 1}-{next_group_end + 1} into {merged_start + 1}-{merged_end + 1}")

            # Merge the groups
            merged_group = list(range(merged_start, merged_end + 1))
            groups[i] = merged_group
            groups.pop(i+1)
            mistimed_subtitles.update(merged_group)
        else:
            i += 1
    return mistimed_subtitles

def _expand_mistimed_group(subtitles: SSAFile, 
                           start_idx: int, 
                           direction: int, 
                           limit: int, 
                           threshold: float) -> set:
    """Expands a group of mistimed lines by searching for adjacent short-duration lines."""
    added_indices = set()
    count = 0
    idx = start_idx
    
    end_condition = idx >= 0 if direction < 0 else idx < len(subtitles.events)
    
    while end_condition and count < limit:
        subtitle = subtitles.events[idx]
        duration_seconds = (subtitle.end - subtitle.start) / 1000.0
        
        added_indices.add(idx)
        
        if duration_seconds > threshold:
            break
            
        idx += direction
        count += 1
        end_condition = idx >= 0 if direction < 0 else idx < len(subtitles.events)
    
    hit_limit = count >= limit
    if hit_limit:
        added_indices = set()

    return added_indices

def _merge_forward(subtitles: SSAFile, 
                    next_idx: int, 
                    group: list[int]) -> SSAFile:
    """Merges a group of mistimed subtitles forward into the next valid subtitle."""
    if next_idx >= len(subtitles.events):
        return subtitles  # No next subtitle to extend to
        
    # Gather text from all short subtitles in the group
    combined_text = " ".join(subtitles.events[j].text for j in group)

    subtitles.events[next_idx].text = combined_text + " " + subtitles.events[next_idx].text
    
    # Update the start time
    subtitles.events[next_idx].start = subtitles.events[group[0]].start
    
    # Clear text from the short subtitles that were concatenated
    for j in group:
        subtitles.events[j].text = ""
        
    return subtitles

def _redistribute_backward(subtitles: SSAFile, 
                     prev_idx: int, 
                     group: list[int]) -> SSAFile:
    """Redistributes timing for a group of mistimed subtitles and the preceding valid subtitle."""

    # Calculate total duration and character count
    total_start_time = subtitles.events[prev_idx].start
    total_end_time = subtitles.events[group[-1]].end
    total_duration = total_end_time - total_start_time
    
    # Get all subtitle indices including the previous one
    all_indices = [prev_idx] + group

    # Calculate total character length
    text_lengths = [len(subtitles.events[prev_idx].text)] + [len(subtitles.events[j].text) for j in group]
    total_chars = sum(text_lengths)
    
    if total_chars == 0:
        return subtitles
    
    # Redistribute duration based on character length
    current_time = total_start_time
    
    # Apply time redistribution in a single loop
    current_time = total_start_time
    for i, idx in enumerate(all_indices):
        subtitles.events[idx].start = current_time
        new_duration = int(total_duration * (text_lengths[i] / total_chars))
        current_time += new_duration
        subtitles.events[idx].end = current_time
    
    return subtitles
