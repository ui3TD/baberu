import time
from pathlib import Path
import logging

import langcodes
from pysubs2 import SSAFile, SSAEvent

from baberu.LLMFactory.factory import AIToolFactory
from baberu.LLMFactory.llm.base import LLMProvider
from baberu.subtitling import elevenlabs_utils


logger = logging.getLogger(__name__)

def translate(sub_file: SSAFile,
              output_file: Path, 
              context_prompt: str, 
              model: str,
              lang_from: str,
              lang_to: str,
              context_lines: int = 100,
              batch_lines: int = 50,
              discard_lines: int = 10,
              translate_retries: int = 3,
              server_retries: int = 5,
              max_cont_lines: int = 5,
              segment: list[int] = []) -> list[str]:
    """Translates a subtitle file from a source to a target language using an LLM.

    Handles batching, context management, retries, and rate limiting.

    Args:
        sub_file: SSAFile object containing the source subtitles.
        output_file: Path to save the translated subtitle text.
        context_prompt: General context to guide the translation.
        model: The LLM model to use for translation.
        lang_from: BCP 47 code for the source language.
        lang_to: BCP 47 code for the target language.
        context_lines: Number of previous translations to use as context.
        batch_lines: Number of lines to process in each API call.
        discard_lines: Number of lookahead lines to add for context, then discard.
        translate_retries: Number of retries for fixing line count mismatches.
        server_retries: Number of retries for handling server/connection errors.
        max_cont_lines: Maximum consecutive lines ending in a continuation flag to group into a batch.
        segment: A list of two integers specifying the start and end line numbers to translate.

    Returns:
        A list of the translated subtitle lines.
    """
    starting_index: int = 0
    last_api_call_time: float = 0
    is_openrouter_free: bool = model.startswith('google/gemini-2.5-pro-exp')
    def_time_per_bath: float = 220 if is_openrouter_free else 81

    if segment:
        starting_index = min(segment)
        translated_lines = [i.text for i in sub_file.events[0:starting_index]]
    else:
        translated_lines = []
        
    # Check for existing translations
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                translated_lines = [line.strip() for line in f.readlines()]
            starting_index = len(translated_lines)
        except Exception as e:
            logger.warning(f"Error loading existing translation file: {str(e)}. Starting from scratch.")
    
    # Get total items
    total_items = len(sub_file.events)

    # If we've already translated everything, we're done
    if starting_index >= total_items:
        logger.warning(f"All {starting_index} lines are already translated.")
        return translated_lines
    
    # Initialize LLM client
    system_prompt: str = _set_sys_prompt(lang_from, lang_to)
    llm_client: LLMProvider = AIToolFactory.get_llm_provider(model_name=model, system_prompt=system_prompt)

    start_time: float = time.perf_counter()

    ending_item = total_items
    if segment:
        ending_item = max(segment) + 1

    i: int = starting_index 
    processed_batches: int = 0

    try:
        while i < ending_item:
            initial_batch_end: int = min(i + batch_lines, ending_item)

            batch_end: int = initial_batch_end 
            continues_added: int = 0
            while batch_end < ending_item and \
                sub_file.events[batch_end - 1].text.endswith(elevenlabs_utils.CONTINUE_FLAG) and \
                    continues_added < max_cont_lines:
                # Expand batch end by one line
                batch_end += 1
                continues_added += 1

            batch_size: int = batch_end - i 

            # Prepare source text batch
            discard_lines: int = min(discard_lines, ending_item - batch_end)
            current_batch = sub_file.events[i:(batch_end + discard_lines)]
            if not current_batch:
                break

            prompt: str = _set_translate_prompt(translated_lines, context_prompt, current_batch, lang_from, lang_to, context_lines)

            # Progress Metrics
            processed_batches += 1
            remaining_items_estimate: int = ending_item - i 
            total_batches: int = (remaining_items_estimate + batch_lines - 1) // batch_lines + processed_batches -1
            
            elapsed_time: float = time.perf_counter() - start_time
            _print_progress(elapsed_time, processed_batches, total_batches, i, batch_end, ending_item, def_time_per_bath) 
            
            # Attempt translation with retries
            translated_text: str = ""
            for translate_attempt in range(translate_retries):

                if is_openrouter_free:
                    time_since_last_call = time.perf_counter() - last_api_call_time
                    if time_since_last_call < 61:
                        wait_time = 61 - time_since_last_call
                        time.sleep(wait_time)

                for server_attempt in range(server_retries):
                    try:                        
                        if is_openrouter_free:
                            last_api_call_time = time.perf_counter()
                            
                        translated_text = llm_client.prompt(prompt)
                            
                        _log_response(prompt, translated_text, processed_batches, total_batches, i, batch_end)
                        break
                    except KeyboardInterrupt:
                        logger.warning("\nTranslation aborted by user. Saving partial results...")
                        raise
                    except ConnectionError:
                        if server_attempt < (server_retries - 1):
                            logger.warning(f"Connection Error (attempt {server_attempt+1}/{server_retries}). Retrying...")
                            time.sleep(min(5 * server_attempt, 65))
                        else:
                            logger.error(f"Connection Error (attempt {server_attempt+1}/{server_retries})")
                            raise
                    except Exception as e:
                        if server_attempt < (server_retries - 1) and ("503 UNAVAILABLE" in str(e) or "overloaded" in str(e)):
                            logger.warning(f"Server overloaded (attempt {server_attempt+1}/{server_retries}). Retrying...")
                            time.sleep(2)
                        else:
                            raise
                
                # Process the translation
                new_lines = translated_text.strip().split('\n')
                lines_are_numbered = _is_numbered(new_lines)
                if lines_are_numbered:
                    new_lines: list[str] = _remove_numbering(new_lines)
                
                new_lines = _clean_ellipses(new_lines)
                
                # Check if line count matches
                if len(new_lines) == len(current_batch):
                    break
                
                # If line count doesn't match, retry with more explicit instructions
                logger.warning(f"Translation attempt {translate_attempt+1} failed: expected {len(current_batch)} lines, got {len(new_lines)}. Retrying...")

                prompt = _set_retry_prompt(new_lines, current_batch, lang_from, lang_to)

            # Handle line count mismatch after all retries
            if len(new_lines) != len(current_batch):
                logger.warning(f"Warning: Failed to get exact translation line count after {translate_retries} attempts. Proceeding with best effort.")
                new_lines = _force_line_count(new_lines, current_batch)

            # Exclude discarded lines
            new_lines = new_lines[:batch_size]
                        
            translated_lines.extend(new_lines)

            write_lines(translated_lines, output_file)

            i = batch_end 

    except Exception as e:
        logger.error(f"Unexpected error during translation: {str(e)}")
        raise
    finally:
        duration: float = time.perf_counter() - start_time
        duration_minutes, duration_seconds = divmod(int(duration), 60)
        logger.info(f"Translation ran for {duration_minutes} min {duration_seconds} s")

    return translated_lines

def load_context(file_path: Path) -> str:
    """Loads a context prompt from a text file.

    Args:
        file_path: The path to the context file.

    Returns:
        The content of the file as a string, or an empty string on error.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning(f"Warning: Context file '{file_path}' not found. Using empty context.")
        return ""
    except Exception as e:
        logger.warning(f"Error reading context file: {str(e)}. Using empty context.")
        return ""

def generate_context(subtitles: SSAFile, model: str, filename: str, lang_from: str, lang_to: str) -> str:
    """Generates a context summary for a subtitle file using an LLM.

    The summary includes content type, a brief synopsis, and a glossary.

    Args:
        subtitles: The SSAFile object containing source subtitles.
        model: The LLM model to use for context generation.
        filename: The original filename, used in the prompt.
        lang_from: BCP 47 code for the source language.
        lang_to: BCP 47 code for the target language.

    Returns:
        A string containing the generated context.
    """
    prompt: str = _set_context_prompt(subtitles, filename, lang_from, lang_to)
    
    #DEBUG
    logger.debug(f"Prompt: {prompt}")

    llm_client: LLMProvider = AIToolFactory.get_llm_provider(model_name=model, 
                                                             system_prompt="You must follow prompt instructions.")
    try:
        context: str = llm_client.prompt(prompt, grounding=True)
    except KeyboardInterrupt:
        logger.warning("\nTranslation aborted by user. Saving partial results...")
        raise
    except Exception as e:
        raise
    return context

def text_to_subs(text_lines: list[str], 
                    timed_events: list[SSAEvent], 
                    prepended_subs: SSAFile) -> SSAFile:
    """Converts a list of text strings into an SSAFile object.

    Combines translated text lines with the timing and style information from
    the original subtitle events.

    Args:
        text_lines: A list of translated subtitle text strings.
        timed_events: A list of original SSAEvent objects for timing and style.
        prepended_subs: An SSAFile to append the new events to.

    Returns:
        The SSAFile with the newly created and appended events.
    """
    for _, (item, text) in enumerate(zip(timed_events, text_lines)):
        event = SSAEvent(
            start=item.start,
            end=item.end,
            style=item.style,
            text=text
        )
        prepended_subs.events.append(event)
    return prepended_subs

def write_lines(lines: list[str], output_file: Path) -> None:
    """Writes a list of strings to a text file, with each string on a new line.

    Args:
        lines: The list of strings to write.
        output_file: The path to the output text file.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    except Exception as e:
        logger.error(f"Error writing lines to file: {str(e)}")
    return

def _set_sys_prompt(lang_from: str, lang_to: str) -> str:
    """Creates the system prompt for the LLM translator role."""
    prompt: str = f"You are a professional translator from {_get_lang_name(lang_from)} to natural colloquial {_get_lang_name(lang_to)}. Translate the provided subtitle entries liberally and concisely while preserving the meaning and nuance. Return ONLY the translated entries, maintaining EXACT entry count and order. Do not merge entries.\n\nSpecial instructions:\n1. '{elevenlabs_utils.CONTINUE_FLAG}' indicates that text was split mid-sentence to be continued on the next entries. You must omit '{elevenlabs_utils.CONTINUE_FLAG}' from the translated text.\n2. As per ASS syntax, use the special escape character '\\N' for line breaks within a subtitle entry.\n3. Use ASS syntax for styling if needed (e.g. {{\\i1}}italics{{\\i0}}). \n4. Do not split {_get_lang_name(lang_to)} words across subtitle entries."
    return prompt

def _set_translate_prompt(previous_translations: list[str], 
                          context_prompt: str, 
                          current_batch: list[SSAEvent], 
                          lang_from: str, 
                          lang_to: str,
                          context_lines: int) -> str:
    """Constructs the user prompt for translating a batch of subtitles."""
    # Prepare translation context if available
    context_section = ""
    if previous_translations:
        context_str = "\n".join(previous_translations[-context_lines:])
        context_section = f"For continuity, here are the last {min(context_lines, len(previous_translations))} translated entries:\n{context_str}\n\n\nContinue directly from the last entry. "

    batch_text: str = "\n".join([
        f"{idx+1}. {item.text.replace('\\n', r'\\N')}"
        for idx, item in enumerate(current_batch)
    ])
    
    prompt: str = f"Context:\n{context_prompt}\n\n{context_section}Translate the following {len(current_batch)} {_get_lang_name(lang_from)} subtitle entries to {_get_lang_name(lang_to)}. Maintain exact entry count and order:\n\n{batch_text}"
    return prompt

def _set_retry_prompt(translated_lines: list[str], 
                      orig_subevents: list[SSAEvent], 
                      lang_from: str, 
                      lang_to: str) -> str:
    """Constructs a prompt to ask the LLM to correct a line count mismatch."""
    translated_text: str = "\n".join([f"{idx+1}. {text}" for idx, text in enumerate(translated_lines)])

    event_text: str = "\n".join([
        f"{idx+1}. {item.text.replace('\\n', r'\\N')}"
        for idx, item in enumerate(orig_subevents)
    ])

    prompt = f"""Translate exactly {len(orig_subevents)} {_get_lang_name(lang_from)} subtitle entries to {_get_lang_name(lang_to)}.\nYour previous translation had {len(translated_lines)} entries, but I need exactly {len(orig_subevents)} entries.\n\nOriginal {_get_lang_name(lang_from)} entries:\n{event_text}\n\nYour previous translation with incorrect entry count:\n{translated_text}\n\nPlease correct your translation to provide exactly {len(orig_subevents)} entries of {_get_lang_name(lang_to)} subtitles.\nMaintain the same content but adjust your output to match the required entry count."""
    return prompt

def _set_context_prompt(subtitles: SSAFile, filename: str, lang_from: str, lang_to: str) -> str:
    """Constructs the prompt for generating a context summary from subtitles."""
    sub_events: str = "\n".join(i.text for i in subtitles.events)

    prompt: str = f"You are commissioning a translator and they have requested information in English. They will be provided with only the transcript attached. They will not be provided the source audio or video or other material. Make no reference to those. You are to: (1) State whether the transcript is high quality or if it was auto-generated with errors to inform the translator how strictly or liberally to follow the text, (2) State what the content is in 1 sentence, (3) Summarize the contents in 4 sentences, and (4) Under the header 'Glossary', provide a comprehensive simple mapping of jargon, names and titles from {_get_lang_name(lang_from)} to {_get_lang_name(lang_to)} for the translator to keep consistent, noting mis-spellings if any (format: '{_get_lang_name(lang_from)}: {_get_lang_name(lang_to)}').\nRespond in text format with no special formatting or numbering. Provide only the information requested. Be concise. Japanese names should be lastname-firstname order. Search the internet to ensure accuracy of the glossary.\n\nTranscript of Video: {filename}\n{sub_events}"

    return prompt

def _print_progress(elapsed_time: float, 
                    current_batch: int, 
                    total_batches: int, 
                    start_item: int, 
                    end_item: int, 
                    total_item: int,
                    def_time_per_batch: float = 40) -> None:
    """Calculates and prints translation progress and estimated time remaining."""
    
    # Time Calculations
    time_per_batch: float = (def_time_per_batch * 4 + elapsed_time) / (4 + current_batch - 1)
    remaining_time: float = time_per_batch * (total_batches - current_batch + 1)

    # Display progress
    remaining_minutes, remaining_seconds = divmod(int(remaining_time), 60)
    logger.info(f"Translating batch {current_batch}/{total_batches} ({start_item+1}-{end_item}/{total_item} items) - ETC: {remaining_minutes}m {remaining_seconds}s")
    return

def _log_response(prompt: str,
                  response: str,
                  current_batch: int, 
                  total_batches: int, 
                  start_item: int, 
                  end_item: int) -> None:
    """Logs a single LLM prompt and its response."""
    logger.debug(f"=== Batch {current_batch}/{total_batches} (Lines {start_item+1}-{end_item}) ===\n")
    logger.debug(f"PROMPT:\n{prompt}\n\n")
    logger.debug(f"RESPONSE:\n{response}\n\n")
    logger.debug("-" * 20 + "\n\n")
    return


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
        if current_line.startswith("..."):
            modified_lines[i+1] = current_line[3:]
        if next_line.startswith("..."):
            modified_lines[i+1] = next_line[3:]

    return modified_lines


def _force_line_count(translated_lines: list[str], 
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

def _get_lang_name(bcp47_code: str) -> str | None:
  """Gets the English display name for a BCP 47 language code."""
  try:
    lang = langcodes.Language.get(bcp47_code)
    return lang.display_name('en')
  except (LookupError, langcodes.tag_parser.LanguageTagError):
    return None