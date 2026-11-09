"""The text sent to the translation model. Everything here is a pure function."""
import logging

import langcodes
from langcodes.tag_parser import LanguageTagError
from pysubs2 import SSAEvent, SSAFile

from baberu import constants

logger = logging.getLogger(__name__)


def system_prompt(lang_from: str, lang_to: str) -> str:
    """Creates the system prompt for the LLM translator role."""
    prompt: str = f"You are a professional translator from {_get_lang_name(lang_from)} to natural colloquial {_get_lang_name(lang_to)}. Translate the provided subtitle entries liberally and concisely while preserving the meaning and nuance. Return ONLY the translated entries, maintaining EXACT entry count and order. Do not merge entries.\n\nSpecial instructions:\n1. '{constants.CONTINUE_FLAG}' indicates that text was split mid-sentence to be continued on the next entries. You must omit '{constants.CONTINUE_FLAG}' from the translated text.\n2. As per ASS syntax, use the special escape character '\\N' for line breaks within a subtitle entry.\n3. Use ASS syntax for styling if needed (e.g. {{\\i1}}italics{{\\i0}}). \n4. Do not split {_get_lang_name(lang_to)} words across subtitle entries."
    return prompt


def translate_prompt(previous_translations: list[str], 
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


def retry_prompt(translated_lines: list[str], 
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


def context_prompt(subtitles: SSAFile, filename: str, lang_from: str, lang_to: str) -> str:
    """Constructs the prompt for generating a context summary from subtitles."""
    sub_events: str = "\n".join(i.text for i in subtitles.events)

    prompt: str = f"You are commissioning a translator and they have requested information in English. They will be provided with only the transcript attached. They will not be provided the source audio or video or other material. Make no reference to those. You are to: (1) State whether the transcript is high quality or if it was auto-generated with errors to inform the translator how strictly or liberally to follow the text, (2) State what the content is in 1 sentence, (3) Summarize the contents in 4 sentences, and (4) Under the header 'Glossary', provide a comprehensive simple mapping of jargon, names and titles from {_get_lang_name(lang_from)} to {_get_lang_name(lang_to)} for the translator to keep consistent, noting mis-spellings if any.\nGlossary Format:\n'Correct {_get_lang_name(lang_from)}: {_get_lang_name(lang_to)} (Mis-spellings: Mis-spelled {_get_lang_name(lang_from)})')\nみりにゃ: Mirinya, nickname of Otani Emiri (Mis-spelled: みーにゃん)\n\nRespond in text format with no special formatting or numbering. Provide only the information requested. Be concise. Japanese names should be lastname-firstname order. Search the internet to ensure accuracy of the glossary.\n\nTranscript of Video: {filename}\n{sub_events}"

    return prompt


def _get_lang_name(bcp47_code: str) -> str:
  """Gets the English display name for a BCP 47 language code."""
  try:
    lang = langcodes.Language.get(bcp47_code)
    return lang.display_name('en')
  except (LookupError, LanguageTagError):
    logger.warning(f"Language code {bcp47_code} could not be parsed.")
    return bcp47_code
