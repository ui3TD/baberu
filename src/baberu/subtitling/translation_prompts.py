"""The text sent to the translation model. Everything here is a pure function."""
import logging

import langcodes
from langcodes.tag_parser import LanguageTagError
from pysubs2 import SSAEvent, SSAFile

from baberu import constants

logger = logging.getLogger(__name__)

RETRY_MARKER = "Translate only these entries, keeping their numbers:\n\n"


def system_prompt(lang_from: str, lang_to: str) -> str:
    """Creates the system prompt for the LLM translator role."""
    prompt: str = f"You are a professional translator from {_get_lang_name(lang_from)} to natural colloquial {_get_lang_name(lang_to)}. Translate the provided subtitle entries liberally and concisely while preserving the meaning and nuance. Return ONLY the translated entries, each with its number, maintaining EXACT entry count and order. Do not merge entries.\n\nSpecial instructions:\n1. '{constants.CONTINUE_FLAG}' indicates that text was split mid-sentence to be continued on the next entries. You must omit '{constants.CONTINUE_FLAG}' from the translated text.\n2. As per ASS syntax, use the special escape character '\\N' for line breaks within a subtitle entry.\n3. Use ASS syntax for styling if needed (e.g. {{\\i1}}italics{{\\i0}}). \n4. Do not split {_get_lang_name(lang_to)} words across subtitle entries."
    return prompt


def translate_prompt(previous_translations: list[str],
                     context_prompt: str,
                     current_batch: list[SSAEvent],
                     lang_from: str,
                     lang_to: str,
                     context_lines: int) -> str:
    """Constructs the user prompt for translating a batch of subtitles.

    `previous_translations` are earlier lines of this translation, never source lines.
    """
    return (f"{_preamble(previous_translations, context_prompt, context_lines)}"
            f"Translate the following {len(current_batch)} {_get_lang_name(lang_from)} subtitle entries to "
            f"{_get_lang_name(lang_to)}. Keep each entry's number. Maintain exact entry count and order:\n\n"
            f"{_entries(current_batch)}")


def targeted_retry_prompt(previous_translations: list[str],
                          context_prompt: str,
                          current_batch: list[SSAEvent],
                          accepted: dict[int, str],
                          needed: dict[int, str],
                          lang_from: str,
                          lang_to: str,
                          context_lines: int) -> str:
    """Asks again for only the entries that came back wrong. Keys are indexes in the batch.

    The whole batch and the translations already accepted are included, so the
    retried entries fit the lines around them.
    """
    source_name, target_name = _get_lang_name(lang_from), _get_lang_name(lang_to)
    accepted_text = "\n".join(f"{i + 1}. {accepted[i]}" for i in sorted(accepted)) or "(none)"
    reasons = "; ".join(f"entry {i + 1} was {reason}" for i, reason in sorted(needed.items()))
    wanted = sorted(needed)
    return (f"{_preamble(previous_translations, context_prompt, context_lines)}"
            f"These are {len(current_batch)} {source_name} subtitle entries:\n\n"
            f"{_entries(current_batch)}\n\n"
            f"Their {target_name} translations so far:\n\n{accepted_text}\n\n"
            f"Some entries are still needed ({reasons}). Translate each of them to {target_name} "
            f"so it fits the entries around it. Return one line per entry and nothing else. "
            f"{RETRY_MARKER}"
            f"{_entries([current_batch[i] for i in wanted], [i + 1 for i in wanted])}")


def _preamble(previous_translations: list[str], context_prompt: str, context_lines: int) -> str:
    section = f"Context:\n{context_prompt}\n\n"
    recent = previous_translations[-context_lines:] if context_lines > 0 else []
    if recent:
        section += (f"For continuity, here are the last {len(recent)} translated entries:\n"
                    + "\n".join(recent) + "\n\n\nContinue directly from the last entry. ")
    return section


def _entries(events: list[SSAEvent], numbers: list[int] | None = None) -> str:
    """Numbers subtitle entries for a prompt, one per line."""
    numbers = numbers or list(range(1, len(events) + 1))
    return "\n".join(f"{number}. {_one_line(event.text)}" for number, event in zip(numbers, events))


def _one_line(text: str) -> str:
    """Sends every line break, real or ASS soft '\\n', as the ASS hard break '\\N'."""
    return text.replace("\r\n", "\n").replace("\n", "\\N").replace("\\n", "\\N")


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
