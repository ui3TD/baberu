"""Builds the list of names and jargon sent to the transcription service as keyterms.

Terms come from three places, most specific first:
  1. A model reading the video's title, description and tags (optional, paid).
  2. The channel's file: <dir>/<name>.yaml, the one name config.yaml maps the channel to,
     plus any file given with --keyterms.
  3. The video's own tags, and phrases quoted in its title.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import logging
import re

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from baberu.errors import BaberuError
from baberu.providers.llm.base import LLMProvider
from baberu.tools.file_utils import atomic_write_text

logger = logging.getLogger(__name__)

# Limits of ElevenLabs Scribe.
MAX_KEYTERMS = 1000
MAX_CHARS = 50
MAX_WORDS = 5
FORBIDDEN_CHARS = set("<>{}[]\\")

# What is kept of yt-dlp's metadata. The rest is formats and thumbnails.
METADATA_FIELDS = ("id", "title", "channel", "channel_id", "uploader", "tags", "description", "webpage_url")

# Phrases set off in a title: 『とくべチューハイ』, 【乾杯】, 「...」.
_QUOTED = re.compile(r"[『「【《〈\"]([^』」】》〉\"]{2,50})[』」】》〉\"]")


def clean(terms: Iterable[str], max_terms: int = 100) -> list[str]:
    """Trims and de-duplicates terms in order, dropping any the service would reject."""
    kept: list[str] = []
    seen: set[str] = set()
    for raw in terms:
        term = " ".join(str(raw).split())
        if not term or term.casefold() in seen:
            continue
        problem = _problem(term)
        if problem:
            logger.warning(f"keyterms: dropping {term!r} ({problem})")
            continue
        seen.add(term.casefold())
        kept.append(term)

    limit = min(max_terms, MAX_KEYTERMS)
    if len(kept) > limit:
        logger.info(f"keyterms: keeping the first {limit} of {len(kept)} terms")
    return kept[:limit]


def _problem(term: str) -> str:
    if len(term) > MAX_CHARS:
        return f"over {MAX_CHARS} characters"
    if len(term.split()) > MAX_WORDS:
        return f"over {MAX_WORDS} words"
    if FORBIDDEN_CHARS.intersection(term):
        return "has one of < > { } [ ] \\"
    return ""


# --- Video metadata ---

def trim_metadata(info: dict[str, Any]) -> dict[str, Any]:
    return {key: info[key] for key in METADATA_FIELDS if info.get(key)}


def save_metadata(info: dict[str, Any], path: Path) -> None:
    atomic_write_text(path, json.dumps(trim_metadata(info), ensure_ascii=False, indent=2))


def load_metadata(path: Path) -> dict[str, Any]:
    """Reads saved metadata, or a yt-dlp .info.json. Missing or unreadable gives {}."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return trim_metadata(data) if isinstance(data, dict) else {}


def find_info_json(media_file: Path) -> Path | None:
    """The <name>.info.json that yt-dlp's --write-info-json puts beside the download, or None."""
    candidate = media_file.with_name(f"{media_file.stem}.info.json")
    return candidate if candidate.is_file() else None


def from_metadata(info: dict[str, Any]) -> list[str]:
    """The video's tags, and the phrases quoted in its title."""
    quoted = _QUOTED.findall(info.get("title") or "")
    return [*quoted, *(info.get("tags") or [])]


# --- The channel's file ---

class MissingChannelTerms(BaberuError):
    """The channel has no keyterms file. Nothing is transcribed until one is made."""


class ChannelTerms(BaseModel):
    """keyterms/<name>.yaml: the channel, and every term used for its videos."""
    model_config = ConfigDict(extra="forbid", strict=True)
    channel_id: str
    channel: str
    terms: list[str]


def channel_file(info: dict[str, Any], channels: dict[str, str], terms_dir: Path) -> Path:
    """The one file the config maps the video's channel to. Raises MissingChannelTerms if there is none."""
    channel_id = info.get("channel_id") or ""
    channel = (info.get("channel") or info.get("uploader") or "").strip() or "This channel"
    name = channels.get(channel_id)
    if not name:
        raise MissingChannelTerms(
            f"{channel} ({channel_id}) has no keyterms file. Add `{channel_id}: <name>` under "
            f"keyterms.channels in the config, and write {terms_dir / '<name>.yaml'}.")
    path = terms_dir / f"{name}.yaml"
    if not path.is_file():
        raise MissingChannelTerms(f"{channel} ({channel_id}) is mapped to '{name}', but {path} doesn't exist.")
    return path


def load_channel(path: Path, channel_id: str | None = None) -> ChannelTerms:
    """Reads a keyterms file. With channel_id, it must be that channel's file."""
    try:
        terms = ChannelTerms.model_validate(yaml.safe_load(path.read_text(encoding="utf-8-sig")))
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise BaberuError(f"Keyterms file {path} is not valid: {error}") from error
    if channel_id and terms.channel_id != channel_id:
        raise BaberuError(f"{path} is for channel {terms.channel_id}, not {channel_id}.")
    return terms


def digest(terms: list[str]) -> str:
    """A short hash that identifies a set of listed terms."""
    return hashlib.sha256("\n".join(terms).encode("utf-8")).hexdigest()[:16]


# --- Asking a model ---

EXTRACT_SYSTEM_PROMPT = "You prepare vocabulary hints for a speech recognition system. Follow the format exactly."


def extract_prompt(info: dict[str, Any], seeds: list[str], lang_name: str, limit: int) -> str:
    seed_text = "\n".join(seeds) if seeds else "(none)"
    return (
        f"A {lang_name} video is about to be transcribed by a speech recognition system that accepts "
        f"a list of keyterms: words it should recognise and spell correctly.\n\n"
        f"From the video's details below, list up to {limit} keyterms that are likely to be SPOKEN in the video "
        f"and that a speech recogniser could get wrong: names of people (with common nicknames), groups, "
        f"products, places, shows, and jargon.\n"
        f"- Write each term as it would be written in a {lang_name} transcript.\n"
        f"- Put the terms most specific to this video first, e.g. the people who appear in it.\n"
        f"- From the known terms, include only those relevant to this video.\n"
        f"- Leave out ordinary words, hashtags, URLs, and sponsor boilerplate.\n"
        f"- Each term at most 5 words and 50 characters, without < > {{ }} [ ] or backslash.\n"
        f"- Reply with one term per line and nothing else: no numbering, notes or translations.\n\n"
        f"Title: {info.get('title', '')}\n"
        f"Channel: {info.get('channel') or info.get('uploader') or ''}\n"
        f"Tags: {', '.join(info.get('tags') or [])}\n"
        f"Description:\n{info.get('description', '')}\n\n"
        f"Known terms for this channel or topic:\n{seed_text}"
    )


def parse_reply(text: str) -> list[str]:
    """One term per line; bullets, numbering and stray quoting are removed."""
    terms = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•・]|\d+[.)])\s*", "", line).strip().strip("`\"'")
        if line and not line.endswith(":"):
            terms.append(line)
    return terms


def extract(info: dict[str, Any], seeds: list[str], llm: LLMProvider, lang_name: str, limit: int) -> list[str]:
    completion = llm.complete(extract_prompt(info, seeds, lang_name, limit), EXTRACT_SYSTEM_PROMPT)
    terms = parse_reply(completion.text)
    if not terms:
        logger.warning(f"keyterms: {llm.model} suggested nothing ({completion.status}); using the lists and tags.")
    return terms


# --- The job's list ---

def build(info: dict[str, Any],
          listed: list[str],
          max_terms: int,
          llm: LLMProvider | None = None,
          lang_name: str = "") -> list[str]:
    """The model's picks first, then the channel's terms, then the tags."""
    from_video = from_metadata(info)
    picked = extract(info, clean(listed + from_video, MAX_KEYTERMS), llm, lang_name, max_terms) \
        if llm and info else []
    return clean([*picked, *listed, *from_video], max_terms)


def write(terms: list[str], path: Path) -> None:
    """One term per line."""
    atomic_write_text(path, "\n".join(terms) + "\n")


def load(path: Path, max_terms: int) -> list[str]:
    if not path.is_file():
        return []
    return clean(path.read_text(encoding="utf-8-sig").splitlines(), max_terms)
