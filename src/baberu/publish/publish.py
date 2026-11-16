"""Uploads a finished translation to NekoCap, exactly as it is on disk.

Used by `baberu publish`, which runs no stages, and by `baberu ... --nekocap`
once its run has finished.
"""
from __future__ import annotations

from dataclasses import dataclass
from os import environ
from pathlib import Path
import json
import logging
import sys

from baberu import legacy
from baberu.errors import BaberuError
from baberu.job import Job, Stage
from baberu.providers.factory import AIToolFactory
from baberu.publish import nekocap
from baberu.settings import Settings
from baberu.subtitling import translation_prompts
from baberu.tools import av_utils
from baberu.tools.file_utils import FileKind, atomic_write_text, file_hash, is_url, kind_of
from baberu.transcription import keyterms

logger = logging.getLogger(__name__)


@dataclass
class PublishRequest:
    """What `baberu publish` was asked to do, already checked."""

    source: str
    output_dir: Path
    lang_from: str
    lang_to: str
    privacy: str | None = None   # overrides the config
    title: str | None = None     # replaces the saved title
    retitle: bool = False        # translates the title again
    dry_run: bool = False


@dataclass
class Upload:
    """One caption to upload, and how."""

    job: Job
    subtitles: Path
    input_file: Path
    lang_from: str
    lang_to: str
    privacy: str
    title: str | None = None
    retitle: bool = False


# --- Shared with the pipeline ---

def client(settings: Settings) -> nekocap.NekoCapClient:
    return nekocap.NekoCapClient(environ.get("NEKOCAP_SESSION_TOKEN", ""),
                                 settings.nekocap.server_url, settings.nekocap.app_id)


def video_info(job: Job, input_file: Path) -> dict:
    """The yt-dlp details saved in the job, or kept beside the file."""
    info = keyterms.load_metadata(job.source_info_path)
    if not info:
        sidecar = keyterms.find_info_json(input_file)
        info = keyterms.load_metadata(sidecar) if sidecar else {}
    return info


def youtube_id(job: Job, input_file: Path) -> str:
    info = video_info(job, input_file)
    url = info.get("webpage_url") or ""
    return info.get("id", "") if ("youtube.com" in url or "youtu.be" in url or not url) else ""


def record(job: Job) -> dict:
    try:
        return json.loads(job.nekocap_record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def check(job: Job, input_file: Path, settings: Settings) -> None:
    """Checks the video and the login before anything is spent. Reads only."""
    if not youtube_id(job, input_file):
        raise BaberuError("Publishing needs a YouTube video: run baberu on its URL, or keep yt-dlp's "
                          ".info.json beside the file.")
    client(settings).user_id()


def title_model(settings: Settings) -> str:
    return settings.nekocap.title_model or settings.translation.default_model


def translate_title(job: Job, title: str, lang_from: str, lang_to: str, settings: Settings) -> str:
    """Translates the title, with the job's notes in the prompt."""
    model = title_model(settings)
    if not model or not title:
        return title
    notes_file = job.path(Stage.CONTEXT)
    notes = notes_file.read_text(encoding="utf-8").strip() if notes_file.exists() else ""
    language = translation_prompts.language_name(lang_to)
    prompt = f"Translate this {translation_prompts.language_name(lang_from)} video title into {language}. " \
             "Keep it short. Reply with the title only."
    if notes:
        prompt += " The subtitles were translated with the notes below; spell names and terms exactly as " \
                  f"their glossary does.\n\nNotes:\n{notes}"
    llm = AIToolFactory.get_llm_provider(model, effort=settings.translation.effort,
                                         max_retries=settings.translation.server_retries)
    reply = llm.prompt(f"{prompt}\n\nTitle:\n{title}")
    return reply.strip().strip('"') or title


def describe(job: Job, settings: Settings, privacy: str | None) -> str:
    """For the plan: privacy, and whether it replaces an earlier upload."""
    earlier = record(job)
    return f"{privacy or settings.nekocap.privacy}, " + ("replaces your upload" if earlier else "new")


def upload(item: Upload, settings: Settings, dry_run: bool = False) -> str | None:
    """Uploads the file, or replaces the user's earlier upload of it. Returns the caption's page."""
    video_id = youtube_id(item.job, item.input_file)
    if not video_id:
        raise BaberuError("Publishing needs a YouTube video: run baberu on its URL, or keep yt-dlp's "
                          ".info.json beside the file.")
    neko = client(settings)
    neko.user_id()
    earlier = record(item.job)
    info = video_info(item.job, item.input_file)
    original_title = info.get("title", "")
    digest = file_hash(item.subtitles)

    title = item.title or ("" if item.retitle else earlier.get("title", ""))
    existing = neko.find_own_caption(video_id, item.lang_to)
    # Skips when the caption uploaded last is still on the site and nothing has changed.
    if existing and existing.id == earlier.get("captionId") and earlier.get("hash") == digest \
            and earlier.get("privacy") == item.privacy and title == earlier.get("title"):
        page = nekocap.caption_url(existing.id)
        logger.info(f"nekocap: already up to date ({page})")
        return page

    action = "replace your caption" if existing else "upload"
    if dry_run:
        shown = title or f"(translated from \"{original_title}\" with {title_model(settings)} when uploading)"
        print(f"Would {action} ({item.privacy}): {item.subtitles}", file=sys.stderr)
        print(f"Title: {shown}", file=sys.stderr)
        return None

    if not title:
        title = translate_title(item.job, original_title, item.lang_from, item.lang_to, settings)
    logger.info(f"nekocap: {'replacing your caption' if existing else 'uploading'} ({item.privacy}): {title}")
    video_language = settings.nekocap.video_languages.get(item.lang_from, item.lang_from)
    content = item.subtitles.read_text(encoding="utf-8-sig")
    caption_id = neko.upload(content, video_id, original_title, title, item.lang_to, video_language,
                             item.privacy, existing)
    atomic_write_text(item.job.nekocap_record_path, json.dumps({
        "captionId": caption_id, "videoId": video_id, "language": item.lang_to,
        "privacy": item.privacy, "title": title, "hash": digest,
    }, ensure_ascii=False, indent=2))
    page = nekocap.caption_url(caption_id)
    logger.info(f"nekocap: done, {page}")
    return page


# --- baberu publish ---

def _locate(request: PublishRequest) -> tuple[Job, Path, Path]:
    """Finds the job and its result from a URL, the media, or the result itself."""
    if is_url(request.source):
        media = av_utils.find_download(request.source, request.output_dir)
        if not media:
            raise BaberuError(f"No download of {request.source} was found in {request.output_dir}. "
                              "Run baberu on it first, or give the result file.")
        source = media
    else:
        source = Path(request.source)
        if not source.is_file():
            raise BaberuError(f"Input file does not exist: {request.source}")

    kind = kind_of(source)
    if kind not in (FileKind.VIDEO, FileKind.AUDIO, FileKind.SUBTITLE):
        raise BaberuError(f"baberu publish takes a URL, the video, or its translated subtitles, not {source.name}.")

    job = Job.locate(source, request.output_dir, request.lang_from, request.lang_to,
                     translating=True, legacy_name=legacy.job_name(source))
    if not job.manifest:
        raise BaberuError(f"{source.name} has no baberu work folder in {request.output_dir}. "
                          f"Run baberu on it first.")
    if kind is FileKind.SUBTITLE:
        return job, source, source

    result = job.result_path(source, "padded")
    if not result.exists():
        raise BaberuError(f"There's no translation to publish: {result} doesn't exist. "
                          f"Run baberu on {source.name} first, or give the subtitle file.")
    return job, result, source


def publish_command(request: PublishRequest, settings: Settings) -> None:
    job, subtitles, input_file = _locate(request)
    if job.published_hash(subtitles.name) not in (None, file_hash(subtitles)):
        logger.info(f"publish: {subtitles.name} was edited by hand; uploading it as it is")
    item = Upload(job, subtitles, input_file, request.lang_from, request.lang_to,
                  request.privacy or settings.nekocap.privacy, request.title, request.retitle)
    page = upload(item, settings, request.dry_run)
    if page:
        print(page)
