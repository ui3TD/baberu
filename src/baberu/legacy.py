"""Brings in work from the old layout, where jobs had no folders of their own.

In that layout, files like `show.raw.ass` sit beside the media.
"""
import logging
from pathlib import Path
import shutil

from baberu.job import Job, Stage
from baberu.tools.file_utils import FileKind, file_hash, kind_of

logger = logging.getLogger(__name__)

_OLD_TAGS = (".padded_custom", ".tr_custom", ".fixed_custom", ".2pass_custom",
             ".padded", ".fixed", ".2pass", ".raw", ".partial", ".context", ".en")

_ANY_LANGUAGE = {".json": Stage.TRANSCRIPT, ".raw.ass": Stage.RAW, ".2pass.ass": Stage.TWOPASS}
_ENGLISH_ONLY = {".context.txt": Stage.CONTEXT, ".en.ass": Stage.TRANSLATED, ".partial.en.txt": Stage.PARTIAL}

_RELATED_KINDS = (FileKind.VIDEO, FileKind.AUDIO, FileKind.TRANSCRIPT, FileKind.SUBTITLE)


def job_name(path: Path) -> str | None:
    """The job an old-style subtitle or transcript belongs to.

    Returns a name only when the file has an old tag and other files from the
    same job are beside it; otherwise None.
    """
    if kind_of(path) not in (FileKind.SUBTITLE, FileKind.TRANSCRIPT):
        return None

    name = path.stem
    stripped = True
    while stripped:
        stripped = False
        for tag in _OLD_TAGS:
            if name.endswith(tag) and name != tag:
                name = name[:-len(tag)]
                stripped = True

    if name == path.stem or not path.parent.is_dir():
        return None

    for sibling in path.parent.iterdir():
        if sibling != path and (sibling.stem == name or sibling.stem.startswith(f"{name}.")) \
                and kind_of(sibling) in _RELATED_KINDS:
            return name
    return None


def adopt(job: Job, folders: list[Path], dry_run: bool) -> None:
    """Copies paid-for work from the old layout into a new job. Originals stay put."""
    if job.manifest_path.exists():
        return

    wanted = dict(_ANY_LANGUAGE)
    if job.lang_to == "en":
        wanted.update(_ENGLISH_ONLY)

    # `show.en.ass` is copied only when other files from the same old job exist.
    old_job_exists = any(_find(job.name + suffix, folders) for suffix in wanted if suffix != ".en.ass")

    copied = 0
    for suffix, stage in wanted.items():
        original = _find(job.name + suffix, folders)
        if original is None or (stage is Stage.TRANSLATED and not old_job_exists):
            continue
        destination = job.path(stage, whole_file=True)
        copied += 1
        if dry_run:
            job.adopted.add(destination)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)

        if stage is Stage.TRANSLATED:
            job.record_published(original.name, file_hash(original))
        if stage is Stage.PARTIAL:
            _adopt_partial_record(job, original, destination)

    if copied:
        logger.info(f"Found {copied} file(s) from an earlier version of baberu; copied them into {job.folder.name}.")
        if dry_run:
            english_result = job.result_dir / f"{job.name}.en.ass"
            if english_result.exists():
                job.record_published(english_result.name, file_hash(english_result))
        else:
            job.save()


def _find(name: str, folders: list[Path]) -> Path | None:
    for folder in dict.fromkeys(folders):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def _adopt_partial_record(job: Job, original: Path, destination: Path) -> None:
    record = original.with_name(original.name + ".source")
    if record.is_file():
        job.record_origin(destination, record.read_text(encoding="utf-8").strip())
