"""Works out what a run will do, checks it can succeed, then does it."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
import hashlib
import json
import logging
import shutil
import sys

import ffmpeg

from baberu.providers.factory import AIToolFactory
from baberu.providers.llm import base as llm_base
from baberu.providers.llm.base import LLMProvider
from baberu.publish import nekocap, publish
from pysubs2 import SSAFile

from baberu import legacy
from baberu.constants import CODEC_TO_EXTENSION_MAP
from baberu.errors import BaberuError
from baberu.job import Job, Stage, enclosing_job
from baberu.settings import Settings, TranslationSettings
from baberu.subtitling import frames, sub_correction, sub_translation, sub_utils, translation_prompts
from baberu.subtitling.types import LineRange
from baberu.tools import av_utils
from baberu.tools.file_utils import FileKind, atomic_path, atomic_write_text, file_hash, kind_of
from baberu.transcription import health, keyterms, repair, transcript_chunked, transcript_conversion, transcript_segmented

if TYPE_CHECKING:
    from baberu.cli import RunRequest

logger: logging.Logger = logging.getLogger(__name__)

RUN, REUSE = "run", "reuse"
_PAID = {"transcribe", "retranscribe", "context", "translate"}
_NEEDS_FFMPEG = {"extract", "transcribe", "retranscribe", "frames", "hardcode", "audio_to_video"}
_SUBTITLE_STAGE_NAMES = (("pad", "padded"), ("translate", "translated"), ("fix", "fixed"),
                         ("retranscribe", "2pass"), ("convert", "raw"))


@dataclass
class PlannedStage:
    name: str
    action: str
    detail: str = ""
    paid: bool = False  # for a stage that costs money only with some settings

    @property
    def spends_money(self) -> bool:
        return self.action == RUN and (self.name in _PAID or self.paid)


@dataclass
class Plan:
    stages: list[PlannedStage]
    input_file: Path
    video: Path | None
    audio: Path | None
    subtitle_target: Path | None = None
    video_target: Path | None = None
    hardcode_video: Path | None = None
    hardcode_subs: Path | None = None
    result_stage: str = ""
    # The file each subtitle stage will read, when it's already on disk before the run starts.
    inputs: dict[str, Path | None] = field(default_factory=dict)
    # --lines with -t on a job that has a whole translation: only those lines are translated again.
    in_place: bool = False
    # The video the frames stage reads, and the height of its frames.
    frames_video: Path | None = None
    frames_height: int = 0
    # The note shown when a translation runs without frames, e.g. "no frames: show.webm is 144p, ...".
    frames_off: str = ""

    def action(self, name: str) -> str | None:
        return next((stage.action for stage in self.stages if stage.name == name), None)

    @property
    def spends_money(self) -> bool:
        return any(stage.spends_money for stage in self.stages)


# --- Planning ---

def _sibling(job: Job, input_file: Path, kind: FileKind) -> Path | None:
    """Finds media with the same name as the input, however its extension is spelled."""
    folder = input_file.parent
    if not folder.is_dir():
        return None
    for entry in sorted(folder.iterdir()):
        if entry.stem == job.name and kind_of(entry) is kind and entry.is_file():
            return entry
    return None


def _duration_detail(path: Path | None) -> str:
    seconds = av_utils.duration_seconds(path) if path else None
    if not seconds:
        return ""
    return f"{round(seconds / 60)} min of audio" if seconds >= 60 else "under 1 min of audio"


def _line_count_detail(request: RunRequest, job: Job) -> str:
    source = request.sub_file or job.path(Stage.RAW)
    if not source.exists():
        return ""
    return f"{len(sub_utils.load(source).events)} lines"


def _wants_redo(request: RunRequest, name: str) -> bool:
    return name in request.redo or "all" in request.redo


def _reuse_or_run(request: RunRequest, job: Job, name: str, output: Path | None) -> str:
    if _wants_redo(request, name) or output is None or not job.has(output):
        return RUN
    return REUSE


def _find_media(request: RunRequest, job: Job, input_file: Path, video: Path | None, audio: Path | None) -> tuple[Path | None, Path | None, bool]:
    """Fills in audio for subtitles or transcripts. Returns video, audio, retranscribe."""
    retranscribe = request.retranscribe
    if not retranscribe or audio or video or (not request.sub_file and not request.json_file):
        return video, audio, retranscribe

    audio = job.find_audio() or _sibling(job, input_file, FileKind.AUDIO)
    if not audio:
        video = _sibling(job, input_file, FileKind.VIDEO)

    if not audio and not video:
        if request.retranscribe_required:
            raise BaberuError(f"-r needs audio, and none was found for {input_file.name}. Use --audio FILE.")
        logger.info("Skipping retranscribe: no audio found for this file.")
        retranscribe = False
    return video, audio, retranscribe


def _build_plan(request: RunRequest, settings: Settings, job: Job, input_file: Path, video: Path | None,
                audio: Path | None, in_place: bool = False) -> Plan:
    stages: list[PlannedStage] = []
    video, audio, retranscribe = _find_media(request, job, input_file, video, audio)

    extracting = bool(video and not audio and (request.extract or retranscribe))
    if extracting:
        stages.append(PlannedStage("extract", _reuse_or_run(request, job, "extract", job.find_audio())))

    will_have_audio = bool(audio) or extracting
    if request.speech_to_text and will_have_audio:
        stages.append(PlannedStage(
            "transcribe", _reuse_or_run(request, job, "transcribe", job.path(Stage.TRANSCRIPT)),
            _duration_detail(audio or video)))

    has_transcript = any(s.name == "transcribe" for s in stages) or bool(request.json_file)
    if request.convert and has_transcript:
        # Converts again when a new transcript is made.
        transcribing = any(s.name == "transcribe" and s.action == RUN for s in stages)
        action = RUN if request.json_file or transcribing else _reuse_or_run(request, job, "convert", job.path(Stage.RAW))
        stages.append(PlannedStage("convert", action))

    has_subtitles = bool(request.sub_file) or any(s.name == "convert" for s in stages)
    # Optional stages that made the job's saved work stay in the plan when a later stage is asked for.
    from_transcript = any(s.name == "convert" for s in stages)
    later_than_retranscribe = request.fix or request.translate or request.pad
    if has_subtitles and ((retranscribe and will_have_audio) or
                          (from_transcript and later_than_retranscribe and job.has(job.path(Stage.TWOPASS)))):
        stages.append(PlannedStage("retranscribe", _reuse_or_run(request, job, "retranscribe", job.path(Stage.TWOPASS))))
    if has_subtitles and (request.fix or (from_transcript and (request.translate or request.pad)
                                          and job.has(job.path(Stage.FIXED)))):
        stages.append(PlannedStage("fix", RUN))
    planned_frames = _PlannedFrames()
    if has_subtitles and request.translate:
        translate = RUN if in_place else _reuse_or_run(request, job, "translate", job.path(Stage.TRANSLATED))
        if translate == RUN:
            planned_frames = _plan_frames(request, settings, job, input_file, video)
            if planned_frames.stage:
                stages.append(planned_frames.stage)
        if translate == RUN and not request.context_file:
            stages.append(PlannedStage("context", _reuse_or_run(request, job, "context", job.path(Stage.CONTEXT))))
        detail = f"lines {request.line_range.label[1:]}, in place" if in_place else _line_count_detail(request, job)
        detail = ", ".join(filter(None, [detail, planned_frames.off]))
        stages.append(PlannedStage("translate", translate, detail))
    if has_subtitles and (request.pad or (request.translate and job.has(job.path(Stage.PADDED)))):
        stages.append(PlannedStage("pad", RUN))
    _plan_keyterms(request, settings, job, stages)

    plan = Plan(stages, input_file, video, audio, in_place=in_place, frames_video=planned_frames.video,
                frames_height=planned_frames.height, frames_off=planned_frames.off)
    plan.inputs = _plan_inputs(request, settings, job, stages)
    names = {stage.name for stage in stages}

    plan.result_stage = next((label for name, label in _SUBTITLE_STAGE_NAMES if name in names), "")
    if plan.result_stage:
        job_result = job.result_path(input_file, plan.result_stage)
        plan.subtitle_target = job_result
        if request.output_folder:
            plan.subtitle_target = request.output_folder / job_result.name
        elif request.output_file and not (request.hardcode or request.image_file):
            plan.subtitle_target = request.output_file

    _plan_video_output(request, job, plan, has_subtitles, input_file)
    if request.nekocap and plan.action("translate"):
        plan.stages.append(PlannedStage("nekocap", RUN, publish.describe(job, settings, request.nekocap_privacy)))
    return plan


@dataclass
class _PlannedFrames:
    """The frames stage, the video it reads and the frames' height; or, in `off`, the note for a run without frames."""
    stage: PlannedStage | None = None
    video: Path | None = None
    height: int = 0
    off: str = ""


def _find_video(job: Job, input_file: Path, video: Path | None) -> Path | None:
    """The video given, else one beside the input with the job's name, else the job's recorded source."""
    if video:
        return video
    sibling = _sibling(job, input_file, FileKind.VIDEO)
    if sibling:
        return sibling
    source = job.manifest.get("source")
    for folder in dict.fromkeys([input_file.parent, job.folder.parent]):
        candidate = folder / source if source else None
        if candidate and candidate.is_file() and kind_of(candidate) is FileKind.VIDEO:
            return candidate
    return None


def _takes_images(model: str) -> bool:
    try:
        return AIToolFactory.get_llm_provider(model).takes_images()
    except ValueError:
        return False


def _cuts_match(job: Job, video: Path, threshold: float) -> bool:
    """Whether the saved scene cuts were found in this video with this threshold."""
    origin = job.origin_of(job.scenes_path)
    return bool(origin) and job.scenes_path.exists() and origin.get("source") == video.name \
        and (origin.get("options") or {}).get("threshold") == threshold


def _plan_frames(request: RunRequest, settings: Settings, job: Job, input_file: Path,
                 video: Path | None) -> _PlannedFrames:
    """Plans the frames stage for a translation, or gives the note for a run without frames. Reads only."""
    options = settings.frames
    if not options.enabled:
        return _PlannedFrames()
    video = _find_video(job, input_file, video)
    if video is None:
        return _PlannedFrames(off="no frames: no video found")
    height = av_utils.video_height(video)
    if height is None:
        return _PlannedFrames(off=f"no frames: can't read the size of {video.name}")
    if height < options.min_height:
        return _PlannedFrames(off=f"no frames: {video.name} is {height}p, under {options.min_height}p "
                                  f"(delete it to download it again at {options.height}p)")
    if not _takes_images(request.model):
        return _PlannedFrames(off=f"no frames: {request.model} doesn't take images")

    action = _reuse_or_run(request, job, "frames", job.scenes_path)
    if action == REUSE and not _cuts_match(job, video, options.scene_threshold):
        action = RUN
    frame_height = min(height, options.height)
    return _PlannedFrames(PlannedStage("frames", action, f"{frame_height}p, up to {options.max_per_batch} a batch"),
                          video, frame_height)


_SUBTITLE_OUTPUTS = {"retranscribe": Stage.TWOPASS, "fix": Stage.FIXED, "translate": Stage.TRANSLATED,
                     "pad": Stage.PADDED}


def _plan_inputs(request: RunRequest, settings: Settings, job: Job, stages: list[PlannedStage]) -> dict[str, Path | None]:
    """Follows the subtitles through the plan while they're files already on disk.

    Marks fix and pad REUSE when their saved output was made from the same input with
    the same settings; the run checks again before using it.
    """
    inputs: dict[str, Path | None] = {}
    current = request.sub_file
    for stage in stages:
        if stage.name == "convert":
            raw = job.path(Stage.RAW)
            current = raw if stage.action == REUSE and raw.exists() else None
            continue
        if stage.name not in _SUBTITLE_OUTPUTS:
            continue
        inputs[stage.name] = current
        output = job.path(_SUBTITLE_OUTPUTS[stage.name])
        if stage.name in _FREE_STAGES and current is not None:
            key = _FREE_STAGES[stage.name](settings, job.line_range)
            if _free_output_matches(job, output, sub_utils.timed_fingerprint(sub_utils.load(current)), key):
                stage.action = REUSE
        current = output if stage.action == REUSE and current is not None and output.exists() else None
    return inputs


def _fix_settings(settings: Settings, line_range: LineRange | None) -> str:
    return _settings_key([settings.mistimed_lines.model_dump(), settings.mistimed_segs.model_dump(),
                          settings.anomalies.max_letters_per_sec], line_range)


def _pad_settings(settings: Settings, line_range: LineRange | None) -> str:
    return _settings_key([settings.subtitle_padding.model_dump()], line_range)


def _settings_key(parts: list, line_range: LineRange | None) -> str:
    text = json.dumps([parts, line_range.label if line_range else None], sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


_FREE_STAGES = {"fix": _fix_settings, "pad": _pad_settings}


def _free_output_matches(job: Job, output: Path, source: str, key: str) -> bool:
    """Whether a free stage's saved output was made from this input with these settings."""
    origin = job.origin_of(output)
    return bool(origin) and output.exists() and origin.get("source") == source \
        and (origin.get("options") or {}).get("settings") == key


def _keyterms_path(job: Job) -> Path:
    return job.folder / "keyterms.txt"


def _source_info_path(job: Job) -> Path:
    return job.source_info_path


def _takes_keyterms(model: str) -> bool:
    try:
        return AIToolFactory.get_transcription_provider_type(model).supports_keyterms
    except ValueError:
        return False


def _keyterm_users(settings: Settings) -> set[str]:
    """The transcription stages that would send keyterms."""
    users = {"transcribe"} if _takes_keyterms(settings.transcription.model) else set()
    if settings.keyterms.retranscribe and _takes_keyterms(_retranscribe_model(settings)):
        users.add("retranscribe")
    return users


def _plan_keyterms(request: RunRequest, settings: Settings, job: Job, stages: list[PlannedStage]) -> None:
    """Keyterms come just before the first transcription that will send them."""
    if not request.use_keyterms:
        return
    users = _keyterm_users(settings)
    first = next((i for i, s in enumerate(stages) if s.name in users and s.action == RUN), None)
    if first is None:
        return
    action = _reuse_or_run(request, job, "keyterms", _keyterms_path(job))
    listed = _listed_terms(request, settings, job, required=False)
    if action == REUSE and listed and _keyterms_are_stale(job, listed):
        action = RUN
    sources = [*(listed.files if listed else ["the channel's file"]), "tags"]
    if settings.keyterms.llm_model:
        sources.insert(0, settings.keyterms.llm_model)
    stages.insert(first, PlannedStage("keyterms", action, "from " + ", ".join(sources),
                                      paid=bool(settings.keyterms.llm_model)))


def _plan_video_output(request: RunRequest, job: Job, plan: Plan, has_subtitles: bool, input_file: Path) -> None:
    default_name = None
    if request.hardcode:
        plan.hardcode_video = request.hardcode_video or plan.video or _sibling(job, input_file, FileKind.VIDEO)
        if plan.result_stage:
            plan.hardcode_subs = plan.subtitle_target
        else:
            plan.hardcode_subs = request.hardcode_subs or request.sub_file or job.earlier_result()
        plan.stages.append(PlannedStage("hardcode", RUN))
        default_name = f"{job.name}.subbed.mp4"
    elif request.image_file and (plan.audio or job.find_audio() or plan.action("extract")):
        plan.stages.append(PlannedStage("audio_to_video", RUN))
        default_name = f"{job.name}.mp4"

    if default_name:
        plan.video_target = request.output_file or (request.output_folder or job.result_dir) / default_name


# --- Checks made before anything is spent ---

def _is_same_file(first: Path | None, second: Path | None) -> bool:
    return bool(first and second) and first.absolute() == second.absolute()


def _preflight(request: RunRequest, settings: Settings, job: Job, plan: Plan) -> None:
    if not plan.stages:
        raise BaberuError(f"Nothing to do for {plan.input_file.name} with those options.")

    if request.hardcode:
        if not plan.hardcode_video:
            raise BaberuError("--hardcode needs a video, and none was found. Use --video FILE.")
        if not plan.hardcode_subs:
            raise BaberuError("--hardcode needs subtitles, and none were found. Create them first, or use --subs FILE.")

    running = {stage.name for stage in plan.stages if stage.action == RUN}
    if running & _NEEDS_FFMPEG and not shutil.which("ffmpeg"):
        raise BaberuError("ffmpeg was not found. Install it and make sure it's on your PATH.")

    _check_api_keys(request, settings, plan)
    if plan.action("keyterms") == RUN:
        _listed_terms(request, settings, job)  # raises on a missing or broken file
    if request.nekocap:
        _check_nekocap(request, settings, job, plan)
    # Also checked when the stage runs.
    if plan.action("transcribe") == REUSE:
        transcript = job.path(Stage.TRANSCRIPT)
        _guard_paid_work(job, transcript, None, settings.transcription.model, "transcribe")
        _guard_options(job, transcript, settings.transcription.options, "transcribe")
        listed = _listed_terms(request, settings, job, required=False) if request.use_keyterms else None
        if listed and _keyterms_are_stale(job, listed):
            logger.warning(f"{', '.join(listed.files)} changed since this job's transcript was made. "
                           f"Use --redo keyterms --redo transcribe to transcribe with the new terms, which costs money.")
    for name, model in (("retranscribe", _retranscribe_model(settings)), ("translate", request.model)):
        source_file = plan.inputs.get(name)
        if plan.action(name) == REUSE and source_file is not None:
            _guard_paid_work(job, job.path(_SUBTITLE_OUTPUTS[name]), sub_utils.fingerprint(sub_utils.load(source_file)),
                             model, name)

    for target in (plan.subtitle_target, plan.video_target):
        if _is_same_file(target, plan.input_file) or _is_same_file(target, plan.hardcode_video):
            raise BaberuError(f"The output would replace its own input: {target}")

    if plan.video_target and plan.video_target.exists() and not request.overwrite:
        raise BaberuError(f"{plan.video_target} already exists. Use --overwrite to replace it.")

    target = plan.subtitle_target
    if target and target.exists() and not request.overwrite and plan.spends_money \
            and file_hash(target) != job.published_hash(target.name):
        raise BaberuError(f"{target} was changed outside baberu, and this run would replace it. "
                          f"Use --overwrite to replace it, or `baberu publish` to upload it as it is.")


def _check_api_keys(request: RunRequest, settings: Settings, plan: Plan) -> None:
    """Builds the providers the plan will run, which checks their API keys. Makes no request."""
    try:
        if plan.action("transcribe") == RUN:
            AIToolFactory.get_transcription_provider(settings.transcription.model, settings.transcription.options)
        if plan.action("retranscribe") == RUN:
            AIToolFactory.get_transcription_provider(_retranscribe_model(settings), _retranscribe_options(settings))
        if plan.action("translate") == RUN:
            AIToolFactory.get_llm_provider(request.model)
        if plan.action("keyterms") == RUN and settings.keyterms.llm_model:
            AIToolFactory.get_llm_provider(settings.keyterms.llm_model)
        if plan.action("context") == RUN:
            websearch = AIToolFactory.get_llm_provider(settings.translation.websearch_model)
            if not websearch.supports_grounding:
                raise BaberuError(
                    f"websearch_model {settings.translation.websearch_model} can't search the web. "
                    f"Choose a model that can, or give your own notes with a context file.")
    except ValueError as error:
        raise BaberuError(str(error)) from error


def _check_nekocap(request: RunRequest, settings: Settings, job: Job, plan: Plan) -> None:
    """Checks the login and the video before anything is spent. Reads only."""
    if not plan.action("nekocap"):
        raise BaberuError("--nekocap uploads a translation, and this run doesn't make one. "
                          "To upload an existing one, use `baberu publish`.")
    if job.line_range:
        raise BaberuError("--nekocap uploads the whole translation, so it can't be used with a separate --lines result.")
    publish.check(job, plan.input_file, settings)


def _print_plan(plan: Plan, job: Job) -> None:
    print(f"Plan for {plan.input_file.name}:", file=sys.stderr)
    for stage in plan.stages:
        notes = ", ".join(filter(None, ["paid" if stage.spends_money else "", stage.detail]))
        print(f"  {stage.name:<15}{stage.action:<7}{notes}", file=sys.stderr)
    for target in (plan.subtitle_target, plan.video_target):
        if target:
            edited = target.exists() and file_hash(target) != job.published_hash(target.name)
            print(f"Result: {target}" + (" (changed outside baberu)" if edited else ""), file=sys.stderr)
    uploaded = publish.record(job)
    if uploaded.get("captionId"):
        print(f"On NekoCap: {nekocap.caption_url(uploaded['captionId'])} ({uploaded.get('privacy')}): "
              f"{uploaded.get('title', '')}", file=sys.stderr)


def _preview_url(url: str) -> None:
    info = av_utils.probe_url(url)
    minutes = round((info.get("duration") or 0) / 60)
    print(f"Would download: {info.get('title', url)} ({minutes} min)", file=sys.stderr)
    print("The rest of the plan is decided once the file has been downloaded.", file=sys.stderr)


# --- Redo ---

def _clear_redone_work(request: RunRequest, job: Job, plan: Plan) -> None:
    if _wants_redo(request, "extract"):
        audio = job.find_audio()
        if audio:
            audio.unlink()
    if _wants_redo(request, "keyterms"):
        job.clear(_keyterms_path(job))
    if _wants_redo(request, "transcribe"):
        job.clear(job.path(Stage.TRANSCRIPT))
    if _wants_redo(request, "convert"):
        job.clear(job.path(Stage.RAW))
    if _wants_redo(request, "retranscribe"):
        job.clear(job.path(Stage.TWOPASS))
        job.clear_segments()
    if _wants_redo(request, "frames"):
        job.clear_frames()
    if _wants_redo(request, "context"):
        job.clear(job.path(Stage.CONTEXT))
    if _wants_redo(request, "translate"):
        if plan.in_place:   # only the lines asked for are translated again
            partial, report = _in_place_paths(job, request.line_range)
            job.clear(partial, report)
        else:
            job.clear(job.path(Stage.TRANSLATED), job.path(Stage.PARTIAL), _translation_report_path(job))
    if request.redo:
        job.save()


# --- Stages ---

def _announce_reuse(stage: str, path: Path) -> None:
    logger.info(f"{stage}: reusing {path.name} (--redo {stage} to do it again)")


def _shift(line_range: LineRange | None, before: int, sub_data: SSAFile) -> LineRange | None:
    """Keeps a line range on the same lines after a stage changes the line count."""
    return line_range.resized(len(sub_data.events) - before) if line_range else None


def _guard_paid_work(job: Job, output: Path, source: str | None, model: str | None, redo_name: str) -> None:
    """Refuses to reuse, or silently replace, paid work that no longer matches. A `source` of None isn't checked."""
    origin = job.origin_of(output)
    if origin is None:
        logger.warning(f"{output.name} has no record of what it was made from; using it as is.")
        return
    if source is not None and origin["source"] != source:
        raise BaberuError(
            f"{output.name} was made from different subtitles. "
            f"Use --redo {redo_name} to do it again, which costs money.")
    if model and origin.get("model") and origin["model"] != model:
        raise BaberuError(
            f"{output.name} was made with {origin['model']}, not {model}. "
            f"Use --redo {redo_name} to do it again, which costs money.")


def _guard_options(job: Job, output: Path, options: dict, redo_name: str) -> None:
    """Refuses to reuse a transcript made with different request options than the config now asks for."""
    origin = job.origin_of(output)
    recorded = (origin or {}).get("options")
    if recorded is None:
        return
    changed = sorted(key for key, value in options.items() if recorded.get(key) != value)
    if changed:
        raise BaberuError(
            f"{output.name} was made with different {', '.join(changed)}. "
            f"Use --redo {redo_name} to do it again, which costs money.")


def _extract(video_file: Path, job: Job, action: str) -> Path:
    existing = job.find_audio()
    if action == REUSE and existing:
        _announce_reuse("extract", existing)
        return existing

    codec_name = av_utils.get_audio_codec(video_file)
    if not codec_name:
        raise BaberuError(f"{video_file.name} has no audio to extract.")
    extension = CODEC_TO_EXTENSION_MAP.get(codec_name.lower(), codec_name)
    audio_file = av_utils.extract_audio(video_file, job.audio_path(extension))
    logger.info(f"extract: wrote {audio_file.name}")
    return audio_file


@dataclass
class _ListedTerms:
    terms: list[str]
    files: list[str]  # the names of the files they came from


def _listed_terms(request: RunRequest, settings: Settings, job: Job,
                  required: bool = True) -> _ListedTerms | None:
    """The terms in the channel's file and the --keyterms file. None when there are none to know of yet.

    When required, a channel with no file raises MissingChannelTerms (unless the config turns that off).
    Otherwise a missing file gives None.
    """
    info = keyterms.load_metadata(_source_info_path(job))
    channel_id = info.get("channel_id")
    listed = _ListedTerms([], [])
    if channel_id:
        try:
            path = keyterms.channel_file(info, settings.keyterms.channels, request.keyterms_dir)
        except keyterms.MissingChannelTerms as error:
            if settings.keyterms.require_channel_file:
                if required:
                    raise
                return None
            logger.warning(f"keyterms: {error}")
        else:
            listed.terms += keyterms.load_channel(path, channel_id).terms
            listed.files.append(path.name)
    if request.keyterms_file:
        listed.terms += keyterms.load_channel(request.keyterms_file).terms
        listed.files.append(request.keyterms_file.name)
    return listed if channel_id or request.keyterms_file else None


def _keyterms_are_stale(job: Job, listed: _ListedTerms) -> bool:
    """True when the listed terms changed since the job's keyterms were made from them."""
    origin = job.origin_of(_keyterms_path(job))
    return bool(origin) and origin["source"] != keyterms.digest(listed.terms)


def _keyterms(request: RunRequest, job: Job, action: str, settings: Settings) -> list[str]:
    """Builds, or reuses, the words sent to the transcription service."""
    options = settings.keyterms
    path = _keyterms_path(job)
    if action == REUSE:
        _announce_reuse("keyterms", path)
        return keyterms.load(path, options.max_terms)

    info = keyterms.load_metadata(_source_info_path(job))
    listed = _listed_terms(request, settings, job) or _ListedTerms([], [])
    llm = None
    if options.llm_model and info:
        llm = AIToolFactory.get_llm_provider(options.llm_model, effort=options.effort,
                                             max_retries=settings.translation.server_retries)
        logger.info(f"keyterms: asking {options.llm_model} for names in the video's details...")

    lang_name = translation_prompts.language_name(request.lang_from)
    terms = keyterms.build(info, listed.terms, options.max_terms, llm, lang_name)
    keyterms.write(terms, path)
    job.record_origin(path, keyterms.digest(listed.terms))
    job.save()
    used = ", ".join(listed.files) or "no keyterms file"
    logger.info(f"keyterms: {len(terms)} terms ({used}{', video details' if info else ''})")

    if terms and options.confirm and _pause_for_review(
            request, f"\n{', '.join(terms)}\n\n"
                     f"Review or edit {path} if needed, then press Enter to continue (Ctrl-C to cancel)..."):
        terms = keyterms.load(path, options.max_terms)
    return terms


def _pause_for_review(request: RunRequest, message: str) -> bool:
    """Waits for Enter while a generated file is reviewed. False when review is off or there's no keyboard."""
    if not request.review or not sys.stdin.isatty():
        return False
    try:
        input(message)
    except EOFError:  # raised for NUL on Windows and shells without a keyboard, where isatty() is also true
        logger.info("no keyboard; continuing without review")
        return False
    return True


def _with_keyterms(options: dict, terms: list[str] | None) -> dict:
    return {**options, "keyterms": terms} if terms else options


def _transcribe(audio_file: Path | None, job: Job, action: str, lang: str, settings: Settings,
                terms: list[str] | None = None):
    json_file = job.path(Stage.TRANSCRIPT)
    model = settings.transcription.model
    options = settings.transcription.options
    if action == REUSE:
        _guard_paid_work(job, json_file, None, model, "transcribe")
        _guard_options(job, json_file, options, "transcribe")
        _announce_reuse("transcribe", json_file)
        return transcript_conversion.load_transcript_json(json_file)

    if terms:
        options = _with_keyterms(options, terms)
        logger.info(f"transcribe: with {len(terms)} keyterms")
    logger.info(f"transcribe: sending audio to the transcription service ({model})...")
    provider = AIToolFactory.get_transcription_provider(model, options)
    transcript, json_data = transcript_chunked.transcribe_file(audio_file, provider, lang)
    transcript_conversion.write_transcript_json(json_data, json_file)
    job.record_origin(json_file, audio_file.name, model, provider.request_options())
    job.save()
    logger.info("transcribe: done")
    return transcript


def _convert(transcript, job: Job, action: str, settings: Settings) -> SSAFile:
    output = job.path(Stage.RAW)
    if action == REUSE:
        _announce_reuse("convert", output)
        return sub_utils.load(output)

    splits = job.load_splits()
    sub_data = transcript_conversion.convert_transcript_to_subs(transcript, settings.parsing, splits, settings.anomalies)
    job.save_splits(splits)
    sub_utils.write(sub_data, output)
    logger.info(f"convert: {len(sub_data.events)} lines")
    return sub_data


def _retranscribe_model(settings: Settings) -> str:
    return settings.transcription.retranscribe_model or settings.transcription.model


def _retranscribe_options(settings: Settings) -> dict:
    options = settings.transcription.retranscribe_options
    return settings.transcription.options if options is None else options


def _twopass(sub_data: SSAFile,
             audio_file: Path,
             job: Job,
             action: str,
             lang: str,
             line_range: LineRange | None,
             settings: Settings,
             transcript=None,
             terms: list[str] | None = None) -> tuple[SSAFile, LineRange | None]:
    """Re-transcribes poorly transcribed segments.

    Candidates come from the transcript's anomalies, when the subtitles were
    made from it in this run, and from runs of very short lines.
    """
    output = job.path(Stage.TWOPASS)
    source = sub_utils.fingerprint(sub_data)
    initial_lines = len(sub_data.events)
    model = _retranscribe_model(settings)

    if action == REUSE:
        _guard_paid_work(job, output, source, model, "retranscribe")
        _announce_reuse("retranscribe", output)
        sub_data = sub_utils.load(output)
        return sub_data, _shift(line_range, initial_lines, sub_data)

    segs = settings.mistimed_segs
    kinds = tuple(settings.transcription.retranscribe_kinds)
    anomalies = repair.unresolved(transcript, settings.anomalies) if transcript is not None else []
    if line_range:
        ranges = [line_range]
    else:
        by_lines = transcript_segmented.find_segments(sub_data, segs.mistimed_seg_thresh_sec, segs)
        by_anomaly = transcript_segmented.anomaly_ranges(sub_data, [a for a in anomalies if a.kind in kinds])
        ranges = transcript_segmented.merge_ranges(by_lines + by_anomaly)
        ranges = transcript_segmented.merge_ranges(transcript_segmented.pad_segments(sub_data, ranges))

    if ranges:
        logger.info(f"retranscribe: transcribing {len(ranges)} segment(s) again with {model}...")
        splits = job.load_splits()
        sub_data = transcript_segmented.transcribe_segments(
            subtitles=sub_data, ranges=ranges, audio_file=audio_file, job=job, lang=lang,
            parsing=settings.parsing, transcription_model=model,
            options=_with_keyterms(_retranscribe_options(settings), terms),
            split_cache=splits,
            limits=settings.anomalies, kinds=kinds, anomalies=anomalies,
            short_line_sec=segs.mistimed_seg_thresh_sec, always_accept=bool(line_range),
            margin_sec=segs.clip_margin_sec)
        job.save_splits(splits)
        sub_data = sub_correction.remove_empty(sub_data)
    else:
        logger.info("retranscribe: no badly timed segments found")

    sub_utils.write(sub_data, output)
    job.record_origin(output, source, model)
    job.save()
    return sub_data, _shift(line_range, initial_lines, sub_data)


def _reuse_free_output(stage: str, job: Job, output: Path, source: str, key: str) -> SSAFile | None:
    """A free stage's saved output, when it was made from this input with these settings."""
    if not _free_output_matches(job, output, source, key):
        return None
    logger.info(f"{stage}: unchanged, reusing {output.name}")
    return sub_utils.load(output)


def _fix(sub_data: SSAFile, job: Job, line_range: LineRange | None, settings: Settings) -> tuple[SSAFile, LineRange | None]:
    initial_lines = len(sub_data.events)
    output = job.path(Stage.FIXED)
    source, key = sub_utils.timed_fingerprint(sub_data), _fix_settings(settings, line_range)
    saved = _reuse_free_output("fix", job, output, source, key)
    if saved is not None:
        return saved, _shift(line_range, initial_lines, saved)

    if settings.mistimed_lines.method == "squeezed":
        sub_data = sub_correction.start_squeezed_lines_earlier(sub_data, settings.mistimed_lines, line_range)
    else:
        sub_data = sub_correction.fix_mistimed_lines(
            sub_data, settings.mistimed_lines.mistimed_line_thresh_sec, settings.mistimed_segs, line_range,
            settings.anomalies.max_letters_per_sec)
    sub_data = sub_correction.remove_empty(sub_data, line_range)
    sub_utils.write(sub_data, output)
    job.record_origin(output, source, options={"settings": key})
    job.save()
    logger.info("fix: done")
    return sub_data, _shift(line_range, initial_lines, sub_data)


def _context(sub_data: SSAFile, request: RunRequest, job: Job, action: str, options: TranslationSettings) -> str:
    """Generates or loads the background notes given to the translator."""
    context_file = job.path(Stage.CONTEXT)
    if request.context_file:
        context_data = sub_translation.load_context(request.context_file)
        if not _is_same_file(request.context_file, context_file):
            # Copies the notes into the job. The NekoCap title is translated with them.
            sub_translation.write_lines([context_data], context_file)
        return context_data

    if action == REUSE:
        _announce_reuse("context", context_file)
        return sub_translation.load_context(context_file)

    logger.info("context: generating background notes...")
    llm = AIToolFactory.get_llm_provider(
        model_name=options.websearch_model,
        system_prompt="You must follow prompt instructions.",
        max_retries=options.server_retries,
        effort=options.effort)
    context_data = sub_translation.generate_context(sub_data, llm, job.name, request.lang_from, request.lang_to,
                                                    _video_details(job))
    sub_translation.write_lines([context_data], context_file)

    if options.confirm_auto_context and _pause_for_review(
            request, f"\n{context_data}\n\nReview or edit {context_file} if needed, then press Enter to continue (Ctrl-C to cancel)..."):
        context_data = sub_translation.load_context(context_file)
    return context_data


def _video_details(job: Job) -> str:
    """The video's title and description, and its keyterms, for the notes model."""
    info = keyterms.load_metadata(_source_info_path(job))
    lines = []
    if info.get("title"):
        lines.append(f"Video title: {info['title']}")
    if info.get("channel"):
        lines.append(f"Channel: {info['channel']}")
    if info.get("description"):
        lines.append(f"Video description:\n{info['description']}")
    terms = keyterms.load(_keyterms_path(job), keyterms.MAX_KEYTERMS)
    if terms:
        lines.append(f"Known names and terms (most won't appear; use them only to identify names in "
                     f"the transcript, including misheard ones): {', '.join(terms)}")
    return "\n".join(lines)


def _frame_source(job: Job, plan: Plan, settings: Settings) -> frames.FrameSource | None:
    """Finds the video's scenes, or reuses them, and gives the frames for translation. None when ffmpeg can't read it."""
    options = settings.frames
    video = plan.frames_video
    if plan.action("frames") == REUSE:
        _announce_reuse("frames", job.scenes_path)
        cuts, duration = frames.load_cuts(job.scenes_path)
    else:
        logger.info(f"frames: finding the scenes in {video.name}...")
        duration = av_utils.duration_seconds(video)
        try:
            cuts = av_utils.scene_cuts(video, options.scene_threshold) if duration else None
        except ffmpeg.Error:
            cuts = None
        if cuts is None:
            logger.warning(f"frames: ffmpeg couldn't read {video.name}; translating without frames")
            return None
        shutil.rmtree(job.frames_dir, ignore_errors=True)
        frames.save_cuts(job.scenes_path, cuts, duration)
        job.record_origin(job.scenes_path, video.name, options={"threshold": options.scene_threshold})
        job.save()
    scenes = frames.scenes_from_cuts(cuts, duration, options.min_scene_sec, options.max_scene_sec)
    logger.info(f"frames: {len(scenes)} scenes")
    return frames.FrameSource(video, scenes, job.frames_dir, plan.frames_height, options.max_per_batch)


def _fallback_llm(system_prompt: str, model: str, options: TranslationSettings) -> LLMProvider | None:
    """The backup translator. None when there is none or its API key is missing."""
    if not options.fallback_model or options.fallback_model == model:
        return None
    try:
        return AIToolFactory.get_llm_provider(
            model_name=options.fallback_model, system_prompt=system_prompt, max_retries=options.server_retries,
            effort=options.effort)
    except ValueError as error:
        logger.warning(f"Fallback model {options.fallback_model} is unavailable: {error}")
        return None


def _translate(sub_data: SSAFile,
               request: RunRequest,
               job: Job,
               plan: Plan,
               line_range: LineRange | None,
               options: TranslationSettings,
               frame_source: frames.FrameSource | None = None) -> SSAFile:
    if plan.in_place:
        return _translate_in_place(sub_data, request, job, plan, line_range, options, frame_source)

    output = job.path(Stage.TRANSLATED)
    partial = job.path(Stage.PARTIAL)
    source = sub_utils.fingerprint(sub_data)

    if plan.action("translate") == REUSE:
        _guard_paid_work(job, output, source, request.model, "translate")
        _announce_reuse("translate", output)
        return sub_utils.load(output)

    if partial.exists():
        _guard_paid_work(job, partial, source, None, "translate")
        earlier_model = (job.origin_of(partial) or {}).get("model")
        if earlier_model and earlier_model != request.model:
            logger.warning(f"translate: resuming lines made with {earlier_model}; the rest will use {request.model}.")
    job.record_origin(partial, source, request.model)
    job.save()

    context = _context(sub_data, request, job, plan.action("context") or REUSE, options)

    system_prompt = translation_prompts.system_prompt(request.lang_from, request.lang_to, frame_source is not None)
    llm = AIToolFactory.get_llm_provider(
        model_name=request.model, system_prompt=system_prompt, max_retries=options.server_retries,
        effort=options.effort)
    start_index = line_range.start if line_range else 0

    translated_lines, report = sub_translation.translate(
        sub_file=sub_data,
        output_file=partial,
        context_prompt=context,
        llm=llm,
        fallback_llm=_fallback_llm(system_prompt, request.model, options),
        lang_from=request.lang_from,
        lang_to=request.lang_to,
        options=options,
        line_range=line_range,
        report_file=_translation_report_path(job),
        frames=frame_source)
    sub_data = sub_utils.replace_lines(translated_lines[start_index:], sub_data, start_index)
    sub_data = sub_utils.md_to_ass(sub_data, line_range)

    sub_utils.write(sub_data, output)
    job.record_origin(output, source, request.model)
    job.clear(partial)
    job.save()
    _log_translation(report)
    return sub_data


def _in_place_paths(job: Job, line_range: LineRange) -> tuple[Path, Path]:
    """Progress and report for translating some lines again, kept apart from the whole file's."""
    folder = job.path(Stage.TRANSLATED).parent
    return folder / f"partial.{line_range.label}.txt", folder / f"translation.{line_range.label}.json"


def _translate_in_place(sub_data: SSAFile, request: RunRequest, job: Job, plan: Plan, line_range: LineRange,
                        options: TranslationSettings, frame_source: frames.FrameSource | None = None) -> SSAFile:
    """Translates only these lines again, and puts them into the job's whole translation."""
    output = job.path(Stage.TRANSLATED)
    source = sub_utils.fingerprint(sub_data)
    whole = sub_utils.load(output)
    origin = job.origin_of(output)
    if (origin and origin.get("source") != source) or len(whole.events) != len(sub_data.events):
        raise BaberuError(f"{output.name} was made from different subtitles, so its lines can't be replaced one by one. "
                          f"Use --redo translate without --lines to translate everything again, which costs money.")
    if line_range.end >= len(whole.events):
        raise BaberuError(f"--lines goes past the last line ({len(whole.events)}).")

    partial, report_file = _in_place_paths(job, line_range)
    context = _context(sub_data, request, job, plan.action("context") or REUSE, options)
    system_prompt = translation_prompts.system_prompt(request.lang_from, request.lang_to, frame_source is not None)
    llm = AIToolFactory.get_llm_provider(
        model_name=request.model, system_prompt=system_prompt, max_retries=options.server_retries,
        effort=options.effort)
    translated_lines, report = sub_translation.translate(
        sub_file=sub_data, output_file=partial, context_prompt=context, llm=llm,
        fallback_llm=_fallback_llm(system_prompt, request.model, options),
        lang_from=request.lang_from, lang_to=request.lang_to, options=options,
        line_range=line_range, report_file=report_file, frames=frame_source)

    new_lines = translated_lines[line_range.start:line_range.end + 1]
    whole = sub_utils.replace_lines(new_lines, whole, line_range.start)
    whole = sub_utils.md_to_ass(whole, line_range)
    sub_utils.write(whole, output)
    job.record_origin(output, source, (origin or {}).get("model") or request.model)
    job.clear(partial)
    job.save()
    logger.info(f"translate: replaced lines {line_range.label[1:]} in {output.name}")
    _log_translation(report)
    return whole


def _translation_report_path(job: Job) -> Path:
    """translation.json sits beside the translation, one per language and --lines range."""
    return job.path(Stage.PARTIAL).parent / "translation.json"


def _log_translation(report: dict) -> None:
    unresolved = sub_translation.unresolved_lines(report)
    if not unresolved:
        logger.info("translate: done")
        return
    shown = ", ".join(str(line) for line in list(unresolved)[:20]) + (", ..." if len(unresolved) > 20 else "")
    logger.warning(f"translate: {len(unresolved)} lines unresolved ({shown}); see translation.json, fix with --lines")


def _pad(sub_data: SSAFile, job: Job, line_range: LineRange | None, settings: Settings) -> SSAFile:
    output = job.path(Stage.PADDED)
    source, key = sub_utils.timed_fingerprint(sub_data), _pad_settings(settings, line_range)
    saved = _reuse_free_output("pad", job, output, source, key)
    if saved is not None:
        return saved

    sub_data = sub_correction.apply_timing_standards(sub_data, settings.subtitle_padding, line_range)
    sub_utils.write(sub_data, output)
    job.record_origin(output, source, options={"settings": key})
    job.save()
    logger.info("pad: done")
    return sub_data


# --- Handing results back ---

def _publish(job: Job, source: Path, target: Path, overwrite: bool) -> None:
    """Copies a finished file into place without clobbering anyone else's work."""
    new_hash = file_hash(source)

    if target.exists():
        current_hash = file_hash(target)
        edited = current_hash != job.published_hash(target.name)
        if edited and new_hash == job.published_hash(target.name) and not overwrite:
            # Keeps the user's edits when there's nothing new.
            logger.info(f"Keeping your edits to {target.name}; baberu has nothing new for it.")
            return
        if current_hash != new_hash and edited and not overwrite:
            raise BaberuError(
                f"{target} already exists and was changed outside baberu. "
                f"Use --overwrite to replace it. The new version is in {job.folder}.")
        if current_hash == new_hash:
            job.record_published(target.name, new_hash)
            job.save()
            return

    with atomic_path(target) as temporary:
        shutil.copyfile(source, temporary)
    job.record_published(target.name, new_hash)
    job.save()


def _publish_subtitles(sub_data: SSAFile, job: Job, target: Path, overwrite: bool) -> Path:
    rendered = job.folder / f"result{target.suffix}"
    sub_utils.write(sub_data, rendered)
    try:
        _publish(job, rendered, target, overwrite)
    finally:
        rendered.unlink(missing_ok=True)
    return target


def _deliver_stage_file(request: RunRequest, job: Job, produced: Path) -> Path:
    """Copies the file made by a run that stops before subtitles to -o or the output folder."""
    if request.output_file:
        if request.output_file.suffix.lower() != produced.suffix.lower():
            raise BaberuError(f"The result is a {produced.suffix} file, so -o needs that extension: {request.output_file}")
        target = request.output_file
    elif request.output_folder:
        target = request.output_folder / f"{job.name}{produced.suffix}"
    else:
        return produced
    _publish(job, produced, target, request.overwrite)
    return target


# --- Health ---

def _write_health(transcript, job: Job, settings: Settings) -> dict:
    """Saves the transcript's health, with the repairs made so far, beside the rest of the job."""
    report = health.report(transcript, settings.anomalies)
    atomic_write_text(job.folder / "health.json", json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _log_health(report: dict | None) -> None:
    if report is not None:
        counts = report["anomaly_counts"]
        logger.info("health: " + (", ".join(f"{kind} {count}" for kind, count in counts.items()) or "no anomalies"))
    if health.repairs:
        logger.info("repairs: " + ", ".join(f"{name} {count}" for name, count in health.repairs.items()))
    if llm_base.ledger:
        logger.info("spend: " + llm_base.usage_summary())


def diagnose(request: RunRequest, settings: Settings) -> Path:
    """Reports a transcript's health without changing anything else. Costs nothing."""
    if request.url:
        raise BaberuError("--diagnose needs a transcript or a file with a work folder, not a URL.")
    job = Job.locate(request.input_file, request.output_dir, request.lang_from, request.lang_to,
                     legacy_name=legacy.job_name(request.input_file))
    transcript_file = request.json_file or job.path(Stage.TRANSCRIPT)
    if not transcript_file.exists():
        raise BaberuError(f"--diagnose needs a transcript, and none was found for {request.input_file.name}.")

    health.repairs.clear()
    transcript = transcript_conversion.load_transcript_json(transcript_file)
    report = _write_health(transcript, job, settings)
    print(f"Health of {transcript_file}:", file=sys.stderr)
    print(health.summary(report), file=sys.stderr)
    result = job.folder / "health.json"
    print(result)
    return result


# --- Running ---

def _is_media_source(input_file: Path) -> bool:
    return kind_of(input_file) in (FileKind.VIDEO, FileKind.AUDIO) and not enclosing_job(input_file)


def run(request: RunRequest, settings: Settings) -> None:
    """Runs one job from start to finish."""
    input_file, video, audio = request.input_file, request.video_file, request.audio_file
    info: dict = {}
    if request.url and request.dry_run:
        # Plans from an earlier download and its work folder, without downloading again.
        input_file = av_utils.find_download(request.url, request.output_dir)
        if not input_file:
            _preview_url(request.url)
            return
        video, audio = (None, input_file) if kind_of(input_file) is FileKind.AUDIO else (input_file, None)
    elif request.url:
        logger.info("download: fetching...")
        max_height = settings.frames.height if settings.frames.enabled and request.translate else None
        input_file, info = av_utils.download(request.url, request.output_dir, max_height)
        logger.info(f"download: {input_file.name}")
        if kind_of(input_file) is FileKind.AUDIO:
            video, audio = None, input_file
        else:
            video = input_file

    job = Job.locate(input_file, request.output_dir, request.lang_from, request.lang_to, request.line_range,
                     request.translate, legacy.job_name(input_file))
    legacy.adopt(job, [input_file.parent, request.output_dir], request.dry_run)

    if _is_media_source(input_file):
        job.check_source(input_file.name)

    # --lines with -t on a job that already has a whole translation corrects those lines in it.
    in_place = bool(request.line_range and request.translate and not request.retranscribe_required
                    and not request.fix_named and job.has(job.path(Stage.TRANSLATED, whole_file=True)))
    if in_place:
        job.line_range = None

    plan = _build_plan(request, settings, job, input_file, video, audio, in_place)
    _preflight(request, settings, job, plan)

    if request.dry_run:
        _print_plan(plan, job)
        return

    _clear_redone_work(request, job, plan)
    health.repairs.clear()
    llm_base.ledger.clear()
    if _is_media_source(input_file):
        job.set_source(input_file.name)
    _save_source_info(job, input_file, info)
    job.save()

    results = _run_stages(request, settings, job, plan)

    logger.info("Done.")
    for result in results:
        print(result)

    if request.clean:
        _clean(job, input_file, results)


def _save_source_info(job: Job, input_file: Path, info: dict) -> None:
    """Saves the video's title, tags and description. Keyterms and translation notes read them."""
    if not info and not _source_info_path(job).exists():
        sidecar = keyterms.find_info_json(input_file)
        info = keyterms.load_metadata(sidecar) if sidecar else {}
    if info:
        keyterms.save_metadata(info, _source_info_path(job))


def _run_stages(request: RunRequest, settings: Settings, job: Job, plan: Plan) -> list[Path]:
    video, audio = plan.video, plan.audio
    # In place, only the translation works on the range; everything else is the whole file.
    line_range = None if plan.in_place else request.line_range
    transcript = None
    sub_data: SSAFile | None = None
    results: list[Path] = []

    if plan.action("extract"):
        audio = _extract(video, job, plan.action("extract"))

    terms: list[str] = []
    if plan.action("keyterms"):
        terms = _keyterms(request, job, plan.action("keyterms"), settings)
    users = _keyterm_users(settings)

    if plan.action("transcribe"):
        transcript = _transcribe(audio, job, plan.action("transcribe"), request.lang_from, settings,
                                 terms if "transcribe" in users else None)
    if request.json_file:
        transcript = transcript_conversion.load_transcript_json(request.json_file)

    if plan.action("convert"):
        sub_data = _convert(transcript, job, plan.action("convert"), settings)
    if request.sub_file:
        sub_data = sub_utils.load(request.sub_file)

    if plan.action("retranscribe"):
        sub_data, line_range = _twopass(sub_data, audio, job, plan.action("retranscribe"), request.lang_from, line_range,
                                        settings, transcript if not request.sub_file else None,
                                        terms if "retranscribe" in users else None)
    if plan.action("fix"):
        sub_data, line_range = _fix(sub_data, job, line_range, settings)
    frame_source = _frame_source(job, plan, settings) if plan.action("frames") else None
    if plan.frames_off and plan.action("translate") == RUN:
        logger.warning(f"translate: {plan.frames_off}")
    if plan.action("translate"):
        sub_data = _translate(sub_data, request, job, plan, request.line_range if plan.in_place else line_range,
                              settings.translation, frame_source)
    if plan.action("pad"):
        sub_data = _pad(sub_data, job, line_range, settings)

    _log_health(_write_health(transcript, job, settings) if transcript is not None else None)

    if plan.subtitle_target:
        results.append(_publish_subtitles(sub_data, job, plan.subtitle_target, request.overwrite))
    elif plan.action("transcribe") and not (plan.action("hardcode") or plan.action("audio_to_video")):
        results.append(_deliver_stage_file(request, job, job.path(Stage.TRANSCRIPT)))
    elif plan.action("extract") and not (plan.action("hardcode") or plan.action("audio_to_video")):
        results.append(_deliver_stage_file(request, job, audio))

    if plan.action("nekocap"):
        publish.upload(publish.Upload(job, plan.subtitle_target, plan.input_file, request.lang_from, request.lang_to,
                                      request.nekocap_privacy or settings.nekocap.privacy), settings)

    if plan.action("hardcode"):
        av_utils.hardcode_subtitles(plan.hardcode_video, plan.hardcode_subs, plan.video_target)
        results.append(plan.video_target)
    if plan.action("audio_to_video"):
        av_utils.audio_to_video(request.image_file, audio or job.find_audio(), plan.video_target)
        results.append(plan.video_target)

    return results


def _clean(job: Job, input_file: Path, results: list[Path]) -> None:
    if enclosing_job(input_file) == job.folder.absolute():
        logger.warning("Keeping the work folder because the input is inside it.")
        return
    if any(enclosing_job(result) == job.folder.absolute() for result in results):
        logger.warning("Keeping the work folder because the result is inside it.")
        return
    shutil.rmtree(job.folder, ignore_errors=True)
