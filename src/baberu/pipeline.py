"""Works out what a run will do, checks it can succeed, then does it."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
import logging
import shutil
import sys

from baberu.providers.factory import AIToolFactory
from baberu.providers.llm.base import LLMProvider
from pysubs2 import SSAFile

from baberu import legacy
from baberu.constants import CODEC_TO_EXTENSION_MAP
from baberu.errors import BaberuError
from baberu.job import Job, Stage, enclosing_job
from baberu.settings import Settings, TranslationSettings
from baberu.subtitling import sub_correction, sub_translation, sub_utils, translation_prompts
from baberu.subtitling.types import LineRange
from baberu.tools import av_utils
from baberu.tools.file_utils import FileKind, atomic_path, file_hash, kind_of
from baberu.transcription import transcript_chunked, transcript_conversion, transcript_segmented

if TYPE_CHECKING:
    from baberu.cli import RunRequest

logger: logging.Logger = logging.getLogger(__name__)

RUN, REUSE = "run", "reuse"
_PAID = {"transcribe", "retranscribe", "context", "translate"}
_NEEDS_FFMPEG = {"extract", "transcribe", "retranscribe", "hardcode", "audio_to_video"}
_SUBTITLE_STAGE_NAMES = (("pad", "padded"), ("translate", "translated"), ("fix", "fixed"),
                         ("retranscribe", "2pass"), ("convert", "raw"))


@dataclass
class PlannedStage:
    name: str
    action: str
    detail: str = ""

    @property
    def spends_money(self) -> bool:
        return self.action == RUN and self.name in _PAID


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


def _build_plan(request: RunRequest, job: Job, input_file: Path, video: Path | None, audio: Path | None) -> Plan:
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
        action = RUN if request.json_file else _reuse_or_run(request, job, "convert", job.path(Stage.RAW))
        stages.append(PlannedStage("convert", action))

    has_subtitles = bool(request.sub_file) or any(s.name == "convert" for s in stages)
    if has_subtitles and retranscribe and will_have_audio:
        stages.append(PlannedStage("retranscribe", _reuse_or_run(request, job, "retranscribe", job.path(Stage.TWOPASS))))
    if has_subtitles and request.fix:
        stages.append(PlannedStage("fix", RUN))
    if has_subtitles and request.translate:
        translate = _reuse_or_run(request, job, "translate", job.path(Stage.TRANSLATED))
        if translate == RUN and not request.context_file:
            stages.append(PlannedStage("context", _reuse_or_run(request, job, "context", job.path(Stage.CONTEXT))))
        stages.append(PlannedStage("translate", translate, _line_count_detail(request, job)))
    if has_subtitles and request.pad:
        stages.append(PlannedStage("pad", RUN))

    plan = Plan(stages, input_file, video, audio)
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
    return plan


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

    for target in (plan.subtitle_target, plan.video_target):
        if _is_same_file(target, plan.input_file) or _is_same_file(target, plan.hardcode_video):
            raise BaberuError(f"The output would replace its own input: {target}")

    if plan.video_target and plan.video_target.exists() and not request.overwrite:
        raise BaberuError(f"{plan.video_target} already exists. Use --overwrite to replace it.")

    target = plan.subtitle_target
    if target and target.exists() and not request.overwrite and plan.spends_money \
            and file_hash(target) != job.published_hash(target.name):
        raise BaberuError(f"{target} already exists and wasn't written by baberu. Use --overwrite to replace it.")


def _check_api_keys(request: RunRequest, settings: Settings, plan: Plan) -> None:
    """Building a provider needs its key but makes no request."""
    try:
        if plan.action("transcribe") == RUN or plan.action("retranscribe") == RUN:
            AIToolFactory.get_transcription_provider(settings.transcription.model)
        if plan.action("translate") == RUN:
            AIToolFactory.get_llm_provider(request.model)
        if plan.action("context") == RUN:
            AIToolFactory.get_llm_provider(settings.translation.websearch_model)
    except ValueError as error:
        raise BaberuError(str(error)) from error


def _print_plan(plan: Plan) -> None:
    print(f"Plan for {plan.input_file.name}:", file=sys.stderr)
    for stage in plan.stages:
        notes = ", ".join(filter(None, ["paid" if stage.spends_money else "", stage.detail]))
        print(f"  {stage.name:<15}{stage.action:<7}{notes}", file=sys.stderr)
    for target in (plan.subtitle_target, plan.video_target):
        if target:
            print(f"Result: {target}", file=sys.stderr)


def _preview_url(url: str) -> None:
    info = av_utils.probe_url(url)
    minutes = round((info.get("duration") or 0) / 60)
    print(f"Would download: {info.get('title', url)} ({minutes} min)", file=sys.stderr)
    print("The rest of the plan is decided once the file has been downloaded.", file=sys.stderr)


# --- Redo ---

def _clear_redone_work(request: RunRequest, job: Job) -> None:
    if _wants_redo(request, "extract"):
        audio = job.find_audio()
        if audio:
            audio.unlink()
    if _wants_redo(request, "transcribe"):
        job.clear(job.path(Stage.TRANSCRIPT))
    if _wants_redo(request, "convert"):
        job.clear(job.path(Stage.RAW))
    if _wants_redo(request, "retranscribe"):
        job.clear(job.path(Stage.TWOPASS))
        job.clear_segments()
    if _wants_redo(request, "context"):
        job.clear(job.path(Stage.CONTEXT))
    if _wants_redo(request, "translate"):
        job.clear(job.path(Stage.TRANSLATED), job.path(Stage.PARTIAL))
    if request.redo:
        job.save()


# --- Stages ---

def _announce_reuse(stage: str, path: Path) -> None:
    logger.info(f"{stage}: reusing {path.name} (--redo {stage} to do it again)")


def _shift(line_range: LineRange | None, before: int, sub_data: SSAFile) -> LineRange | None:
    """Keeps a line range on the same lines after a stage changes the line count."""
    return line_range.resized(len(sub_data.events) - before) if line_range else None


def _guard_paid_work(job: Job, output: Path, source: str, model: str | None, redo_name: str) -> None:
    """Refuses to reuse, or silently replace, paid work that no longer matches."""
    origin = job.origin_of(output)
    if origin is None:
        logger.warning(f"{output.name} has no record of what it was made from; using it as is.")
        return
    if origin["source"] != source:
        raise BaberuError(
            f"{output.name} was made from different subtitles. "
            f"Use --redo {redo_name} to do it again, which costs money.")
    if model and origin.get("model") and origin["model"] != model:
        raise BaberuError(
            f"{output.name} was made with {origin['model']}, not {model}. "
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


def _transcribe(audio_file: Path | None, job: Job, action: str, lang: str, settings: Settings):
    json_file = job.path(Stage.TRANSCRIPT)
    if action == REUSE:
        _announce_reuse("transcribe", json_file)
        return transcript_conversion.load_transcript_json(json_file)

    logger.info("transcribe: sending audio to the transcription service...")
    provider = AIToolFactory.get_transcription_provider(settings.transcription.model)
    transcript, json_data = transcript_chunked.transcribe_file(audio_file, provider, lang)
    transcript_conversion.write_transcript_json(json_data, json_file)
    logger.info("transcribe: done")
    return transcript


def _convert(transcript, job: Job, action: str, settings: Settings) -> SSAFile:
    output = job.path(Stage.RAW)
    if action == REUSE:
        _announce_reuse("convert", output)
        return sub_utils.load(output)

    sub_data = transcript_conversion.convert_transcript_to_subs(transcript, settings.parsing)
    sub_utils.write(sub_data, output)
    logger.info(f"convert: {len(sub_data.events)} lines")
    return sub_data


def _twopass(sub_data: SSAFile,
             audio_file: Path,
             job: Job,
             action: str,
             lang: str,
             line_range: LineRange | None,
             settings: Settings) -> tuple[SSAFile, LineRange | None]:
    """Re-transcribes poorly transcribed segments."""
    output = job.path(Stage.TWOPASS)
    source = sub_utils.fingerprint(sub_data)
    initial_lines = len(sub_data.events)

    if action == REUSE:
        _guard_paid_work(job, output, source, settings.transcription.model, "retranscribe")
        _announce_reuse("retranscribe", output)
        sub_data = sub_utils.load(output)
        return sub_data, _shift(line_range, initial_lines, sub_data)

    segs = settings.mistimed_segs
    if line_range:
        ranges = [line_range]
    else:
        ranges = transcript_segmented.find_segments(sub_data, segs.mistimed_seg_thresh_sec, segs)
        ranges = transcript_segmented.pad_segments(sub_data, ranges) if ranges else []

    if ranges:
        logger.info(f"retranscribe: transcribing {len(ranges)} segment(s) again...")
        sub_data = transcript_segmented.transcribe_segments(
            subtitles=sub_data, ranges=ranges, audio_file=audio_file, job=job, lang=lang,
            parsing=settings.parsing, transcription_model=settings.transcription.model)
        sub_data = sub_correction.remove_empty(sub_data)
    else:
        logger.info("retranscribe: no badly timed segments found")

    sub_utils.write(sub_data, output)
    job.record_origin(output, source, settings.transcription.model)
    job.save()
    return sub_data, _shift(line_range, initial_lines, sub_data)


def _fix(sub_data: SSAFile, job: Job, line_range: LineRange | None, settings: Settings) -> tuple[SSAFile, LineRange | None]:
    initial_lines = len(sub_data.events)
    sub_data = sub_correction.fix_mistimed_lines(
        sub_data, settings.mistimed_lines.mistimed_line_thresh_sec, settings.mistimed_segs, line_range)
    sub_data = sub_correction.remove_empty(sub_data, line_range)
    sub_utils.write(sub_data, job.path(Stage.FIXED))
    logger.info("fix: done")
    return sub_data, _shift(line_range, initial_lines, sub_data)


def _context(sub_data: SSAFile, request: RunRequest, job: Job, action: str, options: TranslationSettings) -> str:
    """Generates or loads the background notes given to the translator."""
    if request.context_file:
        return sub_translation.load_context(request.context_file)

    context_file = job.path(Stage.CONTEXT)
    if action == REUSE:
        _announce_reuse("context", context_file)
        return sub_translation.load_context(context_file)

    logger.info("context: generating background notes...")
    llm = AIToolFactory.get_llm_provider(
        model_name=options.websearch_model,
        system_prompt="You must follow prompt instructions.",
        max_retries=options.server_retries)
    context_data = sub_translation.generate_context(sub_data, llm, job.name, request.lang_from, request.lang_to)
    sub_translation.write_lines([context_data], context_file)

    if options.confirm_auto_context and sys.stdin.isatty():
        input(f"\n{context_data}\n\nReview or edit {context_file} if needed, then press Enter to continue (Ctrl-C to cancel)...")
        context_data = sub_translation.load_context(context_file)
    return context_data


def _fallback_llm(system_prompt: str, model: str, options: TranslationSettings) -> LLMProvider | None:
    """The backup translator. A missing API key shouldn't stop the main model from running."""
    if not options.fallback_model or options.fallback_model == model:
        return None
    try:
        return AIToolFactory.get_llm_provider(
            model_name=options.fallback_model, system_prompt=system_prompt, max_retries=options.server_retries)
    except ValueError as error:
        logger.warning(f"Fallback model {options.fallback_model} is unavailable: {error}")
        return None


def _translate(sub_data: SSAFile,
               request: RunRequest,
               job: Job,
               plan: Plan,
               line_range: LineRange | None,
               options: TranslationSettings) -> SSAFile:
    output = job.path(Stage.TRANSLATED)
    partial = job.path(Stage.PARTIAL)
    source = sub_utils.fingerprint(sub_data)

    if plan.action("translate") == REUSE:
        _guard_paid_work(job, output, source, request.model, "translate")
        _announce_reuse("translate", output)
        return sub_utils.load(output)

    if partial.exists():
        _guard_paid_work(job, partial, source, None, "translate")
    job.record_origin(partial, source, request.model)
    job.save()

    context = _context(sub_data, request, job, plan.action("context") or REUSE, options)

    system_prompt = translation_prompts.system_prompt(request.lang_from, request.lang_to)
    llm = AIToolFactory.get_llm_provider(
        model_name=request.model, system_prompt=system_prompt, max_retries=options.server_retries)
    start_index = line_range.start if line_range else 0

    translated_lines = sub_translation.translate(
        sub_file=sub_data,
        output_file=partial,
        context_prompt=context,
        llm=llm,
        fallback_llm=_fallback_llm(system_prompt, request.model, options),
        lang_from=request.lang_from,
        lang_to=request.lang_to,
        options=options,
        line_range=line_range)
    sub_data = sub_utils.replace_lines(translated_lines[start_index:], sub_data, start_index)
    sub_data = sub_utils.md_to_ass(sub_data)

    sub_utils.write(sub_data, output)
    job.record_origin(output, source, request.model)
    job.clear(partial)
    job.save()
    logger.info("translate: done")
    return sub_data


def _pad(sub_data: SSAFile, job: Job, line_range: LineRange | None, settings: Settings) -> SSAFile:
    sub_data = sub_correction.apply_timing_standards(sub_data, settings.subtitle_padding, line_range)
    sub_utils.write(sub_data, job.path(Stage.PADDED))
    logger.info("pad: done")
    return sub_data


# --- Handing results back ---

def _publish(job: Job, source: Path, target: Path, overwrite: bool) -> None:
    """Copies a finished file into place without clobbering anyone else's work."""
    new_hash = file_hash(source)

    if target.exists():
        current_hash = file_hash(target)
        if current_hash != new_hash and current_hash != job.published_hash(target.name) and not overwrite:
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
    """For runs that stop before subtitles, -o can copy the file out of the work folder."""
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


# --- Running ---

def _is_media_source(input_file: Path) -> bool:
    return kind_of(input_file) in (FileKind.VIDEO, FileKind.AUDIO) and not enclosing_job(input_file)


def run(request: RunRequest, settings: Settings) -> None:
    """Runs one job from start to finish."""
    if request.url and request.dry_run:
        _preview_url(request.url)
        return

    input_file, video, audio = request.input_file, request.video_file, request.audio_file
    if request.url:
        logger.info("download: fetching...")
        input_file = av_utils.download(request.url, request.output_dir)
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

    plan = _build_plan(request, job, input_file, video, audio)
    _preflight(request, settings, job, plan)

    if request.dry_run:
        _print_plan(plan)
        return

    _clear_redone_work(request, job)
    if _is_media_source(input_file):
        job.set_source(input_file.name)
    job.save()

    results = _run_stages(request, settings, job, plan)

    logger.info("Done.")
    for result in results:
        print(result)

    if request.clean:
        _clean(job, input_file, results)


def _run_stages(request: RunRequest, settings: Settings, job: Job, plan: Plan) -> list[Path]:
    video, audio = plan.video, plan.audio
    line_range = request.line_range
    transcript = None
    sub_data: SSAFile | None = None
    results: list[Path] = []

    if plan.action("extract"):
        audio = _extract(video, job, plan.action("extract"))

    if plan.action("transcribe"):
        transcript = _transcribe(audio, job, plan.action("transcribe"), request.lang_from, settings)
    if request.json_file:
        transcript = transcript_conversion.load_transcript_json(request.json_file)

    if plan.action("convert"):
        sub_data = _convert(transcript, job, plan.action("convert"), settings)
    if request.sub_file:
        sub_data = sub_utils.load(request.sub_file)

    if plan.action("retranscribe"):
        sub_data, line_range = _twopass(sub_data, audio, job, plan.action("retranscribe"), request.lang_from, line_range, settings)
    if plan.action("fix"):
        sub_data, line_range = _fix(sub_data, job, line_range, settings)
    if plan.action("translate"):
        sub_data = _translate(sub_data, request, job, plan, line_range, settings.translation)
    if plan.action("pad"):
        sub_data = _pad(sub_data, job, line_range, settings)

    if plan.subtitle_target:
        results.append(_publish_subtitles(sub_data, job, plan.subtitle_target, request.overwrite))
    elif plan.action("transcribe") and not (plan.action("hardcode") or plan.action("audio_to_video")):
        results.append(_deliver_stage_file(request, job, job.path(Stage.TRANSCRIPT)))
    elif plan.action("extract") and not (plan.action("hardcode") or plan.action("audio_to_video")):
        results.append(_deliver_stage_file(request, job, audio))

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
