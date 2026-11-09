"""Runs the requested stages, in order, skipping any whose output already exists."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import logging

from LLMFactory.factory import AIToolFactory
from LLMFactory.llm.base import LLMProvider
from LLMFactory.transcription.base import TranscriptionResult
from pysubs2 import SSAFile

from baberu.artifacts import Artifacts
from baberu.constants import CODEC_TO_EXTENSION_MAP
from baberu.settings import (
    MistimedLineSettings,
    MistimedSegmentSettings,
    PaddingSettings,
    ParsingSettings,
    Settings,
    TranscriptionSettings,
    TranslationSettings,
)
from baberu.subtitling import sub_correction, sub_translation, sub_utils, translation_prompts
from baberu.subtitling.types import LineRange
from baberu.tools import av_utils, file_utils
from baberu.tools.file_utils import formats
from baberu.transcription import transcript_chunked, transcript_conversion, transcript_segmented

if TYPE_CHECKING:
    from baberu.cli import RunRequest

logger: logging.Logger = logging.getLogger(__name__)


def _shift(line_range: LineRange | None, before: int, sub_data: SSAFile) -> LineRange | None:
    """Keeps a line range on the same lines after a stage changes the line count."""
    return line_range.resized(len(sub_data.events) - before) if line_range else None


def _download(url: str, output_file: Path | None, output_dir: Path) -> Path:
    if output_file and output_file.exists():
        logger.info(f"Download skipped. File already exists: {output_file}")
        return output_file

    logger.info(f"Downloading from {url}...")
    downloaded_file = av_utils.download(url, output_filename=output_file, download_directory=output_dir)
    logger.info(f"Downloaded to: {downloaded_file}")
    return downloaded_file


def _extract(video_file: Path, artifacts: Artifacts, output_file: Path | None) -> Path:
    """Extracts the audio stream from a video file."""
    codec_name = av_utils.get_audio_codec(video_file)
    if not codec_name:
        logger.error(f"No audio codec found for direct copy extraction of {video_file}")
        raise ValueError("No audio codec found for direct copy extraction.")

    extension = CODEC_TO_EXTENSION_MAP.get(codec_name.lower())
    if extension is None:
        logger.warning(f"No conventional extension found for codec '{codec_name}'. Using codec name as extension.")
        extension = codec_name

    output_audio_file = output_file or artifacts.audio(extension)
    if output_audio_file.exists():
        logger.info(f"Extraction skipped. File already exists: {output_audio_file}")
        return output_audio_file

    logger.info(f"Extracting audio from: {video_file} to {output_audio_file}")
    audio_file = av_utils.extract_audio(video_file, output_audio_file)
    logger.info(f"Audio extracted: {audio_file}")
    return audio_file


def _transcribe(audio_file: Path,
                artifacts: Artifacts,
                output_file: Path | None,
                lang: str,
                options: TranscriptionSettings) -> TranscriptionResult:
    """Transcribes an audio file, or reads the saved transcript."""
    json_file = output_file or artifacts.transcript()

    if json_file.exists():
        logger.info(f"Transcription skipped. File already exists: {json_file}")
        return transcript_conversion.load_transcript_json(json_file)

    logger.info(f"Transcribing audio from: {audio_file} to {json_file}")
    provider = AIToolFactory.get_transcription_provider(options.model)
    transcript, json_data = transcript_chunked.transcribe_file(audio_file, provider, lang)
    transcript_conversion.write_transcript_json(json_data, json_file)
    logger.info(f"Audio transcribed: {json_file}")
    return transcript


def _convert(transcript: TranscriptionResult, artifacts: Artifacts, parsing: ParsingSettings) -> SSAFile:
    """Converts a transcript to subtitles."""
    output_sub_file = artifacts.raw()

    if output_sub_file.exists():
        logger.info(f"Conversion skipped. File already exists: {output_sub_file}")
        return sub_utils.load(output_sub_file)

    logger.info(f"Converting transcription JSON to subtitles: {output_sub_file}")
    sub_data = transcript_conversion.convert_transcript_to_subs(transcript, parsing)
    sub_utils.write(sub_data, output_sub_file)
    logger.info(f"Transcription converted: {output_sub_file}")
    return sub_data


def _twopass(sub_data: SSAFile,
             audio_file: Path,
             artifacts: Artifacts,
             lang: str,
             line_range: LineRange | None,
             settings: Settings) -> tuple[SSAFile, LineRange | None, Path]:
    """Re-transcribes poorly transcribed segments."""
    segs: MistimedSegmentSettings = settings.mistimed_segs
    initial_lines = len(sub_data.events)
    output_sub_file = artifacts.twopass()

    if output_sub_file.exists():
        logger.info(f"Retranscription skipped. File already exists: {output_sub_file}")
        sub_data = sub_utils.load(output_sub_file)
        return sub_data, _shift(line_range, initial_lines, sub_data), output_sub_file

    if line_range:
        ranges = [line_range]
        logger.info(f"Retranscribing lines {line_range.label} for two-pass process...")
    else:
        ranges = transcript_segmented.find_segments(sub_data, segs.mistimed_seg_thresh_sec, segs)
        if not ranges:
            logger.info("Retranscription skipped. No segments identified for retranscription.")
            sub_utils.write(sub_data, output_sub_file)
            return sub_data, line_range, output_sub_file
        logger.info(f"Retranscribing {len(ranges)} segment(s) for two-pass process...")
        ranges = transcript_segmented.pad_segments(sub_data, ranges)

    sub_data = transcript_segmented.transcribe_segments(
        subtitles=sub_data,
        ranges=ranges,
        audio_file=audio_file,
        artifacts=artifacts,
        lang=lang,
        parsing=settings.parsing,
        transcription_model=settings.transcription.model)
    sub_data = sub_correction.remove_empty(sub_data)
    line_range = _shift(line_range, initial_lines, sub_data)

    sub_utils.write(sub_data, output_sub_file)
    logger.info(f"Retranscription processed: {output_sub_file}")
    return sub_data, line_range, output_sub_file


def _fix(sub_data: SSAFile,
         artifacts: Artifacts,
         line_range: LineRange | None,
         lines: MistimedLineSettings,
         segs: MistimedSegmentSettings) -> tuple[SSAFile, LineRange | None, Path]:
    """Fixes mistimed subtitle lines by merging and adjusting them."""
    initial_lines = len(sub_data.events)
    output_sub_file = artifacts.fixed()

    if output_sub_file.exists():
        logger.info(f"Subtitle Fix skipped. File already exists: {output_sub_file}")
        sub_data = sub_utils.load(output_sub_file)
        return sub_data, _shift(line_range, initial_lines, sub_data), output_sub_file

    logger.debug(f"Fixing mistimed subtitles... Output: {output_sub_file}")
    sub_data = sub_correction.fix_mistimed_lines(sub_data, lines.mistimed_line_thresh_sec, segs, line_range)
    sub_data = sub_correction.remove_empty(sub_data, line_range)
    line_range = _shift(line_range, initial_lines, sub_data)

    sub_utils.write(sub_data, output_sub_file)
    logger.info(f"Subtitles fixed: {output_sub_file}")
    return sub_data, line_range, output_sub_file


def _contextualize(sub_data: SSAFile,
                   source: str,
                   artifacts: Artifacts,
                   lang_from: str,
                   lang_to: str,
                   options: TranslationSettings) -> str:
    """Generates or loads the background notes given to the translator."""
    if source != "auto":
        context_file = Path(source)
        context_data = sub_translation.load_context(context_file)
        logger.info(f"Context loaded: {context_file}")
        return context_data

    context_file = artifacts.context()
    if context_file.exists():
        context_data = sub_translation.load_context(context_file)
    else:
        logger.info(f"Generating context file... Output: {context_file}")
        llm = AIToolFactory.get_llm_provider(
            model_name=options.websearch_model,
            system_prompt="You must follow prompt instructions.",
            max_retries=options.server_retries)
        context_data = sub_translation.generate_context(
            subtitles=sub_data,
            llm=llm,
            filename=Path(artifacts.root).name,
            lang_from=lang_from,
            lang_to=lang_to)
        sub_translation.write_lines([context_data], context_file)

        if options.confirm_auto_context:
            input(f"Context generated:\n{context_data}\n\nContext automatically generated and saved to:\n  '{context_file}'\nPlease review the file if needed. Press any key to continue or Ctrl-C to cancel...")
            context_data = sub_translation.load_context(context_file)

    logger.info(f"Context loaded: {context_file}")
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
               context: str,
               artifacts: Artifacts,
               model: str,
               lang_from: str,
               lang_to: str,
               line_range: LineRange | None,
               options: TranslationSettings) -> tuple[SSAFile, Path]:
    """Translates subtitle text from a source to a target language."""
    output_sub_file = artifacts.translated()

    if output_sub_file.exists():
        logger.info(f"Subtitle Translation skipped. File already exists: {output_sub_file}")
        return sub_utils.load(output_sub_file), output_sub_file

    logger.info(f"Initializing translation to: {output_sub_file}")
    system_prompt = translation_prompts.system_prompt(lang_from, lang_to)
    llm = AIToolFactory.get_llm_provider(
        model_name=model, system_prompt=system_prompt, max_retries=options.server_retries)
    start_index = line_range.start if line_range else 0

    translated_lines = sub_translation.translate(
        sub_file=sub_data,
        output_file=artifacts.partial_translation(),
        context_prompt=context,
        llm=llm,
        fallback_llm=_fallback_llm(system_prompt, model, options),
        lang_from=lang_from,
        lang_to=lang_to,
        options=options,
        line_range=line_range)
    sub_data = sub_utils.replace_lines(
        text_lines=translated_lines[start_index:],
        source_subs=sub_data,
        idx=start_index)
    sub_data = sub_utils.md_to_ass(sub_data)

    sub_utils.write(sub_data, output_sub_file)
    logger.info(f"Subtitles translated: {output_sub_file}")
    return sub_data, output_sub_file


def _pad(sub_data: SSAFile,
         artifacts: Artifacts,
         line_range: LineRange | None,
         padding: PaddingSettings) -> tuple[SSAFile, Path]:
    """Pads subtitle timings to improve readability."""
    output_sub_file = artifacts.padded()

    if output_sub_file.exists():
        logger.info(f"Subtitle padding skipped. File already exists: {output_sub_file}")
        return sub_utils.load(output_sub_file), output_sub_file

    logger.debug(f"Padding subtitles... Output: {output_sub_file}")
    sub_data = sub_correction.apply_timing_standards(sub_data, padding, line_range)
    sub_utils.write(sub_data, output_sub_file)
    logger.info(f"Subtitles padded: {output_sub_file}")
    return sub_data, output_sub_file


def _make_video_from_audio(audio_file: Path, image_file: Path, output_file: Path | None) -> Path:
    output = output_file if formats.is_video(output_file) else audio_file.with_name(f"{audio_file.stem}_template.mp4")
    video_file = av_utils.audio_to_video(image_file, audio_file, output)
    logger.info(f"Created video file '{output.resolve()}' from audio '{audio_file.resolve()}' and image '{image_file.resolve()}'")
    return video_file


def _hardcode(target: Path, video_file: Path | None, sub_file: Path | None, output_file: Path | None) -> None:
    """Burns subtitles into a video. `target` supplies whichever of the two is missing."""
    if formats.is_sub(target):
        if sub_file:
            logger.error(f"Error: Two subtitle files provided: {sub_file} and {target}")
            raise ValueError
        sub_file = target
    elif formats.is_video(target):
        if video_file and video_file.resolve() != target.resolve():
            logger.error(f"Error: Two video files provided: {video_file} and {target}")
            raise ValueError
        video_file = target

    if not (video_file and sub_file):
        logger.warning("Skipping --hardcode: it needs both a video and a subtitle file.")
        return

    if not formats.is_video(output_file):
        output_file = video_file.with_stem(f"{video_file.stem}_subbed")
    av_utils.hardcode_subtitles(video_file, sub_file, output_file)
    logger.info(f"Hardcoded subtitles to '{output_file.resolve()}' from video '{video_file.resolve()}' and subtitles '{sub_file.resolve()}'")


def run(request: RunRequest, settings: Settings) -> None:
    """Runs one job from start to finish."""
    input_file = request.input_file
    video_file = request.video_file
    audio_file = request.audio_file
    sub_file = request.sub_file
    output_file = request.output_file
    lang_from, lang_to = request.lang_from, request.lang_to
    line_range = request.line_range
    transcript: TranscriptionResult | None = None
    sub_data: SSAFile | None = None

    if request.url:
        named_output = output_file if formats.is_video(output_file) else None
        input_file = _download(request.url, named_output, request.output_dir)
        if formats.is_video(input_file):
            video_file = input_file
        elif formats.is_audio(input_file):
            audio_file = input_file

    if input_file is None:
        logger.error("Input file could not be processed.")
        raise ValueError("Input file could not be processed.")

    # Caches are keyed on the range as requested; line_range itself moves between stages.
    artifacts = Artifacts(
        root=file_utils.get_file_root(request.output_dir / input_file.name, lang_to),
        lang_to=lang_to,
        lines_label=line_range.label if line_range else "",
        previous_root=file_utils.get_file_root(input_file, lang_to))

    if video_file and request.extract:
        audio_file = _extract(video_file, artifacts, output_file if formats.is_audio(output_file) else None)

    if audio_file and not sub_file and request.speech_to_text:
        transcript = _transcribe(
            audio_file, artifacts, output_file if formats.is_json(output_file) else None, lang_from, settings.transcription)

    if request.json_file:
        logger.info(f"Loading provided transcript JSON: {request.json_file.resolve()}")
        transcript = transcript_conversion.load_transcript_json(request.json_file)

    if transcript and request.convert:
        sub_data = _convert(transcript, artifacts, settings.parsing)

    if sub_file:
        sub_data = sub_utils.load(sub_file)

    if sub_data and audio_file and request.retranscribe:
        sub_data, line_range, sub_file = _twopass(sub_data, audio_file, artifacts, lang_from, line_range, settings)

    if sub_data and request.fix:
        sub_data, line_range, sub_file = _fix(
            sub_data, artifacts, line_range, settings.mistimed_lines, settings.mistimed_segs)

    if sub_data and request.translate:
        context = _contextualize(sub_data, request.context_source, artifacts, lang_from, lang_to, settings.translation)
        sub_data, sub_file = _translate(
            sub_data, context, artifacts, request.model, lang_from, lang_to, line_range, settings.translation)

    if sub_data and request.pad:
        sub_data, sub_file = _pad(sub_data, artifacts, line_range, settings.subtitle_padding)

    if sub_data and (formats.is_sub(output_file) or formats.is_text(output_file)):
        sub_file = sub_utils.write(sub_data, output_file)
        logger.info(f"Saved subtitles to: {output_file.resolve()}")

    if audio_file and request.image_file:
        video_file = _make_video_from_audio(audio_file, request.image_file, output_file)

    if request.hardcode_target:
        _hardcode(request.hardcode_target, video_file, sub_file, output_file)
