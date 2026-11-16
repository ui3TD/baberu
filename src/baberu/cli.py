"""Turns the command line into a validated RunRequest and starts the pipeline."""
from dataclasses import dataclass, field
from os import environ
from pathlib import Path
import logging
import sys

import dotenv
import ffmpeg
import pydantic

from baberu import pipeline
from baberu.errors import BaberuError
from baberu.job import enclosing_job
from baberu.publish import publish
from baberu.settings import Settings
from baberu.setup import args_setup, config_setup, logging_setup
from baberu.subtitling.types import LineRange
from baberu.tools.file_utils import FileKind, is_url, kind_of, supported_inputs

logger: logging.Logger = logging.getLogger(__name__)

_MEDIA_KINDS = (FileKind.VIDEO, FileKind.AUDIO)


@dataclass
class RunRequest:
    """Everything one run of baberu needs to know, already checked."""

    output_dir: Path
    model: str
    lang_from: str
    lang_to: str
    url: str | None = None
    input_file: Path | None = None
    video_file: Path | None = None
    audio_file: Path | None = None
    json_file: Path | None = None
    sub_file: Path | None = None
    image_file: Path | None = None
    context_file: Path | None = None
    output_file: Path | None = None
    output_folder: Path | None = None
    line_range: LineRange | None = None
    extract: bool = False
    speech_to_text: bool = False
    convert: bool = False
    retranscribe: bool = False
    retranscribe_required: bool = False
    fix: bool = False
    fix_named: bool = False  # -f was given, not just implied by a full run
    translate: bool = False
    pad: bool = False
    hardcode: bool = False
    hardcode_subs: Path | None = None
    hardcode_video: Path | None = None
    overwrite: bool = False
    dry_run: bool = False
    clean: bool = False
    redo: set[str] = field(default_factory=set)
    nekocap: bool = False
    nekocap_privacy: str | None = None  # overrides the config for this run
    keyterms_file: Path | None = None  # terms added to the channel's, from --keyterms
    use_keyterms: bool = True
    keyterms_dir: Path | None = None
    review: bool = True  # pause so generated keyterms and context can be edited


def _existing_file(value: str, description: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise BaberuError(f"{description} does not exist: {value}")
    if path.is_dir():
        raise BaberuError(f"{description} cannot be a folder: {value}")
    return path


def _existing_file_of_kind(value: str, description: str, kind: FileKind) -> Path:
    path = _existing_file(value, description)
    if kind_of(path) is not kind:
        raise BaberuError(f"{description} is not a recognized {kind.name.lower()} file: {value}")
    return path


def _read_source(args, request: RunRequest) -> FileKind:
    """Works out what the main input is. A URL is treated as video until it's downloaded."""
    if is_url(args.source_file_path):
        request.url = args.source_file_path
        return FileKind.VIDEO

    source = _existing_file(args.source_file_path, "Input file")
    kind = kind_of(source)
    request.input_file = source
    if kind is FileKind.VIDEO:
        request.video_file = source
    elif kind is FileKind.AUDIO:
        request.audio_file = source
    elif kind is FileKind.TRANSCRIPT:
        request.json_file = source
    elif kind is FileKind.SUBTITLE:
        request.sub_file = source
    else:
        raise BaberuError(f"baberu can't read {source.name}. Supported types: {supported_inputs()}")
    return kind


def _read_extra_inputs(args, request: RunRequest) -> None:
    if args.audio:
        if request.audio_file:
            raise BaberuError(f"Two audio files provided: {request.audio_file} and {args.audio}")
        request.audio_file = _existing_file_of_kind(args.audio, "Audio file", FileKind.AUDIO)
    if args.context:
        request.context_file = _existing_file(args.context, "Context file")
    if args.subs:
        request.hardcode_subs = _existing_file_of_kind(args.subs, "Subtitle file", FileKind.SUBTITLE)
    if args.video:
        request.hardcode_video = _existing_file_of_kind(args.video, "Video file", FileKind.VIDEO)
    if args.audio_to_video:
        request.image_file = _existing_file_of_kind(args.audio_to_video, "Image file", FileKind.IMAGE)


def _reject_impossible_stages(args, kind: FileKind) -> None:
    earlier_stages = []
    if kind is FileKind.SUBTITLE:
        earlier_stages = [('-x', args.extract), ('-s', args.speech_to_text), ('-c', args.convert)]
        description = "subtitle file"
    elif kind is FileKind.TRANSCRIPT:
        earlier_stages = [('-x', args.extract), ('-s', args.speech_to_text)]
        description = "transcript"
    elif kind is FileKind.AUDIO:
        earlier_stages = [('-x', args.extract)]
        description = "audio file"

    for flag, requested in earlier_stages:
        if requested:
            raise BaberuError(f"{flag} can't be used here: the input is already a {description}.")


def _choose_stages(args, request: RunRequest, kind: FileKind) -> None:
    """Applies the default of running everything, then adds what each stage depends on.

    --nekocap doesn't count as naming a stage: on its own it means a full run, then an upload.
    """
    named_a_stage = any([
        args.extract, args.speech_to_text, args.convert, args.retranscribe, args.fix, args.translate, args.pad,
        args.hardcode, args.audio_to_video, args.audio, args.context, args.subs, args.video,
    ])
    full_run = args.auto_pilot or not named_a_stage

    if not full_run:
        _reject_impossible_stages(args, kind)

    redo = set(args.redo)
    request.redo = redo
    request.extract = full_run or args.extract or "extract" in redo
    request.speech_to_text = full_run or args.speech_to_text or "transcribe" in redo
    request.convert = full_run or args.convert or "convert" in redo
    request.retranscribe = full_run or args.retranscribe or bool(args.audio) or "retranscribe" in redo
    request.retranscribe_required = args.retranscribe or bool(args.audio) or "retranscribe" in redo
    request.fix = full_run or args.fix
    request.fix_named = args.fix
    request.nekocap = args.nekocap or args.nekocap_unlisted
    request.nekocap_privacy = "unlisted" if args.nekocap_unlisted else None
    request.translate = full_run or args.translate or bool(args.context) or bool(redo & {"translate", "context"}) \
        or request.nekocap
    request.pad = full_run or args.pad
    request.hardcode = args.hardcode or bool(args.subs) or bool(args.video)

    needs_subtitles = request.retranscribe or request.fix or request.translate or request.pad
    if kind in (*_MEDIA_KINDS, FileKind.TRANSCRIPT) and needs_subtitles:
        request.convert = True
    if kind in _MEDIA_KINDS and request.convert:
        request.speech_to_text = True
    if kind is FileKind.VIDEO and request.speech_to_text and not args.audio:
        request.extract = True

    if kind in (FileKind.AUDIO, FileKind.TRANSCRIPT, FileKind.SUBTITLE):
        request.extract = False
    if kind in (FileKind.TRANSCRIPT, FileKind.SUBTITLE):
        request.speech_to_text = False
    if kind is FileKind.SUBTITLE:
        request.convert = False
    if args.audio:
        request.extract = False


def _read_output(args, request: RunRequest) -> None:
    """Checks that -o suits what this run will finally produce."""
    if not args.output:
        return

    path = Path(args.output)
    if path.is_dir() or args.output.endswith(("/", "\\")):
        request.output_folder = path
        return

    if request.hardcode or request.image_file:
        allowed, description = (FileKind.VIDEO,), "a video"
    elif request.retranscribe or request.fix or request.translate or request.pad or request.convert:
        allowed, description = (FileKind.SUBTITLE, FileKind.TEXT), "a subtitle or text"
    elif request.speech_to_text:
        allowed, description = (FileKind.TRANSCRIPT,), "a .json"
    elif request.extract:
        allowed, description = (FileKind.AUDIO,), "an audio"
    else:
        allowed, description = (), "any"

    if kind_of(path) not in allowed:
        raise BaberuError(f"-o must be {description} file or a folder for this run: {args.output}")
    request.output_file = path


def build_request(args, settings: Settings) -> RunRequest:
    """Validates command-line arguments against the files on disk."""
    request = RunRequest(
        output_dir=Path.cwd(),
        model=args.model or settings.translation.default_model,
        lang_from=args.lang_from or settings.transcription.default_lang_from,
        lang_to=args.lang_to or settings.translation.default_lang_to,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        clean=args.clean,
        keyterms_file=Path(args.keyterms) if args.keyterms else None,
        use_keyterms=settings.keyterms.enabled and not args.no_keyterms,
        review=not args.yes,
    )
    kind = _read_source(args, request)
    _read_extra_inputs(args, request)

    if args.lines:
        try:
            request.line_range = LineRange.parse(args.lines)
        except ValueError:
            raise BaberuError(f"--lines must look like 5-10 or 5: {args.lines}") from None

    _choose_stages(args, request, kind)
    _read_output(args, request)

    input_dir = None
    if request.input_file:
        enclosing = enclosing_job(request.input_file)
        input_dir = enclosing.parent if enclosing else request.input_file.parent
    request.output_dir = Path(
        args.directory or environ.get("BABERU_DIR") or settings.working_dir or input_dir or Path.cwd())
    return request


def _tolerate_unicode_output() -> None:
    """Keeps Japanese titles intact when output is piped, where Windows defaults to a legacy code page."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            if stream.isatty():
                stream.reconfigure(errors="replace")
            else:
                stream.reconfigure(encoding="utf-8", errors="replace")


def _load_settings(args) -> Settings:
    config_path = Path(args.config) if args.config else None
    if config_path and not config_path.is_file():
        raise BaberuError(f"Config file does not exist: {args.config}")
    try:
        settings = Settings.model_validate(config_setup.load_config(config_path))
    except pydantic.ValidationError as error:
        raise BaberuError(f"The config has settings baberu can't use:\n{error}") from error

    log_dir_text = environ.get("BABERU_LOG_DIR")
    logging_setup.setup_logging(
        console_level="DEBUG" if args.debug else settings.logging.console_level,
        file_level=settings.logging.file_level,
        log_to_file=settings.logging.log_to_file,
        log_dir=Path(log_dir_text) if log_dir_text else None,
    )
    return settings


def _publish(args) -> None:
    settings = _load_settings(args)
    source_dir = None
    if not is_url(args.source_file_path):
        source = Path(args.source_file_path)
        enclosing = enclosing_job(source)
        source_dir = enclosing.parent if enclosing else source.parent
    lang_from = args.lang_from or settings.transcription.default_lang_from
    request = publish.PublishRequest(
        source=args.source_file_path,
        output_dir=Path(args.directory or environ.get("BABERU_DIR") or settings.working_dir or source_dir or Path.cwd()),
        lang_from=lang_from,
        lang_to=args.lang_to or settings.translation.default_lang_to,
        privacy="unlisted" if args.unlisted else ("public" if args.public else None),
        title=args.title,
        retitle=args.retitle,
        dry_run=args.dry_run,
    )
    publish.publish_command(request, config_setup.apply_profiles(settings, lang_from))


def _run(args) -> None:
    config_path = Path(args.config) if args.config else None
    settings = _load_settings(args)
    request = build_request(args, settings)
    terms_dir = settings.keyterms.dir
    request.keyterms_dir = Path(terms_dir) if terms_dir else config_setup.config_dir(config_path) / "keyterms"
    settings = config_setup.apply_profiles(settings, request.lang_from)
    if args.diagnose:
        pipeline.diagnose(request, settings)
        return
    pipeline.run(request, settings)


def _describe(error: Exception) -> str:
    if isinstance(error, ffmpeg.Error) and error.stderr:
        lines = [line for line in error.stderr.decode(errors="replace").splitlines() if line.strip()]
        if lines:
            return f"ffmpeg failed: {lines[-1]}"
    return str(error) or type(error).__name__


def main() -> None:
    dotenv.load_dotenv()
    _tolerate_unicode_output()
    argv = sys.argv[1:]
    if argv[:1] == ["publish"]:
        args, command = args_setup.init_publish_parser().parse_args(argv[1:]), _publish
    else:
        args, command = args_setup.init_parser().parse_args(argv), _run

    try:
        command(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Finished work is saved; run the same command to resume.", file=sys.stderr)
        sys.exit(130)
    except BaberuError as error:
        print(f"baberu: error: {error}", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        if args.debug:
            raise
        logger.debug("Unexpected failure", exc_info=True)
        print(f"baberu: error: {_describe(error)} (run with --debug for details)", file=sys.stderr)
        sys.exit(1)
