"""Turns the command line into a validated RunRequest and starts the pipeline."""
from dataclasses import dataclass
from os import environ
from pathlib import Path
import logging

import dotenv

from baberu import pipeline
from baberu.settings import Settings
from baberu.setup import args_setup, config_setup, logging_setup
from baberu.subtitling.types import LineRange
from baberu.tools.file_utils import formats

logger: logging.Logger = logging.getLogger(__name__)


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
    hardcode_target: Path | None = None
    output_file: Path | None = None
    line_range: LineRange | None = None
    extract: bool = False
    speech_to_text: bool = False
    convert: bool = False
    retranscribe: bool = False
    fix: bool = False
    translate: bool = False
    context_source: str = "auto"
    pad: bool = False


def _reject(message: str, error: type[Exception] = ValueError) -> None:
    logger.error(message)
    raise error(message)


def _check_file(path: Path, description: str, value: object) -> None:
    if not path.exists():
        _reject(f"{description} does not exist: {value}", FileNotFoundError)
    if path.is_dir():
        _reject(f"{description} cannot be a directory: {value}", IsADirectoryError)


def _parse_line_range(text: str) -> LineRange:
    try:
        return LineRange.parse(text)
    except ValueError:
        logger.error(f"Line range must be in format XX-YY (e.g. 5-10) or XX (e.g. 5): {text}")
        raise


def build_request(args, settings: Settings) -> RunRequest:
    """Validates command-line arguments against the files on disk."""
    autopilot: bool = args.auto_pilot
    request = RunRequest(
        output_dir=Path.cwd(),
        model=args.model or settings.translation.default_model,
        lang_from=args.lang_from or settings.transcription.default_lang_from,
        lang_to=args.lang_to or settings.translation.default_lang_to,
        output_file=Path(args.output) if args.output else None,
        hardcode_target=Path(args.hardcode) if args.hardcode else None,
        extract=args.extract or autopilot,
        speech_to_text=args.speech_to_text or autopilot,
        convert=args.convert or autopilot,
        retranscribe=bool(args.retranscribe) or autopilot,
        fix=args.fix or autopilot,
        translate=bool(args.translate) or autopilot,
        context_source=args.translate or "auto",
        pad=args.pad or autopilot,
    )

    if formats.is_url(args.source_file_path):
        request.url = args.source_file_path
    else:
        source = Path(args.source_file_path)
        request.input_file = source
        if formats.is_video(source):
            request.video_file = source
        elif formats.is_audio(source):
            request.audio_file = source
        elif formats.is_json(source):
            request.json_file = source
        elif formats.is_sub(source):
            request.sub_file = source
        else:
            _reject(f"Invalid input file: {source}")

    if args.retranscribe and args.retranscribe != "auto":
        audio_path = Path(args.retranscribe)
        _check_file(audio_path, "Retranscribe file", args.retranscribe)
        if not formats.is_audio(audio_path):
            _reject(f"Retranscribe file is not a recognized audio format: {args.retranscribe}")
        if request.audio_file:
            _reject(f"Two audio files provided: {request.audio_file} and {args.retranscribe}")
        request.audio_file = audio_path

    if args.translate and args.translate != "auto":
        context_path = Path(args.translate)
        if context_path.name == "auto":
            _reject(f"Translation context file cannot be named 'auto' because 'auto' is a special instruction: {args.translate}")
        _check_file(context_path, "Context file", args.translate)

    if args.lines:
        request.line_range = _parse_line_range(args.lines)

    if args.audio_to_video:
        image_path = Path(args.audio_to_video)
        if not formats.is_image(image_path):
            _reject(f"Audio-to-Video file is not a recognized image format: {args.audio_to_video}")
        _check_file(image_path, "Audio-to-Video image file", args.audio_to_video)
        request.image_file = image_path

    input_dir = request.input_file.parent if request.input_file else None
    request.output_dir = Path(
        args.directory or environ.get("BABERU_DIR") or settings.working_dir or input_dir or Path.cwd())
    request.output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory set to {request.output_dir.resolve()}")

    return request


def main() -> None:
    dotenv.load_dotenv()

    args = args_setup.init_parser().parse_args()

    config_path = Path(args.config) if args.config else None
    if config_path and not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {args.config}")
    settings = Settings.model_validate(config_setup.load_config(config_path))

    log_dir_text = environ.get("BABERU_LOG_DIR")
    logging_setup.setup_logging(
        console_level=settings.logging.console_level,
        file_level=settings.logging.file_level,
        log_to_file=settings.logging.log_to_file,
        log_dir=Path(log_dir_text) if log_dir_text else None,
    )

    pipeline.run(build_request(args, settings), settings)
