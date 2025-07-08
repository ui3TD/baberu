from typing import Any
from pathlib import Path
from os import environ
import logging
import json

import dotenv
from pysubs2 import SSAFile

from baberu.setup import config_setup, logging_setup, args_setup
from baberu.subtitling import elevenlabs_utils, sub_correction, sub_translation, sub_twopass, sub_utils
from baberu.tools import av_utils, file_utils
from baberu.tools.file_utils import formats
from baberu.LLMFactory.factory import AIToolFactory
from baberu.LLMFactory.transcription.base import TranscriptionResult

app_config: dict[str, Any] = None
logger: logging.Logger = None

def _download(url: Path,
              output_file: Path | None = None, 
              output_dir: Path | None = None) -> Path:
    """Downloads a file from a URL if it doesn't already exist."""
    if output_file and output_file.exists():
        logger.warning(f"Download skipped. File already exists: {output_file}")
        return output_file

    logger.debug(f"Downloading from {url}...")
    downloaded_file: Path = av_utils.download(url, output_filename=output_file, download_directory=output_dir)
    logger.info(f"Downloaded to: {downloaded_file}")
    return downloaded_file

def _extract(video_file: Path,
             output_root: str, 
             output_file: Path | None) -> Path:
    """Extracts the audio stream from a video file."""
    audio_file: Path = None

    codec_name = av_utils.get_audio_codec(video_file)
    if not codec_name:
        logger.error("No audio codec found for direct copy extraction.")
        raise ValueError(
            f"No audio codec found for direct copy extraction."
        )
    
    output_audio_file: Path = output_file or Path(output_root + "." + codec_name)
    
    if output_audio_file.exists():
        logger.warning(f"Extraction skipped. File already exists: {output_audio_file}")
        return output_audio_file
    
    logger.debug(f"Extracting audio from: {video_file} to {output_audio_file}")
    audio_file = av_utils.extract_audio(video_file, output_audio_file)
    logger.info(f"Audio extracted: {audio_file}")
    
    return audio_file

def _transcribe(audio_file: Path,
                output_root: str, 
                output_file: Path | None,
                lang: str) -> TranscriptionResult:
    """Transcribes an audio file into a TranscriptionResult object."""
    config = app_config['transcription']
    model: str = config['elevenlabs_model']

    json_file: Path = output_file or Path(output_root + ".json")
    transcript: TranscriptionResult = None 
    transcript_provider_type = AIToolFactory.get_transcription_provider_type(model)

    if json_file.exists():
        logger.warning(f"Transcription skipped. File already exists: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        json_data = transcript_provider_type.validate(json_data)
    else:
        logger.debug(f"Transcribing audio from: {audio_file} to {json_file}")
        transcript_provider = AIToolFactory.get_transcription_provider(model)
        provider_response = transcript_provider.transcribe(audio_file, lang=lang)
        elevenlabs_utils.write_transcript_json(provider_response, json_file)
        logger.info(f"Audio transcribed: {json_file}")

    transcript: TranscriptionResult = transcript_provider_type.parse(provider_response)
    return transcript

def _convert(transcript: TranscriptionResult,
             output_root: str) -> SSAFile:
    """Converts a transcription JSON object to a subtitle file."""
    config = app_config['parsing']
    delimiters: str = config['delimiters']
    soft_delimiters: str = config['soft_delimiters']
    soft_max_lines: int = config['soft_max_lines']
    hard_max_lines: int = config['hard_max_lines']
    hard_max_carryover: int = config['hard_max_carryover']
    parsing_model: str = config['parsing_model']

    output_sub_file = Path(output_root + ".raw.ass")
    sub_data: SSAFile = None

    if output_sub_file.exists() :
        logger.warning(f"Conversion skipped. File already exists: {output_sub_file}")
        sub_data = sub_utils.load(output_sub_file)
        return sub_data

    logger.debug(f"Converting transcription JSON to subtitles: {output_sub_file}")
    sub_data = elevenlabs_utils.convert_transcript_to_subs(transcript, delimiters, soft_delimiters, soft_max_lines, hard_max_lines, hard_max_carryover, parsing_model)
    sub_utils.write(sub_data, output_sub_file)
    logger.info(f"Transcription converted: {output_sub_file}")

    return sub_data

def _twopass(sub_data: SSAFile, 
             audio_file: Path,
             output_root: str,
             lang: str,
             segment: list[int] = []) -> tuple[SSAFile, list[int], Path]:
    """Performs a two-pass transcription to correct poorly transcribed segments."""
    transcription_config = app_config['transcription']
    parsing_config = app_config['parsing']
    mistiming_config = app_config['mistimed_segs']
    transcription_model: str = transcription_config['elevenlabs_model']
    delimiters: str = parsing_config['delimiters']
    soft_delimiters: str = parsing_config['soft_delimiters']
    soft_max_lines: int = parsing_config['soft_max_lines']
    hard_max_lines: int = parsing_config['hard_max_lines']
    hard_max_carryover: int = parsing_config['hard_max_carryover']
    parsing_model: int = parsing_config['parsing_model']
    mistimed_seg_thresh_sec: float = mistiming_config['mistimed_seg_thresh_sec']
    seg_min_lines: int = mistiming_config['seg_min_lines']
    seg_backtrace_limit: int = mistiming_config['seg_backtrace_limit']
    seg_foretrace_limit: int = mistiming_config['seg_foretrace_limit']
    seg_min_delay: float = mistiming_config['seg_min_delay']
    seg_max_gap: int = mistiming_config['seg_max_gap']
    
    initial_lines = len(sub_data.events)

    if segment:
        output_sub_file = Path(output_root + ".2pass_custom.ass")

        if output_sub_file.exists():
            logger.warning(f"Retranscription skipped. File already exists: {output_sub_file}")
            sub_data = sub_utils.load(output_sub_file)
            new_lines = len(sub_data.events) - initial_lines
            segment = [min(segment), max(segment) + new_lines]
            return sub_data, segment, output_sub_file
        
        segments = [segment]
    else:
        output_sub_file = Path(output_root + ".2pass.ass")
    
        if output_sub_file.exists():
            logger.warning(f"Retranscription skipped. File already exists: {output_sub_file}")
            sub_data = sub_utils.load(output_sub_file)
            return sub_data, segment, output_sub_file
            
        segments: list[list[int]] = sub_twopass.find_segments(sub_data, mistimed_seg_thresh_sec, seg_min_lines, seg_backtrace_limit, seg_foretrace_limit, seg_min_delay, seg_max_gap)
        if len(segments) == 0:
            logger.info("Retranscription skipped. No segments identified for retranscription.")
            return sub_data, segment, output_sub_file

        logger.debug(f"Retranscribing {len(segments)} segments for two-pass process...")
        segments = sub_twopass.pad_segments(sub_data, segments)
    
    sub_data = sub_twopass.transcribe_segments(sub_data, segments, audio_file, lang, delimiters, soft_delimiters, soft_max_lines, hard_max_lines, hard_max_carryover, transcription_model, parsing_model)
    sub_data = sub_correction.remove_empty(sub_data)
    
    if segment:
        new_lines = len(sub_data.events) - initial_lines
        segment = [min(segment), max(min(segment), max(segment) + new_lines)]

    sub_utils.write(sub_data, output_sub_file)
    logger.info(f"Retranscription processed: {output_sub_file}")
    
    return sub_data, segment, output_sub_file

def _fix(sub_data: SSAFile, 
         output_root: str,
         segment: list[int] = []) -> tuple[SSAFile, list[int]]:  
    """Fixes mistimed subtitle lines by merging and adjusting them."""  
    mistiming_config = app_config['mistimed_lines']
    segs_config = app_config['mistimed_segs']
    mistimed_line_thresh_sec: float = mistiming_config['mistimed_line_thresh_sec']
    seg_min_lines: int = segs_config['seg_min_lines']
    seg_backtrace_limit: int = segs_config['seg_backtrace_limit']
    seg_foretrace_limit: int = segs_config['seg_foretrace_limit']
    seg_min_delay: float = segs_config['seg_min_delay']
    seg_max_gap: int = segs_config['seg_max_gap']

    initial_lines = len(sub_data.events)

    if segment:
        output_sub_file = Path(output_root + ".fixed_custom.ass")
    else:
        output_sub_file = Path(output_root + ".fixed.ass")

    if output_sub_file.exists():
        logger.warning(f"Subtitle Fix skipped. File already exists: {output_sub_file}")
        sub_data = sub_utils.load(output_sub_file)
        return sub_data, segment, output_sub_file
    
    logger.debug(f"Fixing mistimed subtitles... Output: {output_sub_file}")
    sub_data = sub_correction.fix_mistimed_lines(sub_data, mistimed_line_thresh_sec, seg_min_lines, seg_backtrace_limit, seg_foretrace_limit, seg_min_delay, seg_max_gap, segment)
    sub_data = sub_correction.remove_empty(sub_data, segment)

    if segment:
        new_lines = len(sub_data.events) - initial_lines
        segment = [min(segment), max(min(segment), max(segment) + new_lines)]

    sub_utils.write(sub_data, output_sub_file)
    logger.info(f"Subtitles fixed: {output_sub_file}")

    return sub_data, segment, output_sub_file

def _contextualize(sub_data: SSAFile, 
                   instruction: str, 
                   output_root: str, 
                   model: str,
                   lang_from: str,
                   lang_to: str) -> str:
    """Generates or loads contextual information for translation."""
    config = app_config['translation']
    confirm_auto_context: bool = config['confirm_auto_context']

    if instruction == "auto":
        context_file = Path(output_root + ".context.txt")
        if context_file.exists():
            context_data = sub_translation.load_context(context_file)
        else:
            logger.debug(f"Generating context file... Output: {context_file}")
            context_data = sub_translation.generate_context(sub_data, model, Path(output_root).name, lang_from, lang_to)
            sub_translation.write_lines([context_data], context_file)

            if confirm_auto_context:
                input(f"Context generated:\n{context_data}\n\nContext automatically generated and saved to:\n  '{context_file}'\nPlease review the file if needed. Press any key to continue or Ctrl-C to cancel...")
                context_data = sub_translation.load_context(context_file)
    else:
        context_data = sub_translation.load_context(Path(instruction))
    
    logger.info(f"Context generated: {context_file}")
    return context_data

def _translate(sub_data: SSAFile, 
               context: str,
               output_root: str, 
               model: str,
               lang_from: str,
               lang_to: str,
               segment: list[int] = []) -> SSAFile:
    """Translates subtitle text from a source to a target language."""
    config = app_config['translation']
    context_lines: int = config['context_lines']
    batch_lines: int = config['batch_lines']
    discard_lines: int = config['discard_lines']
    translate_retries: int = config['translate_retries']
    server_retries: int = config['server_retries']
    max_cont_lines: int = config['max_cont_lines']
    
    if segment:
        output_sub_file = Path(output_root + ".tr_custom.ass")

        if output_sub_file.exists():
            logger.warning(f"Subtitle Translation skipped. File already exists: {output_sub_file}")
            sub_data = sub_utils.load(output_sub_file)
            return sub_data, output_sub_file
        
        partial_file = Path(output_root + ".partial.tr_custom.txt")
        start_index = min(segment)
    else:
        output_sub_file = Path(output_root + ".en.ass")

        if output_sub_file.exists():
            logger.warning(f"Subtitle Translation skipped. File already exists: {output_sub_file}")
            sub_data = sub_utils.load(output_sub_file)
            return sub_data, output_sub_file

        partial_file = Path(output_root + ".partial.en.txt")
        start_index = 0

    txt_data = sub_translation.translate(sub_data, partial_file, context, model, lang_from, lang_to, context_lines, batch_lines, discard_lines, translate_retries, server_retries, max_cont_lines, segment)
    sub_data = sub_utils.replace_text(txt_data[start_index:], sub_data, start_index)

    if output_sub_file.suffix == ".ass":
        sub_data = sub_utils.md_to_ass(sub_data)
    
    sub_utils.write(sub_data, output_sub_file)
    logger.info(f"Subtitles translated: {output_sub_file}")

    return sub_data, output_sub_file

def _pad(sub_data: SSAFile, 
            output_root: str,
            segment: list[int] = []) -> SSAFile:   
    """Pads subtitle timings to improve readability."""
    config = app_config['subtitle_padding']
    max_lead_out_sec: float = config['max_lead_out_sec']
    max_lead_in_sec: float = config['max_lead_in_sec']
    max_cps: float = config['max_cps']
    min_sec: float = config['min_sec']
    
    if segment:
        output_sub_file = Path(output_root + ".padded_custom.ass")
    else:
        output_sub_file = Path(output_root + ".padded.ass")

    if output_sub_file.exists():
        logger.warning(f"Subtitle padding skipped. File already exists: {output_sub_file}")
        sub_data = sub_utils.load(output_sub_file)
        return sub_data, output_sub_file

    logger.debug(f"Padding subtitles... Output: {output_sub_file}")
    sub_data = sub_correction.apply_timing_standards(sub_data, max_lead_out_sec, max_lead_in_sec, max_cps, min_sec, segment)

    sub_utils.write(sub_data, output_sub_file)
    logger.info(f"Subtitles padded: {output_sub_file}")

    return sub_data, output_sub_file

def main():
    dotenv.load_dotenv()

    global app_config, logger
    app_config = config_setup.load_config()
    
    log_dir_str = environ.get("BABERU_LOG_DIR")
    log_dir = Path(log_dir_str) if log_dir_str else None

    logging_setup.setup_logging(
        console_level=app_config['logging']['console_level'],
        file_level=app_config['logging']['file_level'],
        log_to_file=app_config['logging']['log_to_file'],
        log_dir=log_dir
    )
    
    logger = logging.getLogger(__name__)

    # Initialize variables
    url: str | None = None
    video_file: Path | None = None
    audio_file: Path | None = None
    image_file: Path | None = None
    json_file: Path | None = None
    sub_file: Path | None = None
    transcript_data: TranscriptionResult | None = None
    sub_data: SSAFile | None = None
    context_data: str | None = None
    segment: list[int] = []

    # Parse arguments
    parser = args_setup.init_parser()
    args: args_setup.args = parser.parse_args()
    
    # Set defaults
    model = args.model or app_config['translation']['default_model']
    lang_from = args.lang_from or app_config['transcription']['default_lang_from']
    lang_to = args.lang_to or app_config['translation']['default_lang_to']
    websearch_model = app_config['translation']['websearch_model']
    transcription_model = app_config['transcription']['elevenlabs_model']

    # Set output file
    output_file: Path | None = None
    if args.output:
        output_file = Path(args.output)
    
    # Check input file type
    input_file: Path | None = None
    if formats.is_url(args.source_file_path):
        url = args.source_file_path
    else:
        input_file = Path(args.source_file_path)
        if formats.is_video(input_file):
            video_file = input_file
        elif formats.is_audio(input_file):
            audio_file = input_file
        elif formats.is_json(input_file):
            json_file = input_file
        elif formats.is_sub(input_file):
            sub_file = input_file
        else:
            logger.error(f"Invalid input file")
            raise ValueError

    # Validate special args
    if args.retranscribe or args.auto_pilot:
        if args.retranscribe == "auto" or args.auto_pilot:
            if audio_file:
                args.retranscribe = audio_file
        else:
            retrans_path = Path(args.retranscribe)
            if not retrans_path.exists():
                logger.error(f"Retranscribe file does not exist: {args.retranscribe}")
                raise FileNotFoundError
            if retrans_path.is_dir():
                logger.error(f"Retranscribe path cannot be a directory: {args.retranscribe}")
                raise IsADirectoryError
            if not file_utils.formats.is_audio(retrans_path):
                logger.error(f"Retranscribe file is not a recognized audio format: {args.retranscribe}")
                raise ValueError
            if audio_file:
                logger.error(f"Error: Two audio files provided: {audio_file} and {args.retranscribe}")
                raise ValueError
            audio_file = retrans_path

    if args.translate or args.auto_pilot:
        if args.translate == "auto" or args.auto_pilot:
            pass
        else:
            context_path = Path(args.translate)
            if context_path.name == "auto":
                logger.error(f"Translation context file cannot be named 'auto' because 'auto' is a special instruction: {args.translate}")
                raise ValueError
            if not context_path.exists():
                logger.error(f"Context file does not exist: {args.translate}")
                raise FileNotFoundError
            if context_path.is_dir():
                logger.error(f"Translation context file cannot be a directory: {args.translate}")
                raise IsADirectoryError
            
    if args.lines:
        try:
            if '-' in args.lines:
                start, end = map(int, args.lines.split('-'))
            else:
                start = end = int(args.lines)
            
            # Create 0-index segment
            segment = [start - 1, end - 1]
        except ValueError:
            logger.error(f"Line range must be in format XX-YY (e.g. 5-10) or XX (e.g. 5): {args.lines}")
            raise

    if args.audio_to_video:
        image_file = Path(args.audio_to_video)
        if not formats.is_image(image_file):
            logger.error(f"Audio-to-Video file is not a recognized image format: {args.audio_to_video}")
            raise ValueError
        if not Path(image_file).exists():
            logger.error(f"Audio-to-Video image file does not exist: {args.translate}")
            raise FileNotFoundError
        if Path(image_file).is_dir():
            logger.error(f"Audio-to-Video image file cannot be a directory: {args.translate}")
            raise IsADirectoryError

    environ.setdefault('BABERU_DIR', app_config['working_dir'])
    environ.setdefault('GEMINI_API_KEY', app_config['keys']['gemini'])
    environ.setdefault('ELEVENLABS_API_KEY', app_config['keys']['elevenlabs'])

    # Prepare ouput dir
    input_dir: Path = input_file.parent if input_file else None
    output_dir: Path = Path(args.directory or environ.get("BABERU_DIR") or input_dir or Path.cwd())
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download
    if url:
        name_defined: bool = file_utils.formats.is_video(output_file)
        input_file = _download(url, output_file if name_defined else None, output_dir)
        if file_utils.formats.is_video(input_file):
            video_file = input_file
        elif file_utils.formats.is_audio(input_file):
            audio_file = input_file

    # Discover output root
    output_root: str = file_utils.get_file_root(output_dir / input_file)

    # Extract audio
    if video_file and (args.extract or args.auto_pilot):
        name_defined: bool = formats.is_audio(output_file)
        audio_file = _extract(video_file, output_root, 
                              output_file if name_defined else None)

    # Transcribe
    if audio_file and (args.speech_to_text or args.auto_pilot):
        force_write: bool = formats.is_json(output_file)
        transcript_data = _transcribe(audio_file, output_root, 
                                output_file if force_write else None,
                                lang_from)

    # Load transcription
    if json_file:
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        transcript_provider_type = AIToolFactory.get_transcription_provider_type(transcription_model)
        json_data = transcript_provider_type.validate(json_data)
        transcript_data = transcript_provider_type.parse(json_data)
    
    # Create subtitles
    if transcript_data and (args.convert or args.auto_pilot):
        sub_data = _convert(transcript_data, output_root)

    # Load subtitles
    if sub_file:
        sub_data = sub_utils.load(sub_file)
    
    # Retranscribe subtitles
    if sub_data and (args.retranscribe or args.auto_pilot):
        if not args.retranscribe:
            args.retranscribe = "auto"
        
        sub_data, segment, sub_file = _twopass(sub_data, audio_file, output_root, lang_from, segment)

    # Fix mistimed segments
    if sub_data and (args.fix or args.auto_pilot):
        sub_data, segment, sub_file = _fix(sub_data, output_root, segment)
        
    # Translate subtitles
    if sub_data and (args.translate or args.auto_pilot):
        if not args.translate:
            args.translate = "auto"

        context_data = _contextualize(sub_data, args.translate, output_root, websearch_model, lang_from, lang_to)            
        sub_data, sub_file = _translate(sub_data, context_data, output_root, model, lang_from, lang_to, segment)

    # Add padding time to subtitles
    if sub_data and (args.pad or args.auto_pilot):
        sub_data, sub_file = _pad(sub_data, output_root, segment)

    # Write final subtitle file
    if sub_data and formats.is_sub(output_file):
        sub_file = sub_utils.write(sub_data, output_file)
    
    if sub_data and formats.is_text(output_file):
        sub_file = sub_utils.write(sub_data, output_file)
        
    if audio_file and image_file and args.audio_to_video:
        if formats.is_video(output_file):
            output_vid_file = output_file
        else:
            output_vid_file = audio_file.with_name(f"{audio_file.stem}_template.mp4")
        
        video_file = av_utils.audio_to_video(image_file, audio_file, output_vid_file)

    
    if args.hardcode:
        hardcode_path: Path = Path(args.hardcode)
        if formats.is_sub(hardcode_path):
            if sub_file:
                logger.error(f"Error: Two subtitle files provided: {sub_file} and {args.hardcode}")
                raise ValueError
            else:
                sub_file = Path(hardcode_path)
        elif formats.is_video(hardcode_path):
            if video_file:
                logger.error(f"Error: Two video files provided: {video_file} and {args.hardcode}")
                raise ValueError
            else:
                video_file = Path(hardcode_path)
            
        if video_file and sub_file:
            if not formats.is_video(output_file):
                output_file = video_file.with_stem(f"{video_file.stem}_subbed")
            av_utils.hardcode_subtitles(video_file, sub_file, output_file)
    

if __name__ == "__main__":
    main()