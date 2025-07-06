from argparse import ArgumentParser, Namespace

import logging

logger = logging.getLogger(__name__)

class args(Namespace):
    source_file_path: str
    directory: str | None
    extract: bool | None
    speech_to_text: bool | None
    convert: bool | None
    retranscribe: str | None
    translate: str | None
    fix: bool | None
    buffer: bool | None
    output: str | None
    wizard: bool | None
    num_speakers: int | None
    model: str | None
    lang_to: str | None
    lang_from: str | None
    auto_pilot: bool | None
    hardcode: str | None
    audio_to_video: str | None
    retranscribe_lines: str | None
    lines: str | None

def init_parser() -> ArgumentParser:
    """Initializes and configures the command-line interface argument parser.

    Returns:
        ArgumentParser: The configured argument parser instance.
    """
    # Set up argument parser
    parser = ArgumentParser(description='Process audio/video into subtitles')
    
    # Required arguments
    parser.add_argument('source_file_path', help='Path to the source file')
    
    # Optional arguments
    parser.add_argument('-d', '--directory', 
                        type=str,
                        help='Output directory of all processed files')
    parser.add_argument('-x', '--extract', 
                        action='store_true', 
                        help='Extract audio from video file')
    parser.add_argument('-s', '--speech-to-text', 
                        action='store_true', 
                        help='Convert audio speech to text via ElevenLabs (segmented JSON format)')
    parser.add_argument('-c', '--convert', 
                        action='store_true', 
                        help='Convert ElevenLabs segmented JSON into subtitles')
    parser.add_argument('-r', '--retranscribe',
                        type=str,
                        metavar='AUDIO FILE',
                        help='Find and retranscribed mistimed segments with provided audio ("auto" to use extracted file)')
    parser.add_argument('-t', '--translate',  
                        type=str,
                        metavar='CONTEXT FILE',
                        help='Translate the subtitles from one language into another via LLM with provided context ("auto" to auto-generate)')
    parser.add_argument('-f', '--fix', 
                        action='store_true', 
                        help='Fix subtitle timing and formatting')
    parser.add_argument('-p', '--pad', 
                        action='store_true', 
                        help='Add padding time by extending subtitles')
    parser.add_argument('--hardcode', 
                        type=str,
                        metavar='SUBTITLE/VIDEO FILE',
                        help='Hardcode subtitles to a video by provide either a subtitle or video path')
    parser.add_argument('--audio-to-video', 
                        type=str,
                        metavar='IMAGE FILE',
                        help='Creates a video with the provided image file and the main audio stream')
    parser.add_argument('-o', '--output',
                        metavar='OUTPUT FILE',
                        help='Path to the output file')
    parser.add_argument('--num-speakers',
                        type=int,
                        help='Number of speakers')
    parser.add_argument('--model',
                        type=str,
                        help='Name of LLM model for translation')
    parser.add_argument('--lang-to',
                        type=str,
                        help='Language code of source media')
    parser.add_argument('--lang-from',
                        type=str,
                        help='Language code of translation')
    parser.add_argument('-A', '--auto-pilot',
                        action='store_true', 
                        help='Automatically apply all processing steps')
    parser.add_argument('--lines',
                   type=str,
                   help='Line number range in format XX-YY (e.g. 5-20)')
    
    return parser
