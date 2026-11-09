from argparse import ArgumentParser

REDO_CHOICES = ("extract", "transcribe", "convert", "retranscribe", "context", "translate", "all")


def init_parser() -> ArgumentParser:
    """Builds the command-line parser."""
    parser = ArgumentParser(
        prog="baberu",
        description="Turn audio or video into translated subtitles. With no stage options, runs every stage.")

    parser.add_argument('source_file_path', metavar='SOURCE', help='URL, or a video, audio, transcript (.json) or subtitle file')

    stages = parser.add_argument_group('stages', 'Earlier stages a stage depends on run automatically.')
    stages.add_argument('-A', '--auto-pilot', action='store_true', help='Run every stage (the default when no stage is named)')
    stages.add_argument('-x', '--extract', action='store_true', help='Extract the audio from a video')
    stages.add_argument('-s', '--speech-to-text', action='store_true', help='Transcribe the audio (paid)')
    stages.add_argument('-c', '--convert', action='store_true', help='Turn a transcript into subtitles')
    stages.add_argument('-r', '--retranscribe', action='store_true', help='Find badly timed segments and transcribe them again (paid)')
    stages.add_argument('-f', '--fix', action='store_true', help='Fix subtitle timing')
    stages.add_argument('-t', '--translate', action='store_true', help='Translate the subtitles (paid)')
    stages.add_argument('-p', '--pad', action='store_true', help='Extend subtitle timing for readability')
    stages.add_argument('--hardcode', action='store_true', help='Burn subtitles into the video')
    stages.add_argument('--audio-to-video', metavar='IMAGE', help='Make a video from the audio and this image')

    inputs = parser.add_argument_group('additional inputs', 'Each one turns on the stage that uses it.')
    inputs.add_argument('--audio', metavar='FILE', help='Audio for --retranscribe, if not found beside the source')
    inputs.add_argument('--context', metavar='FILE', help='Background notes for --translate, instead of generating them (paid)')
    inputs.add_argument('--subs', metavar='FILE', help='Subtitles for --hardcode')
    inputs.add_argument('--video', metavar='FILE', help='Video for --hardcode, if not found beside the source')

    output = parser.add_argument_group('output')
    output.add_argument('-o', '--output', metavar='PATH', help='Where to put the final result (a file or a folder)')
    output.add_argument('-d', '--directory', metavar='DIR', help='Folder for the result and its work folder (default: beside the source)')
    output.add_argument('--overwrite', action='store_true', help='Replace an existing file baberu did not write')
    output.add_argument('--clean', action='store_true', help='Delete the work folder after a successful run')

    options = parser.add_argument_group('options')
    options.add_argument('--model', help='Translation model')
    options.add_argument('--lang-from', metavar='CODE', help='Language of the source media')
    options.add_argument('--lang-to', metavar='CODE', help='Language to translate into')
    options.add_argument('--lines', metavar='RANGE', help='Work on only these subtitle lines, e.g. 5-20')
    options.add_argument('--redo', action='append', default=[], choices=REDO_CHOICES, metavar='STAGE',
                         help=f"Throw away a stage's saved work and do it again: {', '.join(REDO_CHOICES)}")
    options.add_argument('--dry-run', action='store_true', help='Show what would happen, without doing or spending anything')
    options.add_argument('--config', metavar='FILE', help='Use this config file')
    options.add_argument('--debug', action='store_true', help='Show detailed logs, and a traceback on errors')

    return parser
