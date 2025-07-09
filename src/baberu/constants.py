APP_NAME: str = "baberu"
CONTINUE_FLAG: str = "%%CONT%%"

VIDEO_SUFFIXES = frozenset([".mp4", ".mkv", ".webm", ".ts", ".flv", ".mov", ".avi", ".wmv"])
AUDIO_SUFFIXES = frozenset([".oga", ".mp3", ".aac", ".m4a", ".wav", ".flac", ".opus", ".ogg", ".ac3", ".eac3"])
JSON_SUFFIXES = frozenset([".json"])
SUBTITLE_SUFFIXES = frozenset([".srt", ".ass", ".ssa", ".vtt", ".sbv"])
TEXT_SUFFIXES = frozenset([".txt"])
IMAGE_SUFFIXES = frozenset([".jpg", ".bmp", ".png", ".jpeg", ".webp"])

BABERU_SUFFIXES = ['.padded', '.en', '.padded_custom', '.tr_custom', '.fixed_custom', '.partial', '.2pass', '.2pass_custom', '.context', '.fixed', '.raw'] 

CODEC_TO_EXTENSION_MAP = {
    "aac": "m4a",        # Advanced Audio Coding, often in an MP4 container (.m4a)
    "mp3": "mp3",        # MPEG Layer 3
    "opus": "ogg",      # Opus Interactive Audio Codec
    "vorbis": "ogg",     # Vorbis, typically in an Ogg container
    "flac": "flac",      # Free Lossless Audio Codec
    "ac3": "ac3",        # Dolby Digital
    "eac3": "eac3",      # Dolby Digital Plus
    "pcm_s16le": "wav",  # Common uncompressed PCM format
    "pcm_s24le": "wav",
    "pcm_f32le": "wav",
}