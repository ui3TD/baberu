APP_NAME: str = "baberu"
CONTINUE_FLAG: str = "%%CONT%%"

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