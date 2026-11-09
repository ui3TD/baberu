"""The shape of config.yaml. A setting left out of the file uses the default here."""
from pydantic import AliasChoices, BaseModel, Field


class LoggingSettings(BaseModel):
    log_to_file: bool = True
    console_level: str = "INFO"
    file_level: str = "DEBUG"


class TranscriptionSettings(BaseModel):
    # Configs written before other services were supported call this elevenlabs_model.
    model: str = Field(default="whisper-1", validation_alias=AliasChoices("model", "elevenlabs_model"))
    default_lang_from: str = "ja"


class ParsingSettings(BaseModel):
    delimiters: list[str] = ["。", "？", "！", "-", "?", "!", "」", "「"]
    soft_delimiters: list[str] = ["、", " ", "　"]
    remove_text: list[str] = ["」", "「"]
    soft_max_chars: int = 20
    hard_max_chars: int = 50
    hard_max_carryover: int = 15
    max_time_gap_sec: float = 2.0
    parsing_model: str = "gpt-4.1-mini"


class MistimedLineSettings(BaseModel):
    mistimed_line_thresh_sec: float = 0.1


class PaddingSettings(BaseModel):
    max_lead_out_sec: float = 2.0
    max_lead_in_sec: float = 0.25
    max_cps: float = 15.0
    min_sec: float = 1.0


class MistimedSegmentSettings(BaseModel):
    mistimed_seg_thresh_sec: float = 0.1
    seg_min_lines: int = 4
    seg_backtrace_limit: int = 20
    seg_foretrace_limit: int = 5
    seg_min_delay: float = 10.0
    seg_max_gap: int = 4


class TranslationSettings(BaseModel):
    default_model: str = "gemini-2.5-pro"
    websearch_model: str = "gemini-2.5-pro"
    fallback_model: str = "o3"
    default_lang_to: str = "en"
    context_lines: int = 100
    batch_lines: int = 50
    discard_lines: int = 10
    translate_retries: int = 3
    server_retries: int = 20
    max_cont_lines: int = 5
    confirm_auto_context: bool = True


class Settings(BaseModel):
    working_dir: str = ""
    logging: LoggingSettings = LoggingSettings()
    transcription: TranscriptionSettings = TranscriptionSettings()
    parsing: ParsingSettings = ParsingSettings()
    mistimed_lines: MistimedLineSettings = MistimedLineSettings()
    subtitle_padding: PaddingSettings = PaddingSettings()
    mistimed_segs: MistimedSegmentSettings = MistimedSegmentSettings()
    translation: TranslationSettings = TranslationSettings()
