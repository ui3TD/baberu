"""The shape of config.yaml. A setting left out of the file uses the default here."""
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator


class LoggingSettings(BaseModel):
    log_to_file: bool = True
    console_level: str = "INFO"
    file_level: str = "DEBUG"


class TranscriptionSettings(BaseModel):
    # Also read from elevenlabs_model.
    model: str = Field(default="whisper-1", validation_alias=AliasChoices("model", "elevenlabs_model"))
    default_lang_from: str = "ja"
    # Request parameters sent to the service as they are, over baberu's defaults.
    options: dict[str, Any] = {}
    # The second pass can use a different model and options. Empty means the same as the first.
    retranscribe_model: str = ""
    retranscribe_options: dict[str, Any] | None = None
    # Which anomalies send audio to be transcribed again (paid).
    retranscribe_kinds: list[str] = ["smear", "collapse", "loop"]


class ParsingSettings(BaseModel):
    delimiters: list[str] = ["。", "？", "！", "-", "?", "!", "」", "「"]
    soft_delimiters: list[str] = ["、", " ", "　"]
    remove_text: list[str] = ["」", "「"]
    soft_max_chars: int = 20
    hard_max_chars: int = 50
    hard_max_carryover: int = 15
    max_time_gap_sec: float = 2.0
    # A tail of this many letters or fewer is joined back onto its line (乾…杯),
    # unless the pause is longer than fragment_max_gap_sec.
    fragment_max_letters: int = 2
    fragment_max_gap_sec: float = 5.0
    parsing_model: str = "gpt-4.1-mini"
    parsing_effort: str | None = None  # how hard the line splitter thinks; empty: the provider's default
    shift_sec: float = 0.0  # added to every line's start and end; negative: earlier


class MistimedLineSettings(BaseModel):
    mistimed_line_thresh_sec: float = 0.1
    # How the fix stage gives time to lines shorter than the threshold:
    #   "backfill": they take the time before them (silence, or the line before's).
    #   "squeezed": only a line pressed against the next one, after a silence, starts
    #               earlier, by about the time its letters take to say. Other lines
    #               are left to padding.
    method: Literal["backfill", "squeezed"] = "backfill"
    squeezed_letters_per_sec: float = 8.0
    squeezed_min_sec: float = 0.3
    squeezed_max_sec: float = 1.0
    squeezed_touch_sec: float = 0.1  # "pressed against": the next line starts within this


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
    # The second pass cuts each clip this much wider on both sides. Words that land in
    # the margin are dropped.
    clip_margin_sec: float = 1.0


class AnomalySettings(BaseModel):
    """Limits for transcript problems. Physical limits, plus a multiple of the file's own median."""
    smear_sec_per_letter: float = 2.5
    smear_min_sec: float = 3.0
    collapse_sec_per_letter: float = 0.02
    collapse_min_tokens: int = 4
    max_letters_per_sec: float = 25.0
    median_multiple: float = 10.0
    loop_min_repeats: int = 5
    loop_max_tokens: int = 6
    low_confidence_logprob: float = -1.5
    low_confidence_min_tokens: int = 3
    disorder_tolerance_sec: float = 0.05


class TranslationSettings(BaseModel):
    default_model: str = "gemini-2.5-pro"
    websearch_model: str = "gemini-2.5-pro"
    fallback_model: str = "o3"
    effort: str | None = None  # how hard the translation and notes models think; empty: the provider's default
    default_lang_to: str = "en"
    context_lines: int = 100
    batch_lines: int = Field(default=50, ge=1)
    discard_lines: int = 10
    translate_retries: int = Field(default=3, ge=1)
    server_retries: int = 20
    max_cont_lines: int = 5
    confirm_auto_context: bool = True


class KeytermSettings(BaseModel):
    """Names and jargon sent to the transcription service."""
    enabled: bool = True
    dir: str = Field(default="", validation_alias=AliasChoices("dir", "lists_dir"))  # empty: keyterms/ beside the config
    # A YouTube channel_id, mapped to the one file (<dir>/<name>.yaml) that holds every term for it.
    channels: dict[str, str] = {}
    # A channel with no file stops the run before transcribing. False: use the video's details only.
    require_channel_file: bool = True
    # Picks names out of the video's title and description. Empty: use the tags and lists only.
    llm_model: str = ""
    effort: str | None = "low"
    # The most terms sent. Scribe charges 20% more for keyterms, and above 100 bills every
    # request for at least 20 seconds.
    max_terms: int = Field(default=100, ge=1, le=1000)
    confirm: bool = True       # show the list and wait while keyterms.txt is reviewed
    retranscribe: bool = True  # send them on the second pass too

    @field_validator("channels", mode="before")
    @classmethod
    def _one_file_per_channel(cls, value: Any) -> Any:
        for channel_id, name in (value or {}).items():
            if not isinstance(name, str):
                raise ValueError(f"keyterms.channels.{channel_id} must name one file, not {name!r}. "
                                 f"Put all of the channel's terms in that one file.")
        return value


class NekoCapSettings(BaseModel):
    """Uploading finished translations to nekocap.com (baberu publish, --nekocap)."""
    privacy: Literal["public", "unlisted"] = "public"
    title_model: str = ""  # translates the video title, with the job's notes; empty: translation.default_model
    # NekoCap's code for the video's language, by baberu's source language. Others are sent as they are.
    video_languages: dict[str, str] = {"ja": "ja_JP"}
    server_url: str = "https://meow.nekocap.com/parse"
    app_id: str = "nekocap042"


# Sections a profile may change.
PROFILE_SECTIONS = ("transcription", "parsing", "mistimed_lines", "mistimed_segs", "anomalies")


class Profile(BaseModel):
    """Settings for one model or source language, laid over the rest of the config."""
    model: str = ""      # applies when this text is part of transcription.model; empty matches any
    lang: str = ""       # applies to this source language; empty matches any
    settings: dict[str, dict[str, Any]] = {}

    def matches(self, model: str, lang: str) -> bool:
        return self.model.lower() in model.lower() and (not self.lang or self.lang.lower() == lang.lower())


class Settings(BaseModel):
    working_dir: str = ""
    logging: LoggingSettings = LoggingSettings()
    transcription: TranscriptionSettings = TranscriptionSettings()
    parsing: ParsingSettings = ParsingSettings()
    mistimed_lines: MistimedLineSettings = MistimedLineSettings()
    subtitle_padding: PaddingSettings = PaddingSettings()
    mistimed_segs: MistimedSegmentSettings = MistimedSegmentSettings()
    anomalies: AnomalySettings = AnomalySettings()
    translation: TranslationSettings = TranslationSettings()
    keyterms: KeytermSettings = KeytermSettings()
    nekocap: NekoCapSettings = NekoCapSettings()
    # Scribe v1 uses a diarization threshold of 0.1; other models use the service's default.
    profiles: list[Profile] = [Profile(model="scribe_v1",
                                       settings={"transcription": {"options": {"diarization_threshold": 0.1}}})]
