import pydantic

class SubtitleLine(pydantic.BaseModel):
    text: str
    start: float
    end: float
    speaker: str = "Unknown"