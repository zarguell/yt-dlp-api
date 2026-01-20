"""Domain models and request/response schemas."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobType(str, Enum):
    video = "video"
    subtitles = "subtitles"
    subtitles_v2 = "subtitles_v2"
    audio = "audio"


class Task(BaseModel):
    id: str
    job_type: JobType
    url: str
    base_output_path: str
    task_output_path: str
    format: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None


class DownloadRequest(BaseModel):
    url: str
    output_path: str = "default"
    format: str = "bestvideo+bestaudio/best"
    quiet: bool = False
    cookie_file: str | None = Field(
        default=None,
        description="Path to cookies.txt file for authentication (overrides COOKIES_FILE env var)",
    )


class SubtitlesRequest(BaseModel):
    url: str
    output_path: str = "default"
    languages: list[str] = Field(default_factory=lambda: ["en", "en.*"])
    write_automatic: bool = True
    write_manual: bool = True
    convert_to: str | None = "srt"
    quiet: bool = False
    cookie_file: str | None = Field(
        default=None,
        description="Path to cookies.txt file for authentication (overrides COOKIES_FILE env var)",
    )
    max_retries: int | None = Field(
        default=None,
        ge=0,
        description="Maximum number of retry attempts (overrides DEFAULT_MAX_RETRIES env var)",
    )
    retry_backoff: float | None = Field(
        default=None,
        ge=0,
        description="Initial backoff delay in seconds (overrides DEFAULT_RETRY_BACKOFF env var)",
    )


class EnglishMode(str, Enum):
    """Policy for English subtitle selection."""

    best_one = "best_one"
    all_english = "all_english"
    explicit = "explicit"


class SubtitlePreference(str, Enum):
    """Preference for manual vs automatic subtitles."""

    manual_then_auto = "manual_then_auto"
    auto_only = "auto_only"
    manual_only = "manual_only"


class SubtitleFormat(str, Enum):
    """Desired subtitle output format(s)."""

    srt = "srt"
    vtt = "vtt"
    both = "both"


class SubtitlesV2Request(BaseModel):
    """Enhanced subtitles request with policy-based selection."""

    url: str
    output_path: str = "default"

    english_mode: EnglishMode = Field(
        default=EnglishMode.best_one,
        description=(
            "Policy for English subtitle selection. "
            "'best_one' picks the single best English track, "
            "'all_english' downloads all English variants, "
            "'explicit' uses the languages field directly."
        ),
    )
    languages: list[str] = Field(
        default_factory=lambda: [],
        description=(
            "Explicit language list (only used when english_mode='explicit'). "
            "Supports regex patterns like 'en.*'."
        ),
    )

    prefer: SubtitlePreference = Field(
        default=SubtitlePreference.manual_then_auto,
        description=(
            "Preference for manual vs automatic subtitles. "
            "'manual_then_auto' prefers manual subtitles with automatic fallback, "
            "'auto_only' uses only automatic captions, "
            "'manual_only' uses only manual subtitles."
        ),
    )

    formats: SubtitleFormat = Field(
        default=SubtitleFormat.srt,
        description=(
            "Desired subtitle output format(s). "
            "'srt' returns SRT only, 'vtt' returns WebVTT only, "
            "'both' returns both formats."
        ),
    )

    english_rank: list[str] = Field(
        default_factory=lambda: ["en", "en-US", "en-GB", "en.*"],
        description=(
            "Ordered ranking of English language tags for 'best_one' mode. "
            "First available match is selected. Supports regex patterns."
        ),
    )

    quiet: bool = False
    cookie_file: str | None = Field(
        default=None,
        description="Path to cookies.txt file for authentication (overrides COOKIES_FILE env var)",
    )
    max_retries: int | None = Field(
        default=None,
        ge=0,
        description="Maximum number of retry attempts (overrides DEFAULT_MAX_RETRIES env var)",
    )
    retry_backoff: float | None = Field(
        default=None,
        ge=0,
        description="Initial backoff delay in seconds (overrides DEFAULT_RETRY_BACKOFF env var)",
    )


class AudioRequest(BaseModel):
    url: str
    output_path: str = "default"
    audio_format: str = "mp3"
    audio_quality: str | None = None
    quiet: bool = False
    cookie_file: str | None = Field(
        default=None,
        description="Path to cookies.txt file for authentication (overrides COOKIES_FILE env var)",
    )
    max_retries: int | None = Field(
        default=None,
        ge=0,
        description="Maximum number of retry attempts (overrides DEFAULT_MAX_RETRIES env var)",
    )
    retry_backoff: float | None = Field(
        default=None,
        ge=0,
        description="Initial backoff delay in seconds (overrides DEFAULT_RETRY_BACKOFF env var)",
    )
