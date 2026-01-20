"""Configuration classes loaded from environment variables."""

import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger("yt-dlp-api")

# ----------------------------
# Environment variable names
# ----------------------------

DEFAULT_API_KEY_HEADER_NAME = "X-API-Key"
DEFAULT_API_KEY_ENABLED_ENV = "API_KEY_AUTH_ENABLED"
DEFAULT_MASTER_API_KEY_ENV = "API_MASTER_KEY"

DEFAULT_MAX_RETRIES_ENV = "DEFAULT_MAX_RETRIES"
DEFAULT_RETRY_BACKOFF_ENV = "DEFAULT_RETRY_BACKOFF"
DEFAULT_RETRY_BACKOFF_MULTIPLIER_ENV = "DEFAULT_RETRY_BACKOFF_MULTIPLIER"
DEFAULT_RETRY_JITTER_ENV = "DEFAULT_RETRY_JITTER"

DEFAULT_COOKIES_FILE_ENV = "COOKIES_FILE"


# ----------------------------
# Helper functions
# ----------------------------


def _env_truthy(value: str | None, *, default: bool = False) -> bool:
    """Parse common truthy/falsey strings from environment variables."""
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _env_int(value: str | None, *, default: int) -> int:
    """Parse integer from environment variable with default."""
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(value: str | None, *, default: float) -> float:
    """Parse float from environment variable with default."""
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


# ----------------------------
# Configuration classes
# ----------------------------


class AuthConfig(BaseModel):
    """
    Authentication configuration loaded from environment variables.

    - enabled: global kill-switch for API key auth
    - master_key: master API key value used for authentication
    - header_name: header used to pass key (default X-API-Key)
    """

    enabled: bool = Field(default=False)
    master_key: str | None = Field(default=None)
    header_name: str = Field(default=DEFAULT_API_KEY_HEADER_NAME)

    @classmethod
    def from_env(cls) -> "AuthConfig":
        enabled = _env_truthy(os.getenv(DEFAULT_API_KEY_ENABLED_ENV), default=False)
        master_key = os.getenv(DEFAULT_MASTER_API_KEY_ENV)
        header_name = os.getenv("API_KEY_HEADER_NAME", DEFAULT_API_KEY_HEADER_NAME).strip()
        cfg = cls(enabled=enabled, master_key=master_key, header_name=header_name)
        logger.info(
            "Auth config loaded enabled=%s header_name=%s master_key_set=%s",
            cfg.enabled,
            cfg.header_name,
            bool(cfg.master_key),
        )
        return cfg


class CookieConfig(BaseModel):
    """
    Cookie configuration loaded from environment variables.

    - cookies_file: path to a cookies.txt file to use for all downloads (optional)
    """

    cookies_file: str | None = Field(default=None)

    @classmethod
    def from_env(cls) -> "CookieConfig":
        cookies_file = os.getenv(DEFAULT_COOKIES_FILE_ENV)
        if cookies_file:
            cookies_file = cookies_file.strip()
            # Verify the file exists
            if not Path(cookies_file).is_file():
                logger.warning("COOKIES_FILE points to non-existent file=%s", cookies_file)
                cookies_file = None
            else:
                logger.info("Cookie config loaded cookies_file=%s", cookies_file)
        cfg = cls(cookies_file=cookies_file)
        return cfg


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    max_retries: int = Field(
        default_factory=lambda: _env_int(os.getenv(DEFAULT_MAX_RETRIES_ENV), default=3),
        ge=0,
        description="Maximum number of retry attempts",
    )
    backoff_base: float = Field(
        default_factory=lambda: _env_float(os.getenv(DEFAULT_RETRY_BACKOFF_ENV), default=5.0),
        ge=0,
        description="Base backoff delay in seconds",
    )
    backoff_multiplier: float = Field(
        default_factory=lambda: _env_float(
            os.getenv(DEFAULT_RETRY_BACKOFF_MULTIPLIER_ENV), default=2.0
        ),
        ge=1.0,
        description="Exponential backoff multiplier",
    )
    jitter: bool = Field(
        default_factory=lambda: _env_truthy(os.getenv(DEFAULT_RETRY_JITTER_ENV), default=True),
        description="Add random jitter to backoff to avoid thundering herd",
    )
    retryable_http_codes: list[int] = Field(
        default_factory=lambda: [429, 500, 502, 503, 504],
        description="HTTP status codes that trigger retry",
    )

    @classmethod
    def from_env(cls) -> "RetryConfig":
        """Create RetryConfig from environment variables."""
        cfg = cls()
        logger.info(
            "Retry config loaded from env max_retries=%s backoff_base=%s backoff_multiplier=%s jitter=%s",
            cfg.max_retries,
            cfg.backoff_base,
            cfg.backoff_multiplier,
            cfg.jitter,
        )
        return cfg
