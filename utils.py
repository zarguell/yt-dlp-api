"""Utility functions for path resolution, validation, and string handling."""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException

if TYPE_CHECKING:
    from config import CookieConfig

logger = logging.getLogger("yt-dlp-api")

# Path configuration
SERVER_OUTPUT_ROOT_ENV = "SERVER_OUTPUT_ROOT"
DEFAULT_SERVER_OUTPUT_ROOT = "./downloads"
SERVER_OUTPUT_ROOT = Path(os.getenv(SERVER_OUTPUT_ROOT_ENV, DEFAULT_SERVER_OUTPUT_ROOT))

COOKIES_DIR_ENV = "COOKIES_DIR"
DEFAULT_COOKIES_DIR = "./cookies"
COOKIES_DIR = Path(os.getenv(COOKIES_DIR_ENV, DEFAULT_COOKIES_DIR))
COOKIES_DIR.mkdir(parents=True, exist_ok=True)

# Global variable to be set by main.py
cookie_config: "CookieConfig | None" = None


def _is_safe_subdir_name(value: str, *, max_length: int = 80) -> bool:
    """Validate an API-provided folder label (single subdirectory)."""
    if not value:
        return False
    if len(value) > max_length:
        return False
    if "/" in value or "\\" in value:
        return False
    if value in {".", ".."}:
        return False
    if ".." in value:
        return False

    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return all(ch in allowed for ch in value)


def resolve_task_base_dir(client_output_path: str) -> Path:
    """Convert client 'output_path' into a server-controlled base directory."""
    label = client_output_path.strip()
    if label in {"", ".", "./"}:
        label = "default"

    if not _is_safe_subdir_name(label):
        logger.warning("Rejected unsafe output_path label=%r", label)
        raise HTTPException(
            status_code=400,
            detail="Invalid output_path. Provide a simple folder name (no slashes or '..').",
        )

    root = SERVER_OUTPUT_ROOT.resolve(strict=False)
    base = (root / label).resolve(strict=False)

    if not base.is_relative_to(root):
        logger.warning(
            "Rejected output_path outside root label=%r base=%s root=%s", label, base, root
        )
        raise HTTPException(status_code=400, detail="Invalid output_path (outside server root).")

    base.mkdir(parents=True, exist_ok=True)
    logger.debug("Resolved base output dir label=%r base=%s", label, base)
    return base


def resolve_cookie_file(request_cookie_file: str | None) -> str | None:
    """
    Resolve the cookie file path from request and environment configuration.

    Priority:
    1. Request-specific cookie_file parameter (relative to COOKIES_DIR)
    2. Global COOKIES_FILE environment variable (absolute or relative to COOKIES_DIR)

    All paths are validated to ensure they remain within the COOKIES_DIR to prevent
    path traversal attacks.

    Returns the absolute path to the cookie file, or None if no cookies are configured.
    """
    if cookie_config is None:
        return None

    cookie_file = request_cookie_file or cookie_config.cookies_file

    if not cookie_file:
        return None

    cookie_path = Path(cookie_file)

    # If it's a relative path, treat it as relative to COOKIES_DIR
    if not cookie_path.is_absolute():
        cookie_path = (COOKIES_DIR / cookie_file).resolve(strict=False)
    else:
        # For absolute paths, just resolve it (we'll validate containment next)
        cookie_path = cookie_path.resolve(strict=False)

    # Validate the path doesn't escape COOKIES_DIR
    if not cookie_path.is_relative_to(COOKIES_DIR.resolve(strict=False)):
        logger.warning("Rejected cookie path outside COOKIES_DIR path=%s", cookie_path)
        raise HTTPException(
            status_code=400,
            detail="Cookie file path must be within the cookies directory",
        )

    # Verify the file exists
    if not cookie_path.is_file():
        logger.warning("Cookie file not found path=%s", cookie_path)
        return None

    return str(cookie_path)


def normalize_string(value: str, max_length: int = 200) -> str:
    """Trim whitespace, replace unsafe filename characters with underscores, and cap length."""
    value = value.strip()
    unsafe_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
    for ch in unsafe_chars:
        value = value.replace(ch, "_")
    if len(value) > max_length:
        value = value[: max_length - 3] + "..."
    return value


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
