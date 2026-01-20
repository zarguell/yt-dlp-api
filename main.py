import asyncio
import contextvars
import logging
import os
import random
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TypeVar, cast
from zipfile import ZIP_DEFLATED, ZipFile

import uvicorn
import yt_dlp
from fastapi import Depends, FastAPI, File, HTTPException, Query, Security, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from starlette.background import BackgroundTask
from starlette.requests import Request

import utils
from config import (
    DEFAULT_MASTER_API_KEY_ENV,
    AuthConfig,
    CookieConfig,
    RetryConfig,
)
from models import (
    AudioRequest,
    DownloadRequest,
    EnglishMode,
    JobType,
    SubtitleFormat,
    SubtitlePreference,
    SubtitlesRequest,
    SubtitlesV2Request,
    Task,
)
from service import YtDlpService
from state import State
from utils import (
    COOKIES_DIR,
    ensure_dir,
    normalize_string,
    resolve_cookie_file,
    resolve_task_base_dir,
)

# ----------------------------
# Logging setup
# ----------------------------

_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Attach request_id to all log records for correlation."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get()
        return True


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("yt-dlp-api")
logger.addFilter(RequestIdFilter())


# ----------------------------
# Auth settings
# ----------------------------

# ----------------------------
# Auth settings
# ----------------------------

auth_config = AuthConfig.from_env()
cookie_config = CookieConfig.from_env()
api_key_header = APIKeyHeader(name=auth_config.header_name, auto_error=False)

# Initialize utils with config
utils.cookie_config = cookie_config

# default_retry_config will be initialized after RetryConfig is defined


async def require_api_key(api_key: str | None = Security(api_key_header)) -> None:
    """Global API key dependency."""
    if not auth_config.enabled:
        return

    if not auth_config.master_key:
        logger.error(
            "API key auth enabled but master key env var missing env=%s", DEFAULT_MASTER_API_KEY_ENV
        )
        raise HTTPException(
            status_code=500,
            detail=f"API key auth is enabled but {DEFAULT_MASTER_API_KEY_ENV} is not set.",
        )

    if not api_key or api_key != auth_config.master_key:
        logger.warning("Authentication failed (invalid/missing API key)")
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ----------------------------
# Domain models
# ----------------------------


# Initialize global retry config from environment
default_retry_config = RetryConfig.from_env()


T = TypeVar("T")


# ----------------------------
# Retry utilities
# ----------------------------


def is_retryable_error(error: Exception, retry_config: RetryConfig) -> bool:
    """Check if an error is retryable based on configuration."""
    error_str = str(error).lower()

    # Check for HTTP error codes
    for code in retry_config.retryable_http_codes:
        if f"http error {code}" in error_str or f"httperror: {code}" in error_str:
            return True

    # Check for common retryable error patterns
    retryable_patterns = [
        "too many requests",
        "rate limit",
        "temporary failure",
        "connection reset",
        "connection refused",
        "timed out",
        "timeout",
        "server error",
    ]

    return any(pattern in error_str for pattern in retryable_patterns)


def calculate_backoff(attempt: int, retry_config: RetryConfig) -> float:
    """Calculate exponential backoff delay with optional jitter."""
    base_delay = retry_config.backoff_base
    multiplier = retry_config.backoff_multiplier

    # Exponential backoff: base * (multiplier ^ attempt)
    delay = base_delay * (multiplier**attempt)

    # Add jitter if enabled (±25% random variation)
    if retry_config.jitter:
        jitter_range = delay * 0.25
        delay = delay + random.uniform(-jitter_range, jitter_range)

    return max(0, delay)


def retry_with_backoff(
    func: Callable[..., T],
    retry_config: RetryConfig,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute a function with retry logic and exponential backoff."""
    last_error: Exception | None = None

    for attempt in range(retry_config.max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            is_last_attempt = attempt >= retry_config.max_retries

            if is_last_attempt or not is_retryable_error(e, retry_config):
                logger.warning(
                    "Non-retryable error or max retries exceeded attempt=%d/%d error=%s",
                    attempt + 1,
                    retry_config.max_retries + 1,
                    str(e)[:200],
                )
                raise

            # Calculate backoff and wait
            backoff = calculate_backoff(attempt, retry_config)
            logger.info(
                "Retryable error encountered, retrying after backoff attempt=%d/%d backoff_seconds=%.1f error=%s",
                attempt + 1,
                retry_config.max_retries + 1,
                backoff,
                str(e)[:200],
            )
            time.sleep(backoff)

    # This should never be reached, but mypy needs it
    if last_error:
        raise last_error
    raise RuntimeError("Retry logic failed without raising an exception")


# ----------------------------
# Persistence (SQLite)
# ----------------------------


state = State(logger=logger)


# ----------------------------
# yt-dlp service
service = YtDlpService()


# ----------------------------
# Async execution
# ----------------------------

# Reuse one executor rather than creating a new pool per call. [web:2]
_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.getenv("MAX_WORKERS", "4")), thread_name_prefix="yt-dlp-worker"
)


async def run_in_threadpool(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_EXECUTOR, lambda: func(*args, **kwargs))


async def process_task(task_id: str, job_type: JobType, payload: dict[str, Any]) -> None:
    logger.info("Process task start task_id=%s job_type=%s", task_id, job_type.value)
    start = time.monotonic()
    try:
        state.update_task(task_id, "running")

        # Build retry config: start with global defaults, override with request-specific values
        retry_config = RetryConfig(
            max_retries=default_retry_config.max_retries,
            backoff_base=default_retry_config.backoff_base,
            backoff_multiplier=default_retry_config.backoff_multiplier,
            jitter=default_retry_config.jitter,
        )

        # Override with request-specific values if provided
        if "max_retries" in payload:
            retry_config.max_retries = payload.pop("max_retries")
        if "retry_backoff" in payload:
            retry_config.backoff_base = payload.pop("retry_backoff")

        if job_type == JobType.video:
            # Apply retry wrapper for video downloads
            result = await run_in_threadpool(
                retry_with_backoff,
                service.download_video,
                retry_config,
                **payload,
            )
        elif job_type == JobType.audio:
            # Apply retry wrapper for audio downloads
            result = await run_in_threadpool(
                retry_with_backoff,
                service.download_audio,
                retry_config,
                **payload,
            )
        elif job_type == JobType.subtitles:
            # For subtitles, handle partial success and retry separately
            result = await run_in_threadpool(
                retry_with_backoff,
                service.download_subtitles,
                retry_config,
                **payload,
            )

            # Check if we got partial success
            if isinstance(result, dict) and result.get("partial"):
                logger.info(
                    "Partial subtitle download success task_id=%s downloaded=%d failed=%d",
                    task_id,
                    len(result.get("downloaded", [])),
                    len(result.get("failed", [])),
                )
                state.update_task(task_id, "partial", result=result)
                return

            # Check if completely failed but retryable
            if (
                isinstance(result, dict)
                and not result.get("success")
                and result.get("is_retryable")
            ):
                # The retry logic should have handled this, but if we still failed:
                logger.warning("Subtitle download failed after retries task_id=%s", task_id)
                state.update_task(task_id, "failed", error=result.get("error", "Unknown error"))
                return
        elif job_type == JobType.subtitles_v2:
            # V2 subtitles: simplified retry handling (no partial mode)
            result = await run_in_threadpool(
                retry_with_backoff,
                service.download_subtitles_v2,
                retry_config,
                **payload,
            )

            # V2 doesn't use partial mode - either success or fail
            if isinstance(result, dict) and not result.get("success"):
                logger.warning("Subtitle v2 download failed task_id=%s", task_id)
                state.update_task(task_id, "failed", error=result.get("error", "Unknown error"))
                return
        else:
            raise ValueError(f"Unsupported job type: {job_type}")

        state.update_task(task_id, "completed", result=result)
        logger.info(
            "Process task completed task_id=%s elapsed_ms=%d",
            task_id,
            int((time.monotonic() - start) * 1000),
        )
    except Exception as exc:
        logger.exception("Process task failed task_id=%s error=%s", task_id, exc)
        state.update_task(task_id, "failed", error=str(exc))


# ----------------------------
# File endpoints (generic)
# ----------------------------


def _require_completed_task(task_id: str) -> Task:
    """Get a task that has completed (either fully or partially)."""
    task = state.get_task(task_id)
    if not task:
        logger.info("Task not found task_id=%s", task_id)
        raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
    if task.status not in ("completed", "partial"):
        logger.info("Task not completed task_id=%s status=%s", task_id, task.status)
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed yet. Current status: {task.status}",
        )
    return task


def list_task_files(task: Task) -> list[Path]:
    task_dir = Path(task.task_output_path)
    if not task_dir.exists():
        logger.warning("Task output directory missing task_id=%s dir=%s", task.id, task_dir)
        return []
    files = [p for p in task_dir.iterdir() if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    logger.debug("Listed task files task_id=%s count=%d", task.id, len(files))
    return files


# ----------------------------
# FastAPI
# ----------------------------

app = FastAPI(
    title="yt-dlp API",
    description="API for downloading videos, audio, and subtitles using yt-dlp",
    dependencies=[Depends(require_api_key)],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = _request_id_ctx.set(request_id)
    start = time.monotonic()
    try:
        logger.info("Request start method=%s path=%s", request.method, request.url.path)
        response = await call_next(request)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "Request end method=%s path=%s status=%d elapsed_ms=%d",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        _request_id_ctx.reset(token)


@app.post("/download", response_class=JSONResponse)
async def api_download_video(request: DownloadRequest):
    base_dir = resolve_task_base_dir(request.output_path)
    cookie_file = resolve_cookie_file(request.cookie_file)

    existing = next(
        (
            t
            for t in state.tasks.values()
            if t.job_type == JobType.video
            and t.url == request.url
            and t.base_output_path == str(base_dir)
            and t.format == request.format
        ),
        None,
    )
    if existing:
        logger.info(
            "Deduped video task existing_task_id=%s url=%s base=%s fmt=%s",
            existing.id,
            request.url,
            base_dir,
            request.format,
        )
        return {"status": "success", "task_id": existing.id}

    task_id = state.add_task(JobType.video, request.url, request.output_path, request.format)
    task = state.get_task(task_id)
    assert task is not None

    logger.info("Queue video task task_id=%s cookie_file=%s", task_id, cookie_file)
    asyncio.create_task(
        process_task(
            task_id=task_id,
            job_type=JobType.video,
            payload={
                "url": request.url,
                "output_path": task.task_output_path,
                "fmt": request.format,
                "quiet": request.quiet,
                "cookie_file": cookie_file,
            },
        )
    )
    return {"status": "success", "task_id": task_id}


@app.post("/audio", response_class=JSONResponse)
async def api_download_audio(request: AudioRequest):
    fmt_key = f"audio:{request.audio_format}:q={request.audio_quality}"
    base_dir = resolve_task_base_dir(request.output_path)
    cookie_file = resolve_cookie_file(request.cookie_file)

    existing = next(
        (
            t
            for t in state.tasks.values()
            if t.job_type == JobType.audio
            and t.url == request.url
            and t.base_output_path == str(base_dir)
            and t.format == fmt_key
        ),
        None,
    )
    if existing:
        logger.info(
            "Deduped audio task existing_task_id=%s url=%s base=%s fmt=%s",
            existing.id,
            request.url,
            base_dir,
            fmt_key,
        )
        return {"status": "success", "task_id": existing.id}

    task_id = state.add_task(JobType.audio, request.url, request.output_path, fmt_key)
    task = state.get_task(task_id)
    assert task is not None

    logger.info("Queue audio task task_id=%s cookie_file=%s", task_id, cookie_file)
    asyncio.create_task(
        process_task(
            task_id=task_id,
            job_type=JobType.audio,
            payload={
                "url": request.url,
                "output_path": task.task_output_path,
                "audio_format": request.audio_format,
                "audio_quality": request.audio_quality,
                "quiet": request.quiet,
                "cookie_file": cookie_file,
            },
        )
    )
    return {"status": "success", "task_id": task_id}


@app.post("/subtitles", response_class=JSONResponse)
async def api_download_subtitles(request: SubtitlesRequest):
    fmt_key = (
        f"subs:{','.join(request.languages)}:"
        f"manual={request.write_manual}:auto={request.write_automatic}:conv={request.convert_to}"
    )
    base_dir = resolve_task_base_dir(request.output_path)
    cookie_file = resolve_cookie_file(request.cookie_file)

    existing = next(
        (
            t
            for t in state.tasks.values()
            if t.job_type == JobType.subtitles
            and t.url == request.url
            and t.base_output_path == str(base_dir)
            and t.format == fmt_key
        ),
        None,
    )
    if existing:
        logger.info(
            "Deduped subtitles task existing_task_id=%s url=%s base=%s fmt=%s",
            existing.id,
            request.url,
            base_dir,
            fmt_key,
        )
        return {"status": "success", "task_id": existing.id}

    task_id = state.add_task(JobType.subtitles, request.url, request.output_path, fmt_key)
    task = state.get_task(task_id)
    assert task is not None

    logger.info("Queue subtitles task task_id=%s cookie_file=%s", task_id, cookie_file)
    asyncio.create_task(
        process_task(
            task_id=task_id,
            job_type=JobType.subtitles,
            payload={
                "url": request.url,
                "output_path": task.task_output_path,
                "languages": request.languages,
                "write_manual": request.write_manual,
                "write_automatic": request.write_automatic,
                "convert_to": request.convert_to,
                "quiet": request.quiet,
                "cookie_file": cookie_file,
            },
        )
    )
    return {"status": "success", "task_id": task_id}


@app.post("/v2/subtitles", response_class=JSONResponse)
async def api_download_subtitles_v2(request: SubtitlesV2Request):
    """Enhanced subtitles endpoint with policy-based language selection.

    Features:
    - Automatic English subtitle selection (best_one, all_english, or explicit)
    - Manual vs automatic subtitle preference
    - Format normalization (SRT, VTT, or both)
    - Intelligent language ranking

    Returns immediately with a task_id for async processing.
    """
    # Create dedupe key that includes all policy fields
    fmt_key = (
        f"v2subs:{request.english_mode.value}:"
        f"prefer={request.prefer.value}:"
        f"fmt={request.formats.value}:"
        f"rank={','.join(request.english_rank)}:"
        f"langs={','.join(request.languages)}"
    )
    base_dir = resolve_task_base_dir(request.output_path)
    cookie_file = resolve_cookie_file(request.cookie_file)

    existing = next(
        (
            t
            for t in state.tasks.values()
            if t.job_type == JobType.subtitles_v2
            and t.url == request.url
            and t.base_output_path == str(base_dir)
            and t.format == fmt_key
        ),
        None,
    )
    if existing:
        logger.info(
            "Deduped subtitles v2 task existing_task_id=%s url=%s base=%s fmt=%s",
            existing.id,
            request.url,
            base_dir,
            fmt_key,
        )
        return {"status": "success", "task_id": existing.id}

    task_id = state.add_task(JobType.subtitles_v2, request.url, request.output_path, fmt_key)
    task = state.get_task(task_id)
    assert task is not None

    logger.info("Queue subtitles v2 task task_id=%s cookie_file=%s", task_id, cookie_file)
    asyncio.create_task(
        process_task(
            task_id=task_id,
            job_type=JobType.subtitles_v2,
            payload={
                "url": request.url,
                "output_path": task.task_output_path,
                "english_mode": request.english_mode,
                "languages": request.languages,
                "prefer": request.prefer,
                "formats": request.formats,
                "english_rank": request.english_rank,
                "quiet": request.quiet,
                "cookie_file": cookie_file,
            },
        )
    )
    return {"status": "success", "task_id": task_id}


@app.get("/task/{task_id}", response_class=JSONResponse)
async def get_task_status(task_id: str):
    task = state.get_task(task_id)
    if not task:
        logger.info("Task not found task_id=%s", task_id)
        raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")

    data: dict[str, Any] = {
        "id": task.id,
        "job_type": task.job_type,
        "url": task.url,
        "status": task.status,
        "base_output_path": task.base_output_path,
        "task_output_path": task.task_output_path,
    }
    # Include result for both completed and partial tasks
    if task.status in ("completed", "partial") and task.result:
        data["result"] = task.result
    if task.status == "failed" and task.error:
        data["error"] = task.error

    return {"status": "success", "data": data}


@app.get("/tasks", response_class=JSONResponse)
async def list_all_tasks():
    logger.debug("List tasks count=%d", len(state.tasks))
    return {"status": "success", "data": state.list_tasks()}


@app.get("/info", response_class=JSONResponse)
async def api_get_video_info(url: str = Query(..., description="Video URL")):
    try:
        logger.info("Info request url=%s", url)
        return {"status": "success", "data": service.get_info(url=url, quiet=True)}
    except Exception as exc:
        logger.exception("Info request failed url=%s error=%s", url, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/formats", response_class=JSONResponse)
async def api_list_formats(url: str = Query(..., description="Video URL")):
    try:
        logger.info("Formats request url=%s", url)
        return {"status": "success", "data": service.list_formats(url)}
    except Exception as exc:
        logger.exception("Formats request failed url=%s error=%s", url, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/cookies/upload", response_class=JSONResponse)
async def upload_cookies_file(
    file: UploadFile = File(..., description="cookies.txt file"),
):
    """
    Upload a cookies.txt file for use in downloads.

    The file is stored in the cookies directory and can be referenced by
    the returned filename in download requests via the cookie_file parameter.
    """
    if not file.filename:
        logger.warning("Cookies upload attempt without filename")
        raise HTTPException(status_code=400, detail="No filename provided")

    # Generate a safe filename
    safe_filename = f"{uuid.uuid4()}_{normalize_string(file.filename, max_length=50)}"
    cookie_path = COOKIES_DIR / safe_filename

    try:
        logger.info("Saving cookies file filename=%s path=%s", file.filename, cookie_path)
        content = await file.read()

        # Validate it's a text file
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as err:
            logger.warning("Cookies file is not valid UTF-8 filename=%s", file.filename)
            raise HTTPException(
                status_code=400, detail="cookies.txt must be a valid text file"
            ) from err

        # Write the file
        with open(cookie_path, "wb") as f:
            f.write(content)

        logger.info(
            "Cookies file saved successfully path=%s size_bytes=%d", cookie_path, len(content)
        )
        return {
            "status": "success",
            "data": {
                "cookie_file": safe_filename,
                "path": str(cookie_path),
                "size_bytes": len(content),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to save cookies file filename=%s error=%s", file.filename, exc)
        raise HTTPException(status_code=500, detail=f"Failed to save cookies file: {exc}") from exc


@app.get("/task/{task_id}/files", response_class=JSONResponse)
async def api_task_files(task_id: str):
    task = _require_completed_task(task_id)
    files = list_task_files(task)
    return {
        "status": "success",
        "data": [{"name": f.name, "size_bytes": f.stat().st_size} for f in files],
    }


@app.get("/task/{task_id}/file", response_class=FileResponse)
async def api_task_file(
    task_id: str,
    name: str = Query(..., description="Exact filename from /task/{task_id}/files"),
):
    task = _require_completed_task(task_id)
    allow = {p.name: p for p in list_task_files(task)}
    if name not in allow:
        logger.info("File not found task_id=%s name=%s", task_id, name)
        raise HTTPException(status_code=404, detail="File not found for this task")

    p = allow[name]
    logger.info("Serving file task_id=%s name=%s path=%s", task_id, name, p)
    return FileResponse(path=str(p), filename=p.name, media_type="application/octet-stream")


@app.get("/task/{task_id}/zip", response_class=FileResponse)
async def api_task_zip(task_id: str):
    task = _require_completed_task(task_id)
    files = list_task_files(task)
    if not files:
        logger.info("No files to zip task_id=%s", task_id)
        raise HTTPException(status_code=404, detail="No files found to zip")

    tmp = NamedTemporaryFile(delete=False, suffix=".zip")
    tmp_path = Path(tmp.name)
    tmp.close()

    def cleanup() -> None:
        try:
            tmp_path.unlink(missing_ok=True)
            logger.debug("Cleaned up temp zip path=%s", tmp_path)
        except Exception:
            logger.exception("Failed to cleanup temp zip path=%s", tmp_path)

    try:
        logger.info(
            "Creating zip task_id=%s tmp_path=%s file_count=%d", task_id, tmp_path, len(files)
        )
        with ZipFile(tmp_path, "w", compression=ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, arcname=f.name)

        return FileResponse(
            path=str(tmp_path),
            filename=f"task-{task_id}.zip",
            media_type="application/zip",
            background=BackgroundTask(cleanup),
        )
    except Exception as exc:
        cleanup()
        logger.exception("Failed to create zip task_id=%s error=%s", task_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to create zip: {exc}") from exc


def start_api() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting uvicorn host=%s port=%s", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    logger.info("Starting yt-dlp API server...")
    start_api()
