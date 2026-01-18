# Coding Conventions

**Analysis Date:** 2025-01-17

## Naming Patterns

**Files:**
- `main.py` - Single-file application entry point
- `test_*.py` - Test files (pytest convention)
- `requirements*.txt` - Dependencies (pinned versions)
- `Dockerfile`, `Makefile` - Build configuration (no extension)
- `*.toml` - Python project configuration
- `*.md` - Documentation files

**Functions:**
- snake_case for all functions: `resolve_task_base_dir`, `normalize_string`, `ensure_dir`
- Private functions: underscore prefix `_env_truthy`, `_load_tasks`, `_save_task`
- Route handlers: `api_*` prefix: `api_download_video`, `api_download_audio`, `api_get_task`
- Async functions: Explicit `async def` keyword with snake_case names

**Variables:**
- snake_case for variables: `task_id`, `base_dir`, `cookie_file`, `max_retries`
- UPPER_SNAKE_CASE for constants: `SERVER_OUTPUT_ROOT`, `COOKIES_DIR`, `DEFAULT_MAX_RETRIES_ENV`
- Environment variable constants: Pattern `*_ENV` suffix: `API_KEY_AUTH_ENABLED_ENV`

**Types:**
- PascalCase for classes: `AuthConfig`, `CookieConfig`, `RetryConfig`, `Task`, `YtDlpService`
- No prefix/suffix for interfaces (not used in this codebase)
- PascalCase for enums: `JobType`, `EnglishMode`, `SubtitlePreference`, `SubtitleFormat`
- UPPER_CASE for enum values: `JobType.VIDEO`, `EnglishMode.BEST_ONE`

## Code Style

**Formatting:**
- Tool: Ruff (configured in `pyproject.toml` lines 52-72)
- Line length: 100 characters max
- Quotes: Double quotes for strings (`"example"`)
- Semicolons: Not used (Python style)
- Indentation: 4 spaces (Python standard)

**Linting:**
- Tool: Ruff (linting + formatting combined)
- Config: `pyproject.toml` lines 52-72
- Enabled rules: E, W, F, I, B, C4, UP (pycodestyle, pyflakes, isort, flake8-bugbear, comprehensions, pyupgrade)
- Type checking: mypy (configured in `pyproject.toml` lines 74-79)
- Run commands: `make lint` (ruff + mypy), `make format` (ruff format)

## Import Organization

**Order:**
1. Standard library imports (`asyncio`, `logging`, `os`, `sqlite3`, `uuid`, `json`, `time`)
2. Third-party imports (`uvicorn`, `yt_dlp`, `fastapi`, `pydantic`, `starlette`)
3. No internal imports (single-file architecture)

**Grouping:**
- Blank line between groups
- Alphabetical within each group (mostly)
- Example from `main.py` lines 1-28

**Path Aliases:**
- Not used (single-file architecture)

## Error Handling

**Patterns:**
- Strategy: Throw exceptions, catch at boundaries (route handlers, task processor)
- Custom errors: No custom error classes (use `HTTPException` from FastAPI)
- Async: Use try/catch, no `.catch()` chains

**Error Types:**
- When to throw: Invalid input, missing dependencies, path traversal attempts, API key auth failures
- When to return: Task failures update status to "failed" (don't throw)
- Logging: Log error with context before raising: `logger.exception()` with error details

**Example from `main.py`:**
```python
# Route handler raises HTTPException for client errors
if not auth_config.enabled:
    raise HTTPException(status_code=401, detail="API key authentication required")

# Task processor catches and updates task status
try:
    result = await run_in_threadpool(
        retry_with_backoff, service.download_video, retry_config, url, options
    )
except Exception as e:
    await state.update_task(task_id, status="failed", error=str(e))
    logger.exception(f"Task {task_id} failed")
```

## Logging

**Framework:**
- Tool: Python stdlib logging (`main.py` lines 36-50)
- Levels: debug, info, warning, error, exception
- Structured logging with request ID correlation

**Patterns:**
- Format: Text logging with request_id context
- When: Log state transitions, external calls, errors
- Where: Log at service boundaries, in task processor, on errors
- Request correlation: `_request_id_ctx` context variable for tracking

**Example from `main.py`:**
```python
logger.info(f"Created task {task_id} for {url}")
logger.warning(f"Retry {attempt}/{max_retries} after {backoff:.2f}s delay")
logger.exception(f"Task {task_id} failed")
```

## Comments

**When to Comment:**
- Explain why, not what
- Document business logic: "Deduplication prevents duplicate downloads for identical requests"
- Explain non-obvious security checks: "Prevent path traversal attacks by constraining to SERVER_OUTPUT_ROOT"
- Avoid obvious comments: "Set variable to 0" (not used)

**Docstrings:**
- Usage: Google-style docstrings for classes and complex functions
- Format: Triple quotes, one-line for simple functions, multi-line with bullet points for complex classes
- Example from `main.py` lines 103-110:
```python
class AuthConfig(BaseModel):
    """
    Authentication configuration loaded from environment variables.

    - enabled: global kill-switch for API key auth
    - master_key: master API key value used for authentication
    - header_name: header used to pass key (default X-API-Key)
    """
```

**TODO Comments:**
- Pattern: Not extensively used (codebase is mature)
- Tracking: Via git blame (no issue references in TODOs)

**Section Dividers:**
- Pattern: `# ----------------------------` followed by section name
- Example from `main.py` lines 29, 53, 181, 240, 304, 531, 618, 806, 1398, 1507

## Function Design

**Size:**
- Keep under 50 lines preferred (but exceptions exist for complex service methods)
- Extract helpers for complex logic (e.g., `_select_best_subtitle_language`)
- One level of abstraction per function

**Parameters:**
- Max 3-4 parameters preferred
- Use object for more parameters: `def download_subtitles_v2(request: SubtitlesV2Request)`
- Destructure objects in parameter list: `def resolve_task_base_dir(*, base_output_path: str) -> Path`

**Return Values:**
- Explicit return statements
- Return early for guard clauses (common in validation functions)
- Use `None` for missing values (typed as `str | None`)

## Module Design

**Exports:**
- Single-file architecture (no module exports)
- Global instances: `state`, `service`, `auth_config`, `cookie_config`, `app`
- No barrel files (not applicable)

**Class Design:**
- Configuration classes: Pydantic BaseModel with `from_env()` classmethod
- Service classes: Static methods only (no instantiation state)
- Data classes: Pydantic BaseModel for request/response models

**Type Hints:**
- Modern Python 3.10+ syntax using pipe operator: `str | None`, `list[str]`
- Required for all function parameters and return values
- Configured in mypy (`pyproject.toml` lines 74-79)

---

*Convention analysis: 2025-01-17*
*Update when patterns change*
