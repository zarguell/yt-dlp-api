# Technology Stack

**Analysis Date:** 2025-01-17

## Languages

**Primary:**
- Python 3.13 - All application code (`main.py`, `tests/`)
- Supports Python 3.11+ per `README.md`
- Target version: py313 per `pyproject.toml`

**Secondary:**
- Shell - Build scripts (Makefile)
- YAML - CI/CD workflows (`.github/workflows/`)

## Runtime

**Environment:**
- Python 3.13 (configured in `Dockerfile`, `pyproject.toml`)
- Uvicorn ASGI server - Production web server (`main.py` line 20, `requirements.txt`)
- ThreadPoolExecutor - Async task processing for blocking operations (`main.py` lines 1403-1405)

**Package Manager:**
- pip - `requirements.txt`, `requirements-test.txt`
- No lockfiles (uses pinned versions in requirements.txt)

## Frameworks

**Core:**
- FastAPI 0.127.0 - Web framework (`requirements.txt`, `main.py` lines 22-23)
- Starlette 0.50.0 - Underlying ASGI toolkit
- Uvicorn 0.34.2 - ASGI server
- Pydantic 2.11.3 - Data validation (`requirements.txt`, `main.py` line 25)

**Testing:**
- pytest 9.0.2 - Test framework (`requirements-test.txt`, `pyproject.toml`)
- pytest-asyncio 1.3.0 - Async test support
- pytest-mock 3.14.0 - Mocking utilities
- httpx 0.28.1 - Async HTTP client for API testing

**Build/Dev:**
- Docker - Multi-stage builds with distroless runtime
- Ruff 0.9.2 - Linting and formatting (`requirements-test.txt`, `pyproject.toml`)
- mypy 1.14.1 - Type checking

## Key Dependencies

**Critical:**
- yt-dlp 2025.12.8 - Video/media downloader library (`requirements.txt`, `main.py` line 21)
- Pydantic - Request/response validation and configuration
- anyio 4.9.0 - Async compatibility layer

**Infrastructure:**
- sqlite3 (Python stdlib) - Embedded database for task persistence
- python-multipart 0.0.20 - File upload support
- typing_extensions 4.13.2 - Type hints

## Configuration

**Environment:**
- Environment variables via `os` module
- No `.env` files (configured in deployment)
- Key configs: `LOG_LEVEL`, `HOST`/`PORT`, `MAX_WORKERS`, `SERVER_OUTPUT_ROOT`, `COOKIES_DIR`
- Auth toggle: `API_KEY_AUTH_ENABLED`, `API_MASTER_KEY`
- Retry configuration: `DEFAULT_MAX_RETRIES`, `DEFAULT_RETRY_BACKOFF`

**Build:**
- `pyproject.toml` - pytest, ruff, mypy, coverage configuration
- `Dockerfile` - Multi-stage Docker build

## Platform Requirements

**Development:**
- Any platform with Python 3.11+
- ffmpeg for media processing (included in Docker image)
- make for convenience commands

**Production:**
- Docker container (multi-arch: amd64, arm64)
- Distroless runtime (Chainguard)
- Non-root user (UID 65532)

---

*Stack analysis: 2025-01-17*
*Update after major dependency changes*
