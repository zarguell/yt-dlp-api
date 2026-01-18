# Codebase Structure

**Analysis Date:** 2025-01-17

## Directory Layout

```
yt-dlp-api/
├── main.py                 # Single-file FastAPI application (1,967 lines)
├── requirements.txt        # Production dependencies
├── requirements-test.txt   # Test dependencies
├── pyproject.toml          # Project configuration (pytest, ruff, mypy, coverage)
├── Dockerfile              # Multi-stage Docker build
├── Makefile                # Developer convenience commands
├── README.md               # User documentation
├── SECURITY.md             # Security policy
├── AGENTS.md               # AI agent context
│
├── tests/                  # Test suite
│   ├── __init__.py         # Package marker
│   ├── conftest.py         # Shared pytest fixtures (199 lines)
│   ├── test_utils.py       # Utility function tests
│   ├── test_config.py      # Configuration class tests
│   ├── test_state.py       # Database/state tests
│   ├── test_retry.py       # Retry logic tests
│   ├── test_api.py         # API endpoint integration tests
│   └── test_ytdlp_contract.py  # yt-dlp contract tests
│
├── .github/
│   ├── workflows/          # CI/CD workflows
│   │   ├── docker-image.yml    # Docker build & publish
│   │   ├── test.yml            # Test automation
│   │   ├── codeql.yml          # CodeQL security scanning
│   │   ├── scorecards.yml      # Supply chain scorecards
│   │   └── dependabot.yml      # Dependency update automation
│   └── dependabot.yml      # Dependency bot configuration
│
└── downloads/              # Default output directory (created at runtime)
    └── default/            # Default output_path folder
        └── {task_id}/      # Individual task outputs
```

## Directory Purposes

**Root Directory (`/`):**
- Purpose: All application code and configuration
- Contains: Single-file application, dependencies, build configs
- Key files: `main.py` (entire application), `requirements.txt`, `Dockerfile`
- Subdirectories: `tests/`, `.github/`

**tests/:**
- Purpose: Test suite with comprehensive coverage
- Contains: `test_*.py` files organized by layer (utils, config, state, retry, api)
- Key files: `conftest.py` (shared fixtures), `test_api.py` (integration tests)
- Subdirectories: None (flat structure)

**.github/workflows/:**
- Purpose: CI/CD automation
- Contains: YAML workflow definitions for testing, building, security scanning
- Key files: `test.yml`, `docker-image.yml`, `codeql.yml`
- Subdirectories: None

## Key File Locations

**Entry Points:**
- `main.py` lines 1965-1967 - Primary entry point (`if __name__ == "__main__"`)
- `main.py` line 1542 - FastAPI application instance
- `Dockerfile` line 36 - Docker entry point

**Configuration:**
- `pyproject.toml` - pytest, ruff, mypy, coverage configuration
- `requirements.txt` - Production dependencies
- `requirements-test.txt` - Test dependencies
- Environment variables - Documented in `README.md`

**Core Logic:**
- `main.py` lines 54-525 - Configuration classes (AuthConfig, CookieConfig, RetryConfig)
- `main.py` lines 304-479 - Domain models (Task, JobType, request models)
- `main.py` lines 618-803 - State class (data persistence)
- `main.py` lines 806-1395 - YtDlpService class (business logic)
- `main.py` lines 531-616 - Retry logic functions
- `main.py` lines 1507-1967 - API routes

**Testing:**
- `tests/conftest.py` - Shared pytest fixtures
- `tests/test_utils.py` - Utility function tests
- `tests/test_config.py` - Configuration tests
- `tests/test_state.py` - Database/state tests
- `tests/test_retry.py` - Retry logic tests
- `tests/test_api.py` - API endpoint integration tests
- `tests/test_ytdlp_contract.py` - yt-dlp contract tests

**Documentation:**
- `README.md` - User-facing documentation
- `SECURITY.md` - Security policy
- `AGENTS.md` - AI agent context

## Naming Conventions

**Files:**
- `main.py` - Single-file application entry point
- `test_*.py` - Test files (pytest convention)
- `requirements*.txt` - Dependencies
- `Dockerfile` - Docker configuration (no extension)
- `Makefile` - Make configuration (no extension)
- `*.toml` - Python project configuration
- `*.md` - Documentation
- `*.yml` - YAML configuration files

**Python Classes (main.py):**
- PascalCase for all classes: `AuthConfig`, `CookieConfig`, `RetryConfig`, `Task`, `YtDlpService`
- Config suffix for configuration classes: `AuthConfig`, `CookieConfig`, `RetryConfig`
- Service suffix for service classes: `YtDlpService`

**Python Functions:**
- snake_case for functions: `resolve_task_base_dir`, `normalize_string`, `ensure_dir`
- Private functions: underscore prefix `_env_truthy`, `_load_tasks`, `_save_task`
- Route handlers: `api_*` prefix: `api_download_video`, `api_download_audio`

**Variables:**
- snake_case for variables: `task_id`, `base_dir`, `cookie_file`
- UPPER_SNAKE_CASE for constants: `SERVER_OUTPUT_ROOT`, `COOKIES_DIR`, `DEFAULT_API_KEY_HEADER_NAME`
- Environment variable constants: Pattern `*_ENV` suffix: `API_KEY_AUTH_ENABLED_ENV`

**API Endpoints:**
- POST operations: resource nouns: `/download`, `/audio`, `/subtitles`, `/cookies/upload`
- GET operations: resource queries: `/task/{id}`, `/tasks`, `/info`, `/formats`
- File operations: nested under task: `/task/{id}/files`, `/task/{id}/file`, `/task/{id}/zip`

**Directories:**
- lowercase for all directories: `tests/`, `.github/`, `downloads/`

## Where to Add New Code

**New Feature (e.g., new download type):**
- Pydantic models: `main.py` after line 479 (before State class)
- Service methods: `main.py` after line 1392 (in YtDlpService class)
- API route: `main.py` after line 1967 (new route handler)
- Tests: `tests/test_api.py` (new test class)

**New Configuration:**
- Config class: `main.py` after line 525 (after RetryConfig)
- Environment variables: Add to config class, document in README
- Tests: `tests/test_config.py` (new test class)

**New Utility Function:**
- Implementation: `main.py` after line 302 (before Configuration section)
- Tests: `tests/test_utils.py` (new test class)

**New Middleware:**
- Implementation: `main.py` after line 1568 (after request logging middleware)
- Tests: `tests/test_api.py` (test middleware behavior)

## Special Directories

**downloads/:**
- Purpose: Default output directory for downloaded files
- Source: Created at runtime if doesn't exist
- Committed: No (in `.gitignore`)
- Structure: `downloads/default/{task_id}/`

**.github/workflows/:**
- Purpose: CI/CD automation
- Source: Version-controlled in repository
- Committed: Yes
- Triggered by: Git push, pull requests

**tests/:**
- Purpose: Test suite
- Source: Version-controlled in repository
- Committed: Yes
- Run: Via `pytest` or `make test`

---

*Structure analysis: 2025-01-17*
*Update when directory structure changes*
