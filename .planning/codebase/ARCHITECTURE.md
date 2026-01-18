# Architecture

**Analysis Date:** 2025-01-17

## Pattern Overview

**Overall:** Single-File Monolithic Layered Architecture with async task processing

**Key Characteristics:**
- All code in one file (`main.py` - 1,967 lines)
- Layered architecture with clear separation of concerns
- Async task processing with background execution
- Stateful design with SQLite persistence
- Thread pool-based blocking operation handling

## Layers

**Configuration Layer:**
- Purpose: Environment-based configuration
- Contains: `AuthConfig`, `CookieConfig`, `RetryConfig` classes
- Location: `main.py` lines 54-525
- Depends on: Python stdlib (`os`, `pydantic`)
- Used by: API layer, service layer

**Domain Models Layer:**
- Purpose: Define API contracts and domain entities
- Contains: Pydantic models (`Task`, `JobType`, request models)
- Location: `main.py` lines 304-479
- Depends on: `pydantic`, `typing`, `enum`
- Used by: All layers (data validation)

**Data Layer:**
- Purpose: Task persistence and retrieval
- Contains: `State` class with CRUD operations
- Location: `main.py` lines 618-803
- Depends on: `sqlite3`, `json`, `pathlib`, `uuid`
- Used by: API layer, service layer

**Service Layer:**
- Purpose: Business logic wrapper around yt-dlp
- Contains: `YtDlpService` class with static methods
- Location: `main.py` lines 806-1395
- Depends on: `yt_dlp`, time, retry logic
- Used by: Async execution layer

**Async Execution Layer:**
- Purpose: Schedule and process background tasks
- Contains: Thread pool executor, `process_task()` function
- Location: `main.py` lines 1398-1505
- Depends on: `asyncio`, `concurrent.futures`, service layer
- Used by: API layer (for spawning tasks)

**API Layer:**
- Purpose: FastAPI routes and HTTP handling
- Contains: Route handlers, middleware, authentication
- Location: `main.py` lines 1507-1967
- Depends on: All layers
- Used by: External clients

**Utility Layer:**
- Purpose: Security and helper functions
- Contains: Path validation, string sanitization, directory creation
- Location: `main.py` lines 239-302, 1508-1536
- Depends on: `pathlib`, `logging`
- Used by: API layer, service layer

**Retry Logic Layer:**
- Purpose: Error handling with exponential backoff
- Contains: `retry_with_backoff()`, `is_retryable_error()`
- Location: `main.py` lines 531-616
- Depends on: `random`, `time`
- Used by: Async execution layer

## Data Flow

**HTTP Request Lifecycle:**

1. Client sends HTTP request (e.g., `POST /download`)
2. FastAPI middleware adds request_id context (`main.py` lines 1549-1568)
3. Optional API key validation via `require_api_key()` dependency (`main.py` lines 162-178)
4. Route handler validates request (Pydantic models)
5. Deduplication check against existing tasks (`main.py` lines 1600-1604)
6. Create new task via `state.add_task()` (`main.py` line 1606)
7. Spawn background task via `asyncio.create_task(process_task())` (`main.py` line 1612)
8. Return immediate response with `task_id`

**Background Task Processing:**

1. `process_task()` updates status to "running" (`main.py` lines 1424-1425)
2. Build retry_config from defaults + request overrides (`main.py` lines 1427-1435)
3. Run in thread pool (yt-dlp is blocking) (`main.py` lines 1437-1461)
4. `retry_with_backoff()` wraps service method call (`main.py` lines 1443-1459)
5. `YtDlpService.download_video()` → yt-dlp library (`main.py` lines 1190-1392)
6. Update status to "completed" or "failed" (`main.py` lines 1463-1471)
7. Persist result to SQLite via `state.update_task()` (`main.py` line 1474)

**Client Polling:**

1. Client polls `GET /task/{id}` endpoint (`main.py` lines 1790-1811)
2. `state.get_task()` retrieves from cache or database (`main.py` lines 678-696)
3. Return task status, result, or error

**File Download:**

1. Client requests file via `GET /task/{id}/file` or `GET /task/{id}/zip` (`main.py` lines 1903-1955)
2. Server validates path and returns `FileResponse`
3. For ZIP: creates temporary archive, serves with cleanup

**State Management:**
- File-based: SQLite database (`tasks.db`) for persistence
- In-memory cache: `State.tasks` dict for fast lookups
- No server-side state (fire-and-forget pattern)
- Each task independent with status tracking

## Key Abstractions

**Service:**
- Purpose: Encapsulate business logic for media operations
- Examples: `YtDlpService` class (`main.py` lines 811-1392)
- Pattern: Static methods (no instantiation required), pure functions with dependencies passed as parameters

**Repository:**
- Purpose: Encapsulate data persistence
- Examples: `State` class (`main.py` lines 623-801)
- Pattern: Singleton instance with in-memory cache + SQLite persistence

**Configuration Object:**
- Purpose: Type-safe environment variable parsing
- Examples: `AuthConfig`, `CookieConfig`, `RetryConfig` (`main.py` lines 103-525)
- Pattern: Pydantic BaseModel with `from_env()` classmethod

**Background Task:**
- Purpose: Long-running operation executed asynchronously
- Examples: `process_task()` function (`main.py` lines 1413-1504)
- Pattern: Fire-and-forget with status progression (pending → running → completed/failed/partial)

**Retry Wrapper:**
- Purpose: Execute function with exponential backoff on retryable errors
- Examples: `retry_with_backoff()` (`main.py` lines 576-615)
- Pattern: Higher-order function with configurable retry logic

**Path Validator:**
- Purpose: Prevent path traversal attacks
- Examples: `resolve_task_base_dir()`, `resolve_cookie_file()` (`main.py` lines 213-285)
- Pattern: Validate paths against allowed roots using `Path.is_relative_to()`

## Entry Points

**Primary Entry Point:**
- Location: `main.py` lines 1965-1967
- Triggers: Direct execution (`python main.py`) or container startup
- Responsibilities: Initialize logging, call `start_api()`

**Server Startup:**
- Location: `main.py` lines 1958-1962 (`start_api()`)
- Triggers: Called when `__name__ == "__main__"`
- Responsibilities: Configure uvicorn, read HOST/PORT from env, start server

**FastAPI Application:**
- Location: `main.py` line 1542 (`app = FastAPI(...)`)
- Triggers: Imported by uvicorn
- Responsibilities: Route registration, middleware, dependency injection

**Test Entry Points:**
- Location: `tests/conftest.py` (pytest fixtures)
- Triggers: `pytest` command
- Responsibilities: Set up test environment, mock services, create async client

## Error Handling

**Strategy:** Throw exceptions, catch at boundaries, log and update task status

**Patterns:**
- Services raise exceptions on failure (yt-dlp errors, validation errors)
- `process_task()` catches exceptions, updates task status to "failed" (`main.py` lines 1467-1474)
- API endpoints raise `HTTPException` for client errors (`main.py` lines 162-178)
- Logging with context: `logger.exception()` for errors, `logger.warning()` for recoverable issues

**Error Types:**
- HTTP 400: Invalid request (validation errors)
- HTTP 401: Missing/invalid API key
- HTTP 404: Task not found
- HTTP 500: Internal errors (yt-dlp failures)

## Cross-Cutting Concerns

**Logging:**
- Python stdlib logging with request ID correlation (`main.py` lines 36-50)
- Context variable `_request_id_ctx` for tracking requests
- Structured logging: `logger.info()`, `logger.warning()`, `logger.error()`, `logger.exception()`

**Validation:**
- Pydantic models for request/response validation at API boundary
- Path traversal prevention via `resolve_task_base_dir()` and `resolve_cookie_file()`
- Cookie file validation with `Path.is_relative_to()`

**Authentication:**
- API key authentication via FastAPI dependency (`main.py` lines 162-178)
- Self-implemented (no external auth provider)
- Toggle via `API_KEY_AUTH_ENABLED` environment variable

**Security:**
- Path traversal prevention for output directories and cookie files
- Non-root Docker user (UID 65532)
- Distroless runtime (minimal attack surface)

---

*Architecture analysis: 2025-01-17*
*Update when major patterns change*
