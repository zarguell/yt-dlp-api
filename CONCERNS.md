# Technical Debt, Issues, and Areas of Concern

This document catalogs actionable technical debt, known issues, security concerns, performance considerations, and documentation gaps in the yt-dlp-api codebase.

---

## Summary

**Overall Assessment:** The codebase is well-structured with good security practices (path traversal prevention, non-root Docker user, hardened runtime) and comprehensive testing (166+ tests, 75% coverage). However, there are several areas requiring attention:

- **High Priority:** Fire-and-forget task pattern, missing .env.example, broad exception handling
- **Medium Priority:** Large monolithic file, duplicate deduplication logic, missing test coverage for error paths
- **Low Priority:** Minor code duplication in subtitle download functions

---

## Technical Debt

### 1. Fire-and-Forget Task Pattern (High Priority)

**Location:** `main.py:1602, 1650, 1702, 1770`

**Issue:** All download endpoints use `asyncio.create_task()` to spawn background tasks without tracking or error handling. If a task fails after the response is sent, there's no mechanism to notify clients or track failures.

```python
# Example from main.py:1602-1614
asyncio.create_task(
    process_task(
        task_id=task_id,
        job_type=JobType.video,
        payload={...},
    )
)
return {"status": "success", "task_id": task_id}
```

**Impact:**
- Tasks may fail silently
- No visibility into background task errors after response
- Difficult to monitor long-running downloads
- No cancellation mechanism for abandoned downloads

**Recommendations:**
1. Implement task result monitoring via `/task/{id}` polling (already partially available)
2. Add background task error logging with context
3. Consider adding a task queue system (Celery, RQ) for production use
4. Document that clients must poll for task completion status

---

### 2. Large Monolithic File (Medium Priority)

**Location:** `main.py` (1,967 lines, 51 functions/classes)

**Issue:** The entire application is contained in a single file, making it difficult to:
- Navigate code efficiently
- Test components in isolation
- Maintain clear separation of concerns
- Onboard new developers

**Impact:**
- Reduced code maintainability
- Higher cognitive load when making changes
- Harder to enforce boundaries between modules

**Current Structure:**
- Lines 1-803: Configuration, utilities, domain models, state/persistence
- Lines 804-1395: YtDlpService (591 lines)
- Lines 1396-1967: FastAPI app, endpoints (571 lines)

**Recommendations:**
1. Split into modules:
   - `config.py` - Configuration classes (`AuthConfig`, `CookieConfig`, `RetryConfig`)
   - `models.py` - Domain models (`Task`, request models, enums)
   - `state.py` - Database persistence (`State` class)
   - `service.py` - YtDlpService
   - `routes/` - FastAPI endpoints by feature
   - `utils.py` - Helper functions (`resolve_task_base_dir`, `normalize_string`, etc.)
2. Consider this as a refactoring effort, not urgent

---

### 3. Duplicate Deduplication Logic (Medium Priority)

**Location:** 
- `main.py:1576-1595` - Video task deduplication
- `main.py:1624-1643` - Audio task deduplication
- `main.py:1676-1695` - Subtitles task deduplication
- `main.py:1744-1763` - Subtitles v2 task deduplication

**Issue:** Each endpoint contains nearly identical deduplication logic:
```python
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
    logger.info("Deduped ...")
    return {"status": "success", "task_id": existing.id}
```

**Impact:**
- Code duplication increases maintenance burden
- Bug fixes must be replicated across all endpoints
- Slight variations in implementation (e.g., format key construction)

**Recommendations:**
1. Extract to a shared function:
```python
def find_existing_task(
    job_type: JobType,
    url: str,
    base_dir: Path,
    format_key: str,
) -> Task | None:
    """Find an existing matching task for deduplication."""
    return next(
        (
            t
            for t in state.tasks.values()
            if t.job_type == job_type
            and t.url == url
            and t.base_output_path == str(base_dir)
            and t.format == format_key
        ),
        None,
    )
```

---

### 4. Missing .env.example File (High Priority)

**Location:** Project root directory

**Issue:** The codebase uses environment variables for configuration but has no `.env.example` file documenting available configuration options.

**Environment Variables Used:**
- `API_KEY_AUTH_ENABLED`
- `API_MASTER_KEY`
- `API_KEY_HEADER_NAME`
- `DEFAULT_MAX_RETRIES`
- `DEFAULT_RETRY_BACKOFF`
- `DEFAULT_RETRY_BACKOFF_MULTIPLIER`
- `DEFAULT_RETRY_JITTER`
- `COOKIES_FILE`
- `COOKIES_DIR`
- `SERVER_OUTPUT_ROOT`
- `HOST`
- `PORT`
- `LOG_LEVEL`
- `MAX_WORKERS`

**Impact:**
- Difficult for new users to discover configuration options
- No reference for what values are expected
- Violates best practices for configurable applications

**Recommendations:**
1. Create `.env.example` with all environment variables documented
2. Add comment blocks explaining each variable's purpose and default value
3. Include in `.gitignore` if not already present (check if `.env` exists)

---

### 5. Broad Exception Handling (Medium Priority)

**Location:** 
- `main.py:696-697` - Database load error
- `main.py:736-737` - Database save error  
- `main.py:1935-1936` - Temp file cleanup error

**Issue:** Several places catch `Exception` without specific error types and only log without handling:

```python
except Exception:
    logger.exception("Error loading tasks from database db_file=%s", self.db_file)
```

**Impact:**
- Masks unexpected errors
- No recovery strategy for database failures
- Application may continue in degraded state without clear indication
- Difficult to debug production issues

**Recommendations:**
1. Use specific exception types where possible (`sqlite3.Error`, `IOError`, etc.)
2. Add metrics/alerting for database failures
3. Consider adding a health check endpoint that validates database connectivity
4. Document what happens when database is unavailable (starts with empty tasks dict)

---

## Security Concerns

### 1. Cookie File Upload Path Traversal Mitigation (Well-Implemented)

**Location:** `main.py:245-285` (`resolve_cookie_file`)

**Status:** ✅ **NOT A CONCERN - Already properly hardened**

**Analysis:** The `resolve_cookie_file` function correctly:
- Validates paths are within `COOKIES_DIR` using `Path.is_relative_to()`
- Rejects paths outside the allowed directory
- Logs rejected paths for security monitoring

**Code (Verified Secure):**
```python
if not cookie_path.is_relative_to(COOKIES_DIR.resolve(strict=False)):
    logger.warning("Rejected cookie path outside COOKIES_DIR path=%s", cookie_path)
    raise HTTPException(
        status_code=400,
        detail="Cookie file path must be within the cookies directory",
    )
```

**Verdict:** This is a good example of proper security implementation. No action needed.

---

### 2. Output Path Traversal Mitigation (Well-Implemented)

**Location:** `main.py:196-237` (`resolve_task_base_dir`, `_is_safe_subdir_name`)

**Status:** ✅ **NOT A CONCERN - Already properly hardened**

**Analysis:** The output path handling correctly:
- Validates folder labels with character whitelist
- Prevents path traversal (`..`, slashes)
- Uses `Path.is_relative_to()` for final validation
- Logs rejected paths

**Verdict:** Proper security implementation. No action needed.

---

### 3. No Size Limits on Cookie Upload (Low Priority)

**Location:** `main.py:1840-1890` (`upload_cookies_file`)

**Issue:** The cookie upload endpoint has no file size limits. A malicious user could upload extremely large files.

**Current Code:**
```python
content = await file.read()  # No size limit
```

**Impact:**
- Potential DoS via large file uploads
- Disk space exhaustion

**Recommendations:**
1. Add file size limit:
```python
MAX_COOKIE_SIZE = 10 * 1024 * 1024  # 10MB
content = await file.read()
if len(content) > MAX_COOKIE_SIZE:
    raise HTTPException(status_code=413, detail="File too large")
```

2. Consider using FastAPI's `UploadFile` with streaming to avoid loading entire file into memory

---

### 4. No Rate Limiting (Medium Priority)

**Location:** All API endpoints

**Issue:** The application has built-in retry logic for external API calls but no rate limiting for API clients. A malicious client could:
- Submit thousands of download tasks
- Exhaust worker threads (default: 4)
- Fill disk space with downloads
- Trigger excessive external API calls

**Impact:**
- Resource exhaustion
- Abuse of service
- Potential costs from external API usage

**Recommendations:**
1. Implement rate limiting per client IP using `slowapi` or similar
2. Add task queue limits (max concurrent downloads per client)
3. Consider adding request throttling for expensive operations
4. Document in README that production deployments should add rate limiting

---

## Performance Concerns

### 1. N+1 Database Query Pattern (Low Priority)

**Location:** `main.py:739-803` (`State` class)

**Issue:** The State class loads all tasks into memory on startup and maintains them in a dict. For large task counts:
- Memory usage grows unbounded
- Startup time increases with task count
- No pagination for `/tasks` endpoint

**Current Implementation:**
```python
def __init__(self, db_file: str = "tasks.db"):
    self.tasks: dict[str, Task] = {}
    self.db_file = db_file
    self._init_db()
    self._load_tasks()  # Loads ALL tasks into memory
```

**Impact:**
- Scaling limitations (memory)
- Slow startup with thousands of tasks
- No cleanup/expiration mechanism for old tasks

**Recommendations:**
1. Implement task cleanup/expiration:
   ```python
   def delete_old_tasks(self, older_than_days: int = 30) -> int:
       """Delete tasks older than specified days."""
       cutoff = datetime.datetime.now() - datetime.timedelta(days=older_than_days)
       # Delete from DB and memory
   ```

2. Add pagination to `/tasks` endpoint:
   ```python
   @app.get("/tasks")
   async def list_all_tasks(
       limit: int = Query(100, ge=1, le=1000),
       offset: int = Query(0, ge=0),
       status: str | None = Query(None)
   ):
   ```

3. Consider not loading all tasks into memory - query DB on demand

---

### 2. No Connection Pooling for SQLite (Low Priority)

**Location:** `main.py:630-737` (database operations)

**Issue:** Each database operation creates a new connection:
```python
conn = sqlite3.connect(self.db_file)
# ... do work ...
conn.close()
```

**Impact:**
- Connection overhead on every save/load
- Potential performance issues with high task churn
- SQLite handles connection pooling poorly

**Recommendations:**
1. Use a single connection with threading mode:
```python
def __init__(self, db_file: str = "tasks.db"):
    self.db_file = db_file
    self._init_db()
    self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
    self._load_tasks()
```

2. Consider using an async database (SQLite with aiosqlite) for better async/await integration

---

### 3. Hardcoded Sleep Intervals (Low Priority)

**Location:** 
- `main.py:841` - `sleep_interval: 10`
- `main.py:842` - `sleep_subtitles: 10`
- `main.py:884-885` - Same values for audio download

**Issue:** Sleep intervals are hardcoded in multiple places and not configurable.

**Impact:**
- Cannot adjust for different platforms' rate limits
- Slower than necessary for some sources, too fast for others
- Values duplicated across download methods

**Recommendations:**
1. Make configurable via environment variables:
   - `YT_DLP_SLEEP_INTERVAL` (default: 10)
   - `YT_DLP_SLEEP_SUBTITLES` (default: 10)

2. Extract to constants or configuration:
```python
DEFAULT_SLEEP_INTERVAL = int(os.getenv("YT_DLP_SLEEP_INTERVAL", "10"))
DEFAULT_SLEEP_SUBTITLES = int(os.getenv("YT_DLP_SLEEP_SUBTITLES", "10"))
```

---

## Documentation Gaps

### 1. Missing Docstrings for Public API Endpoints (Medium Priority)

**Location:** FastAPI route handlers (lines 1571-1956)

**Issue:** Most endpoints lack docstrings, relying on FastAPI's auto-generated docs from Pydantic models. However, complex business logic is not documented.

**Examples:**
- `api_download_video` (line 1572) - No docstring
- `api_download_subtitles_v2` (line 1722) - Has good docstring
- `api_task_zip` (line 1920) - No docstring

**Impact:**
- Difficult to understand endpoint behavior beyond parameter validation
- No documentation of side effects (background tasks)
- No explanation of return value semantics

**Recommendations:**
1. Add docstrings to all endpoint handlers documenting:
   - What the endpoint does
   - Whether it spawns background tasks
   - Return value structure
   - Error conditions
   - Rate limiting considerations

2. Example:
```python
@app.post("/download", response_class=JSONResponse)
async def api_download_video(request: DownloadRequest):
    """
    Submit a video download task for asynchronous processing.
    
    This endpoint returns immediately with a task_id. The download
    runs in the background. Use GET /task/{task_id} to check status.
    
    Deduplication: Returns existing task_id if an identical download
    is already pending or completed.
    
    Returns:
        dict: {"status": "success", "task_id": "..."}
    """
```

---

### 2. Complex Algorithms Lack Inline Comments (Low Priority)

**Location:**
- `main.py:912-1068` - `download_subtitles` (157 lines)
- `main.py:1171-1392` - `download_subtitles_v2` (222 lines)
- `main.py:1071-1168` - `_select_best_subtitle_language` (98 lines)
- `main.py:1143-1168` - `_get_all_english_languages` (26 lines)

**Issue:** Complex subtitle selection logic has some docstrings but limited inline comments explaining the algorithm flow.

**Example from `download_subtitles_v2`:**
- Has good high-level docstring (lines 1182-1198)
- Uses numbered step comments (lines 1222, 1263, 1322)
- But some sections lack explanatory comments

**Impact:**
- More difficult for maintainers to understand algorithm flow
- Harder to debug issues with subtitle selection
- Knowledge transfer risk

**Recommendations:**
1. Add inline comments for non-obvious logic:
   ```python
   # Step 2: Select languages based on policy
   # This ensures we download the right subtitles before making
   # any network requests for the actual subtitle files
   ```

2. Consider extracting regex compilation to constants for clarity

---

### 3. Missing Error Scenario Documentation (Medium Priority)

**Location:** README.md

**Issue:** The README documents success cases well but doesn't explain:
- What happens when rate limits are hit (HTTP 429)
- How to handle partial success for subtitles
- What error messages to expect
- How to diagnose failed downloads

**Impact:**
- Users may not know how to handle errors
- Debugging failed downloads is difficult
- Unclear what retry behavior to expect

**Recommendations:**
1. Add "Error Handling" section with:
   - Common error responses
   - Rate limit behavior (automatic retry)
   - How to check task status after errors
   - Troubleshooting guide

2. Document the partial success pattern:
   ```markdown
   ### Partial Success for Subtitles
   
   When downloading multiple subtitle languages, some may fail due to
   rate limiting while others succeed. In this case, the task status
   will be "partial" and the result will contain:
   - `downloaded`: List of successfully downloaded files
   - `failed`: List of errors for failed downloads
   
   You can still retrieve the successfully downloaded files via the
   artifact endpoints.
   ```

---

## Missing Tests

### 1. Error Path Testing Gaps (Medium Priority)

**Location:** `tests/test_api.py`

**Issue:** API tests focus on happy paths but don't adequately test:
- Error responses for invalid URLs
- Rate limit error handling
- Database failure scenarios
- Cookie file validation errors
- Output path rejection

**Current Test Coverage:**
- Test classes present: `TestHealthEndpoints`, `TestDownloadEndpoints`, `TestTaskEndpoints`, `TestInfoEndpoints`, `TestCookieEndpoints`, `TestTaskFileEndpoints`, `TestSubtitlesV2Endpoints`, `TestAuthentication`
- Tests appear to mock `process_task`, reducing integration value

**Recommendations:**
1. Add error case tests:
   ```python
   async def test_invalid_output_path_rejected(self):
       """Test that paths with .. are rejected."""
       payload = {"url": "http://example.com", "output_path": "../etc"}
       response = await client.post("/download", json=payload)
       assert response.status_code == 400
   ```

2. Test database failure scenarios:
   ```python
   async def test_task_not_found_error(self):
       """Test 404 for non-existent task."""
       response = await client.get("/task/nonexistent")
       assert response.status_code == 404
   ```

3. Test cookie path traversal attempts:
   ```python
   async def test_cookie_path_traversal_rejected(self):
       """Test that cookie paths outside COOKIES_DIR are rejected."""
       # Upload attempt with malicious path
   ```

---

### 2. Retry Logic Edge Cases (Medium Priority)

**Location:** `tests/test_retry.py` (325 lines)

**Issue:** While retry logic has dedicated tests, they may not cover:
- Max retry exhaustion scenarios
- Jitter randomness (may make tests flaky)
- Non-retryable error handling
- Backoff calculation edge cases

**Current Implementation:**
- Uses `jitter=False` in test fixture for predictability
- Tests retryable error detection
- Tests backoff calculation

**Recommendations:**
1. Add test for max retries exhausted:
   ```python
   def test_max_retries_exhausted(self):
       """Verify error is raised after max retries."""
       mock_func = Mock(side_effect=Exception("HTTP error 429"))
       with pytest.raises(Exception):
           retry_with_backoff(mock_func, retry_config)
       assert mock_func.call_count == retry_config.max_retries + 1
   ```

2. Test non-retryable errors:
   ```python
   def test_non_retryable_error_fails_immediately(self):
       """Verify non-retryable errors fail immediately."""
       mock_func = Mock(side_effect=ValueError("Bad input"))
       with pytest.raises(ValueError):
           retry_with_backoff(mock_func, retry_config)
       assert mock_func.call_count == 1  # Only called once
   ```

---

### 3. Subtitle Selection Algorithm Tests (Low Priority)

**Location:** `tests/test_api.py` - `TestSubtitlesV2Endpoints`

**Issue:** The complex subtitle selection logic (`_select_best_subtitle_language`, `_get_all_english_languages`) may not be thoroughly tested with:
- Edge cases in language code matching
- Regex pattern variations
- Manual vs automatic subtitle preference combinations

**Recommendations:**
1. Add unit tests for `_select_best_subtitle_language`:
   ```python
   def test_select_best_finds_exact_match(self):
       """Test exact language match is preferred."""
       info = {"subtitles": {"en": [...]}, "automatic_captions": {"en": [...]}}
       result = YtDlpService._select_best_subtitle_language(
           info, ["en"], SubtitlePreference.manual_then_auto
       )
       assert result == "en"
   ```

2. Test regex pattern matching:
   ```python
   def test_select_best_uses_regex_pattern(self):
       """Test that 'en.*' matches 'en-US'."""
       info = {"subtitles": {"en-US": [...]}}
       result = YtDlpService._select_best_subtitle_language(
           info, ["en.*"], SubtitlePreference.manual_only
       )
       assert result == "en-US"
   ```

---

## Dependencies

### 1. Python Version Target vs Runtime (Low Priority)

**Location:** 
- `pyproject.toml:54` - `target-version = "py313"`
- `Dockerfile:5` - `FROM docker.io/python:3.13-slim-bookworm`
- `.github/workflows/test.yml:177` - Tests 3.11, 3.12, 3.13
- `README.md:29` - States "Python 3.10+ (3.11+ recommended)"

**Issue:** Inconsistency in Python version requirements:
- Code targets Python 3.13
- Tests run on 3.11, 3.12, 3.13
- README says 3.10+ is OK

**Impact:**
- Confusion about supported versions
- Potential for type errors on older versions
- Features like `str | None` syntax require Python 3.10+

**Recommendations:**
1. Standardize on minimum Python version (recommend 3.11)
2. Update `pyproject.toml`: `target-version = "py311"`
3. Update Dockerfile to use 3.11 for broader compatibility
4. Update CI to test 3.11 and 3.13 (drop 3.12 if not needed)
5. Update README to state "Python 3.11+"

---

### 2. zipp Version Pinning Comment (Informational)

**Location:** `requirements.txt:16`

**Issue:** Comment states "pinned by Snyk to avoid a vulnerability" but doesn't document the specific vulnerability or when it can be unpinned.

```
zipp>=3.19.1 # not directly required, pinned by Snyk to avoid a vulnerability
```

**Impact:**
- Unclear when pinning can be removed
- Potential for outdated dependency

**Recommendations:**
1. Create SECURITY_NOTES.md or add to CONCERNS.md:
   ```markdown
   ### zipp Dependency
   
   Version 3.19.1+ is pinned due to CVE-XXXX-XXXX (path traversal).
   Can be unpinned when dependency chain upgrades.
   See: [link to CVE/issue]
   ```

2. Set calendar reminder to review quarterly

---

## Duplicate Code Patterns

### 1. Subtitle Download File Tracking (Low Priority)

**Location:** 
- `main.py:997-1007` - File tracking in `download_subtitles` (try block)
- `main.py:1034-1044` - File tracking in `download_subtitles` (except block)
- `main.py:1337-1345` - File tracking in `download_subtitles_v2` (try block)
- `main.py:1368-1376` - File tracking in `download_subtitles_v2` (except block)

**Issue:** Identical file discovery and tracking logic duplicated across both subtitle download functions:

```python
for f in new_files:
    if f.is_file():
        downloaded_files.append(
            {
                "name": f.name,
                "size_bytes": f.stat().st_size,
                "path": str(f),
            }
        )
```

**Impact:**
- Code duplication (4 occurrences)
- Bug fixes must be replicated
- Harder to ensure consistent behavior

**Recommendations:**
1. Extract to helper function:
```python
def collect_file_info(files: set[Path]) -> list[dict[str, Any]]:
    """Collect file metadata from a set of file paths."""
    downloaded_files = []
    for f in files:
        if f.is_file():
            downloaded_files.append(
                {
                    "name": f.name,
                    "size_bytes": f.stat().st_size,
                    "path": str(f),
                }
            )
    return downloaded_files
```

---

### 2. yt-dlp Options Construction (Low Priority)

**Location:** 
- `main.py:835-848` - Video download options
- `main.py:876-893` - Audio download options
- `main.py:934-960` - Subtitle download options
- `main.py:1285-1310` - Subtitles v2 download options

**Issue:** Similar yt-dlp option dictionaries constructed in each download method with overlapping fields (`quiet`, `no_warnings`, `cookiefile`, `sleep_interval`, `sleep_subtitles`).

**Impact:**
- Inconsistent option values across methods
- New options must be added to all methods
- Risk of missing critical options in some methods

**Recommendations:**
1. Extract common options to helper:
```python
def build_base_ydl_opts(
    quiet: bool,
    output_path: str,
    cookie_file: str | None = None,
) -> dict[str, Any]:
    """Build base yt-dlp options common to all downloads."""
    opts = {
        "outtmpl": str(Path(output_path) / "%(title).180s.%(ext)s"),
        "quiet": quiet,
        "no_warnings": quiet,
        "no_abort_on_error": True,
        "sleep_interval": DEFAULT_SLEEP_INTERVAL,
        "sleep_subtitles": DEFAULT_SLEEP_SUBTITLES,
    }
    if cookie_file:
        opts["cookiefile"] = cookie_file
    return opts
```

---

## Code Quality

### 1. Type Checking Inconsistency (Low Priority)

**Location:** `main.py` throughout

**Issue:** 
- `pyproject.toml:78` - `disallow_untyped_defs = false`
- Most code has type hints
- Some functions missing return type hints

**Examples:**
- Line 636-651: `_init_db()` has no return type
- Line 653-697: `_load_tasks()` has no return type

**Impact:**
- Reduced type safety
- mypy doesn't catch all type errors
- Harder for IDE autocomplete

**Recommendations:**
1. Gradually add missing type hints
2. Set `disallow_untyped_defs = true` after coverage is complete
3. Use mypy strict mode for new code

---

### 2. Magic Numbers (Low Priority)

**Location:** Various

**Issue:** Several magic numbers without explanation:
- `max_length=80` (line 196) - output path validation
- `max_length=200` (line 294) - filename normalization
- `%.180s` (line 834, 875, 933, 1264) - title truncation
- `max_workers=4` (line 1404) - thread pool size

**Impact:**
- Unclear why these values were chosen
- Difficult to tune for different environments
- Hardcoded values limit configurability

**Recommendations:**
1. Extract to constants with comments:
```python
# Output path validation limits
MAX_OUTPUT_PATH_LENGTH = 80  # Prevents excessively long directory names

# Filename normalization limits
MAX_FILENAME_LENGTH = 200  # Common filesystem limit

# yt-dlp title template
MAX_TITLE_LENGTH = 180  # Prevents excessively long filenames
```

2. Make thread pool size configurable (already partially done with `MAX_WORKERS` env var)

---

## Positive Observations (Not Concerns)

The following areas are well-implemented and serve as good practices:

1. **Path Traversal Protection** - Both `resolve_task_base_dir` and `resolve_cookie_file` properly validate paths using `Path.is_relative_to()`

2. **Security Hardened Docker** - Multi-stage build, distroless runtime, non-root user

3. **Comprehensive Logging** - Request IDs, structured logging, appropriate log levels

4. **Retry Logic** - Well-implemented exponential backoff with jitter for rate limiting

5. **Task Deduplication** - Prevents duplicate downloads for identical requests

6. **Partial Success Handling** - Graceful handling when some subtitles fail

7. **Type Hints** - Most code has comprehensive type annotations

8. **Testing Infrastructure** - Good test setup with fixtures, mocking, and CI/CD

9. **Documentation** - Excellent README with comprehensive examples

10. **Environment-based Configuration** - No hardcoded secrets, all config via env vars

---

## Action Priority Summary

### Immediate (This Sprint)
1. Create `.env.example` file with all environment variables documented
2. Add file size limits to cookie upload endpoint (10MB max)

### Short-term (Next Sprint)
3. Extract deduplication logic to shared function (reduces 4 duplicate blocks)
4. Add error path tests for critical endpoints
5. Document fire-and-forget pattern and polling requirement in README

### Medium-term (Next Quarter)
6. Consider refactoring monolithic file into modules
7. Implement task cleanup/expiration mechanism
8. Add pagination to `/tasks` endpoint
9. Implement rate limiting for production deployments

### Long-term (As Needed)
10. Evaluate task queue system (Celery/RQ) for better background task management
11. Migrate to async database (aiosqlite) for better async integration
12. Standardize Python version requirements

---

## Conclusion

This codebase demonstrates strong security practices, good testing coverage, and thoughtful design. The primary areas of concern are:

1. **Architectural**: Fire-and-forget task pattern needs better error visibility
2. **Maintainability**: Monolithic file structure and code duplication
3. **Documentation**: Missing .env.example and error scenario documentation
4. **Configuration**: Some hardcoded values should be environment variables

None of the concerns represent critical security vulnerabilities or showstoppers. The codebase is production-ready with the understanding that:
- Clients must poll `/task/{id}` for download completion
- Production deployments should add rate limiting
- Task cleanup is manual (no automatic expiration)

Overall, this is a well-maintained project with room for incremental improvements.
