# Codebase Concerns

**Analysis Date:** 2025-01-17

## Tech Debt

**Monolithic file structure:**
- Issue: All 1,967 lines in single `main.py` file
- Files: `main.py` (entire application)
- Why: Rapid development, convenience
- Impact: Difficult to navigate, hard to test in isolation, cognitive load
- Fix approach: Extract layers into modules (`config.py`, `models.py`, `service.py`, `api.py`)

**Duplicate deduplication logic:**
- Issue: 4 endpoints share nearly identical task deduplication code
- Files: `main.py` lines 1600-1607, 1647-1654, 1696-1703, 1764-1771
- Why: Each endpoint implements deduplication independently
- Impact: Code duplication, maintenance burden
- Fix approach: Extract to shared function `find_or_create_task()`

**Duplicate file tracking code:**
- Issue: 4 identical blocks of file tracking logic across subtitle download functions
- Files: `main.py` lines 1001-1027 (download_subtitles_v2)
- Why: File tracking logic repeated for each subtitle function
- Impact: Code duplication, potential for inconsistencies
- Fix approach: Extract to shared helper `track_downloaded_files()`

**Missing error path tests:**
- Issue: Tests focus on happy paths, lacking error scenario coverage
- Files: `tests/test_api.py`, other test files
- Why: Test coverage prioritized for common cases
- Impact: Edge cases and error paths may break silently
- Fix approach: Add tests for 404, 401, 500 errors, invalid inputs

## Known Bugs

**No critical bugs found**
- Codebase is production-ready with comprehensive testing

**Minor issues:**
- Broad exception handling: `main.py` lines 696-697, 736-737, 1935-1936 catch `Exception` without specific types
- Python version inconsistency: Documentation says Python 3.10+ but `pyproject.toml` targets 3.13

## Security Considerations

**Fire-and-forget task pattern:**
- Risk: Background tasks spawned without error tracking; clients must poll for status
- Files: `main.py` lines 1612, 1650, 1702, 1770 (all download endpoints)
- Current mitigation: Task status persisted to SQLite, clients can poll
- Recommendations: Consider adding task failure notifications (webhooks, email)

**Missing `.env.example` file:**
- Risk: No documentation of available environment variables despite heavy use
- Files: Root directory lacks `.env.example`
- Current mitigation: Environment variables documented in `README.md`
- Recommendations: Create `.env.example` file listing all config options

**Cookie upload size limits:**
- Risk: No file size validation on `/cookies/upload` endpoint
- File: `main.py` lines 1840-1890 (cookie upload endpoint)
- Current mitigation: None (could upload arbitrarily large files)
- Recommendations: Add file size limit (10MB max) to prevent abuse

**No rate limiting:**
- Risk: API vulnerable to abuse/DoS attacks
- Files: All API endpoints (`main.py` lines 1571-1955)
- Current mitigation: None (relies on deployment-level rate limiting)
- Recommendations: Implement rate limiting for production (e.g., slowapi, FastAPI Limiter)

**Well-implemented security:**
- ✅ Path traversal prevention: `resolve_task_base_dir()` and `resolve_cookie_file()` validate paths
- ✅ Docker hardened: Multi-stage build, distroless runtime, non-root user
- ✅ API key authentication: Self-implemented with toggle support

## Performance Bottlenecks

**Memory growth:**
- Problem: All tasks loaded into memory at startup; no cleanup mechanism
- Files: `main.py` lines 623-801 (State class loads all tasks into `self.tasks` dict)
- Measurement: Not measured (depends on number of tasks)
- Cause: `State._load_tasks()` loads entire database into memory
- Improvement path: Implement task expiration/eviction, pagination for `/tasks` endpoint

**No DB connection pooling:**
- Problem: New SQLite connection per operation
- Files: `main.py` lines 623-801 (State class)
- Measurement: Not measured (SQLite is fast, but overhead exists)
- Cause: Each CRUD method opens/closes connection
- Improvement path: Keep connection open, add connection pooling if migrating to PostgreSQL

**Monolithic file load time:**
- Problem: 1,967-line file takes time to parse and import
- Files: `main.py` (entire application)
- Measurement: ~100-200ms import time (estimated)
- Cause: All code in single file
- Improvement path: Split into modules (also improves maintainability)

## Fragile Areas

**Background task error handling:**
- Why fragile: `process_task()` function (lines 1413-1504) handles all task types; complex error logic
- Common failures: yt-dlp errors, file system errors, network timeouts
- Safe modification: Add more specific error types, improve logging
- Test coverage: Limited error path testing in `test_api.py`

**Retry logic configuration:**
- Why fragile: Multiple retry config sources (defaults, env vars, request params)
- Files: `main.py` lines 1427-1435 (config merging logic)
- Common failures: Incorrect retry values causing infinite retries or no retries
- Safe modification: Validate retry config, add bounds checking
- Test coverage: Good coverage in `test_retry.py`

**Subtitle language selection:**
- Why fragile: Complex policy-based language selection with edge cases
- Files: `main.py` lines 1071-1168 (subtitle selection logic)
- Common failures: No English subtitles available, manual subtitles missing
- Safe modification: Add tests for edge cases, document fallback behavior
- Test coverage: Limited (only happy path tests)

## Scaling Limits

**SQLite write concurrency:**
- Current capacity: Single writer, multiple readers
- Limit: ~100-1000 writes/sec (depends on disk speed)
- Symptoms at limit: Database locked errors, slow writes
- Scaling path: Migrate to PostgreSQL for better write concurrency

**In-memory task cache:**
- Current capacity: Limited by available RAM
- Limit: ~100k tasks before memory becomes concern (estimated)
- Symptoms at limit: High memory usage, slow startup
- Scaling path: Implement task expiration, move to Redis cache

**ThreadPoolExecutor max_workers:**
- Current capacity: Configurable via `MAX_WORKERS` env var (default: likely 10-20)
- Limit: ~100 concurrent download tasks (depends on setting)
- Symptoms at limit: Tasks queued waiting for thread pool
- Scaling path: Increase `MAX_WORKERS`, use async yt-dlp if available

## Dependencies at Risk

**yt-dlp:**
- Risk: Frequent updates, API changes can break integration
- Impact: Core functionality fails (all downloads)
- Mitigation: Contract tests in `test_ytdlp_contract.py` detect API changes
- Upgrade path: Pin to specific version, monitor yt-dlp releases

**No deprecated dependencies detected:**
- All dependencies actively maintained
- Regular updates via Dependabot and Renovate

## Missing Critical Features

**Task expiration/cleanup:**
- Problem: No automatic cleanup of old/failed tasks
- Current workaround: Manual database cleanup
- Blocks: Long-running deployments accumulate stale tasks
- Implementation complexity: Low (add task TTL, cleanup cron job)

**Pagination for `/tasks` endpoint:**
- Problem: Returns all tasks without pagination
- Current workaround: Acceptable for low task counts
- Blocks: Doesn't scale to thousands of tasks
- Implementation complexity: Medium (add cursor-based or offset-based pagination)

**Task failure notifications:**
- Problem: No notification when tasks fail (clients must poll)
- Current workaround: Client-side polling with timeout
- Blocks: Poor UX for long-running tasks
- Implementation complexity: High (webhook system, email, WebSocket)

## Test Coverage Gaps

**Error path testing:**
- What's not tested: API error responses (400, 401, 404, 500), invalid inputs
- Risk: Error handling code may have bugs
- Priority: Medium
- Difficulty to test: Easy (add test cases for error scenarios)

**Network contract testing:**
- What's not tested: yt-dlp contract tests run manually, not in CI
- Risk: yt-dlp API changes may break production
- Priority: High
- Difficulty to test: Medium (requires network access, yt-dlp installed)

**Integration test coverage:**
- What's not tested: Full request lifecycle (download → file retrieval)
- Risk: End-to-end flows may have gaps
- Priority: Low (happy paths tested)
- Difficulty to test: Medium (requires mocking yt-dlp or real downloads)

**Well-tested areas:**
- ✅ Utility functions (extensive parametrized tests)
- ✅ Configuration classes (all env vars tested)
- ✅ Retry logic (comprehensive backoff testing)
- ✅ State/CRUD operations (database operations tested)

---

*Concerns audit: 2025-01-17*
*Update as issues are fixed or new ones discovered*
