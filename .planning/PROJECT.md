# Monolithic Code Refactoring

## What This Is

Refactor the yt-dlp-api codebase from a single 1,967-line `main.py` file into a modular, maintainable structure with clear separation of concerns. This improves code organization, testability, and ability to evolve the codebase without introducing breaking changes to the existing REST API.

## Core Value

**No breaking changes.** The refactored code must provide identical API behavior and functionality while establishing a foundation for future improvements.

## Requirements

### Validated

- ✓ Video download API endpoint (POST /download) — existing
- ✓ Audio-only download API endpoint (POST /audio) — existing
- ✓ Subtitles download API endpoint (POST /subtitles) — existing
- ✓ Enhanced subtitles v2 download (POST /v2/subtitles) — existing
- ✓ Task status and listing endpoints (GET /task/{id}, GET /tasks) — existing
- ✓ Video metadata endpoints (GET /info, GET /formats) — existing
- ✓ File retrieval endpoints (GET /task/{id}/files, /file, /zip) — existing
- ✓ Cookie upload endpoint (POST /cookies/upload) — existing
- ✓ API key authentication (optional, env-controlled) — existing
- ✓ Task deduplication logic — existing
- ✓ Path traversal protection for output paths — existing
- ✓ Path traversal protection for cookie files — existing
- ✓ Retry logic with exponential backoff — existing
- ✓ SQLite task persistence — existing
- ✓ Async background task processing — existing

### Active

- [ ] Split main.py into logical modules following concerns.md recommendation
- [ ] Create config.py for configuration classes (AuthConfig, CookieConfig, RetryConfig)
- [ ] Create models.py for domain models (Task, request models, enums)
- [ ] Create state.py for database persistence (State class)
- [ ] Create service.py for YtDlpService
- [ ] Create routes/ directory with endpoint modules by feature
- [ ] Create utils.py for helper functions (resolve_task_base_dir, normalize_string, etc.)
- [ ] Ensure all imports resolve correctly after refactoring
- [ ] Verify all existing tests pass without modification
- [ ] Verify Docker container builds and runs successfully
- [ ] Verify API endpoints respond identically to current implementation

### Out of Scope

- API contract changes — maintain backward compatibility
- Database schema changes — keep tasks.db structure unchanged
- New features or capabilities — focus on structural refactoring only
- Performance optimization — preserve current performance characteristics
- External dependency changes — maintain existing dependencies
- Test coverage expansion — maintain existing 75% coverage

## Context

The yt-dlp-api project is a production-ready RESTful API service built with FastAPI that provides a web interface to yt-dlp (YouTube/media downloader). The current codebase consists of a single 1,967-line `main.py` file containing all application logic, which has been identified as the top concern in concerns.md (Technical Debt #2).

**Current structure (from ARCHITECTURE.md):**
- Lines 1-803: Configuration, utilities, domain models, state/persistence
- Lines 804-1395: YtDlpService (591 lines)
- Lines 1396-1967: FastAPI app, endpoints (571 lines)

**Issues with current structure:**
- Difficult to navigate code efficiently
- Hard to test components in isolation
- Poor separation of concerns
- High cognitive load for maintenance
- Harder to enforce boundaries between modules

**Recommendation from concerns.md:**
Split into modules:
- config.py - Configuration classes
- models.py - Domain models
- state.py - Database persistence
- service.py - YtDlpService
- routes/ - FastAPI endpoints by feature
- utils.py - Helper functions

The project has comprehensive testing (166+ tests, 75% coverage), Docker support, and CI/CD workflows that must continue to work after refactoring.

## Constraints

None specified. Focus on code quality and maintainability improvements while preserving all existing functionality.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Split monolithic main.py into modules | Improves maintainability, testability, and scalability | — Pending |
| Maintain API contract unchanged | Ensures no breaking changes for existing users | — Pending |
| Keep all existing dependencies | Avoid scope creep and maintain Docker compatibility | — Pending |

---
*Last updated: 2026-01-17 after initialization*
