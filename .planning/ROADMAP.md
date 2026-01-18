# Roadmap: Monolithic Code Refactoring

## Overview

Refactor a 1,967-line monolithic `main.py` into a modular, maintainable structure with clear separation of concerns. This is an internal refactoring—no API contract changes or new features. The journey moves from extraction of foundational components (config, models) through each major layer (state, service, routes), with validation at each step that tests still pass and Docker builds succeed.

## Domain Expertise

None

## Phases

- [ ] **Phase 1: Foundation** - Extract configuration and domain models
- [ ] **Phase 2: State & Persistence** - Extract database layer and validate
- [ ] **Phase 3: Service Layer** - Extract YtDlpService and validate
- [ ] **Phase 4: Routes Organization** - Split endpoints into feature modules
- [ ] **Phase 5: Integration Testing** - Verify API behavior identical
- [ ] **Phase 6: Docker & CI Validation** - Ensure deployment still works

## Phase Details

### Phase 1: Foundation
**Goal**: Extract configuration classes and domain models into separate modules
**Depends on**: Nothing (first phase)
**Research**: Unlikely (internal refactoring, established patterns)
**Plans**: 2-3 plans

Plans:
- [ ] 01-01: Create config.py with AuthConfig, CookieConfig, RetryConfig
- [ ] 01-02: Create models.py with Task, request models, enums
- [ ] 01-03: Create utils.py for helper functions (path resolution, string normalization)

### Phase 2: State & Persistence
**Goal**: Extract State class and database operations into state.py
**Depends on**: Phase 1 (models needed for type hints)
**Research**: Unlikely (SQLite operations already exist)
**Plans**: 2 plans

Plans:
- [ ] 02-01: Create state.py with State class and database operations
- [ ] 02-02: Update imports and verify tests pass

### Phase 3: Service Layer
**Goal**: Extract YtDlpService into service.py
**Depends on**: Phase 2 (State class needed by YtDlpService)
**Research**: Unlikely (wrapper around existing yt-dlp library)
**Plans**: 2 plans

Plans:
- [ ] 03-01: Create service.py with YtDlpService class
- [ ] 03-02: Update imports and verify tests pass

### Phase 4: Routes Organization
**Goal**: Split FastAPI endpoints into feature-based route modules
**Depends on**: Phase 3 (service layer needed by endpoints)
**Research**: Unlikely (FastAPI patterns already established)
**Plans**: 3 plans

Plans:
- [ ] 04-01: Create routes/ directory structure
- [ ] 04-02: Extract endpoint groups by feature (downloads, tasks, files, cookies)
- [ ] 04-03: Create main.py that imports and registers routes

### Phase 5: Integration Testing
**Goal**: Verify API behavior is identical to original implementation
**Depends on**: Phase 4 (all modules extracted)
**Research**: Unlikely (existing test suite)
**Plans**: 2 plans

Plans:
- [ ] 05-01: Run full test suite and verify 75%+ coverage maintained
- [ ] 05-02: Test API endpoints manually or with integration tests

### Phase 6: Docker & CI Validation
**Goal**: Ensure Docker builds and CI/CD workflows still function
**Depends on**: Phase 5 (refactoring complete)
**Research**: Unlikely (existing Docker setup)
**Plans**: 2 plans

Plans:
- [ ] 06-01: Build Docker image locally and verify it runs
- [ ] 06-02: Verify GitHub Actions workflows pass

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 0/3 | Not started | - |
| 2. State & Persistence | 0/2 | Not started | - |
| 3. Service Layer | 0/2 | Not started | - |
| 4. Routes Organization | 0/3 | Not started | - |
| 5. Integration Testing | 0/2 | Not started | - |
| 6. Docker & CI Validation | 0/2 | Not started | - |
