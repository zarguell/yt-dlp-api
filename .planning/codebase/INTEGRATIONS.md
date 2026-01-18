# External Integrations

**Analysis Date:** 2025-01-17

## APIs & External Services

**Video Platform APIs:**
- YouTube - Primary target via yt-dlp extractor
- Video platforms - Any yt-dlp-supported site (500+ sites)
- Integration method: yt-dlp library (no direct API calls)
- Auth: Cookie-based authentication via `COOKIES_FILE` or `/cookies/upload` endpoint

**No External APIs Used:**
- No payment processors (Stripe, PayPal)
- No email services (SendGrid, Mailgun)
- No analytics (Google Analytics, Mixpanel)
- No CDN services

## Data Storage

**Databases:**
- SQLite (embedded) - Task persistence and metadata
  - Connection: `tasks.db` file in working directory
  - Client: Python sqlite3 stdlib (`main.py` line 8, lines 623-803)
  - Migrations: Manual schema creation in `State._init_db()`
  - In-memory cache: `State.tasks` dict for fast lookups

**File Storage:**
- Local filesystem - Downloaded media files
  - Location: `SERVER_OUTPUT_ROOT` environment variable
  - Path: `downloads/default/{task_id}/` by default
  - No cloud storage (no S3, GCS, Azure Blob)

**Caching:**
- In-memory task cache - `State.tasks` dict
- No Redis or external caching

## Authentication & Identity

**Auth Provider:**
- Self-implemented API key authentication (`main.py` lines 103-178)
- No external auth provider (no Auth0, Supabase Auth, etc.)
- Token storage: Client-side in `X-API-Key` header
- Session management: Stateless (no sessions)
- Implementation: FastAPI dependency injection

**OAuth Integrations:**
- None (all auth via API keys or cookies passed to yt-dlp)

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry, Datadog, etc.)
- Python stdlib logging only (`main.py` lines 36-50)

**Analytics:**
- None (no product analytics)

**Logs:**
- Python stdlib logging - stdout/stderr only
- Structured logging with request ID correlation
- No log aggregation service

## CI/CD & Deployment

**Hosting:**
- Docker containers - Multi-arch builds (amd64, arm64)
- Deployment: Manual or via CI/CD
- Environment vars: Configured in container environment

**CI Pipeline:**
- GitHub Actions - `.github/workflows/`
  - `test.yml` - Test automation across Python 3.11, 3.12, 3.13
  - `docker-image.yml` - Docker build and publish
  - `codeql.yml` - CodeQL security scanning
  - `scorecards.yml` - Supply chain scorecards
- Secrets: Stored in GitHub repository secrets

**Code Quality:**
- DeepSource - Automated code analysis (`.deepsource.toml`)
- Codecov - Coverage reporting
- Dependabot - Dependency updates (`.github/dependabot.yml`)
- Renovate - Alternative dependency bot (`renovate.json`)

## Environment Configuration

**Development:**
- Required env vars: Optional (has sensible defaults)
- Secrets location: Environment variables directly (no .env files)
- Mock/stub services: yt-dlp mocked in tests, real for local development

**Staging:**
- Not documented (likely uses same Docker image with different env vars)

**Production:**
- Secrets management: Container environment variables
- Failover/redundancy: Not configured (single-instance deployment)

## Webhooks & Callbacks

**Incoming:**
- None (no webhooks received from external services)

**Outgoing:**
- None (no webhooks sent to external services)

## Third-Party Media Processing

**ffmpeg/ffprobe:**
- Static binaries from `mwader/static-ffmpeg:8.0.1`
- Used for: Audio extraction, subtitle conversion, video processing
- Integration: yt-dlp calls ffmpeg automatically
- Location: `/usr/local/bin/ffmpeg`, `/usr/local/bin/ffprobe` in container

---

*Integration audit: 2025-01-17*
*Update when adding/removing external services*
