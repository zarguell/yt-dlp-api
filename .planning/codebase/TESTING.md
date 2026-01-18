# Testing Patterns

**Analysis Date:** 2025-01-17

## Test Framework

**Runner:**
- pytest 9.0.2 - configured in `pyproject.toml` lines 1-28
- Config: `pyproject.toml` [tool.pytest.ini_options] section

**Assertion Library:**
- pytest built-in `assert` statement
- Matchers: Standard Python comparison operators (`==`, `!=`, `in`, `isinstance`)

**Run Commands:**
```bash
make test              # Run fast unit tests (exclude slow/network)
make test-cov          # Run tests with coverage report
make test-unit         # Run unit tests only (`-m "unit"`)
make test-integration  # Run integration tests only (`-m "integration"`)
pytest                 # Run all tests
pytest -m "not network" # Skip network tests
pytest -v              # Verbose output
```

## Test File Organization

**Location:**
- `tests/` directory - separate test tree
- `tests/test_*.py` pattern - all test files
- No co-located tests (tests not alongside source)

**Naming:**
- `test_*.py` for all test files
- Test classes: `Test*` (PascalCase) - e.g. `TestEnvTruthy`
- Test methods: `test_*` (snake_case) - e.g. `test_env_truthy_values`

**Structure:**
```
tests/
├── __init__.py         # Package marker (empty)
├── conftest.py         # Shared pytest fixtures (199 lines)
├── test_utils.py       # Utility function tests
├── test_config.py      # Configuration class tests
├── test_state.py       # Database/state tests
├── test_retry.py       # Retry logic tests
├── test_api.py         # API endpoint integration tests
└── test_ytdlp_contract.py  # yt-dlp contract tests
```

## Test Structure

**Suite Organization:**
```python
# Example from tests/test_utils.py
class TestEnvTruthy:
    """Tests for _env_truthy function."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("1", True),
            ("true", True),
            ("0", False),
            ("false", False),
        ],
    )
    def test_env_truthy_values(self, value: str, expected: bool) -> None:
        """Test various truthy/falsey string values."""
        assert _env_truthy(value) == expected
```

**Patterns:**
- Use `@pytest.mark.parametrize` for data-driven tests
- Use fixtures from `conftest.py` for shared test data
- Test class names: `Test*` prefix
- Test method names: `test_*` prefix
- Type hints on test function signatures

## Mocking

**Framework:**
- pytest-mock 3.14.0 - `mocker` fixture
- unittest.mock.MagicMock - Manual mocking
- Module mocking via `unittest.mock.patch`

**Patterns:**
```python
# Example from tests/test_api.py
def test_download_video_success(
    self, async_client: httpx.AsyncClient, sample_video_url: str
) -> None:
    """Test successful video download endpoint."""
    with unittest.mock.patch.object(
        main.YtDlpService, "download_video"
    ) as mock_download:
        mock_download.return_value = {"status": "downloaded"}

        response = async_client.post(
            "/download",
            json={"url": sample_video_url, "output_path": "test"},
        )

        assert response.status_code == 200
        assert "task_id" in response.json()
```

**What to Mock:**
- External APIs: yt-dlp library (YtDlpService methods)
- File system: Temporary directories via fixtures
- Database: In-memory SQLite via fixtures
- Network: Skip tests marked with `@pytest.mark.network` in CI

**What NOT to Mock:**
- Pure functions (utility functions test real implementation)
- Configuration classes (test real environment parsing)
- State class (test with temporary database)

## Fixtures and Factories

**Test Data:**
```python
# Example from tests/conftest.py
@pytest.fixture
def temp_dir():
    """Create a temporary directory that auto-cleans."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def test_state(temp_dir: Path, temp_db: Path):
    """Create State instance with temporary database."""
    return State(db_path=temp_db, output_root=temp_dir)

@pytest.fixture
def sample_task_id() -> str:
    """Generate a sample task ID for testing."""
    return str(uuid.uuid4())
```

**Location:**
- Shared fixtures: `tests/conftest.py` (199 lines)
- Test-specific fixtures: Define in test file
- No factory functions (use fixtures directly)

## Coverage

**Requirements:**
- Target: ~75% coverage (per `AGENTS.md`)
- Enforcement: No enforcement (coverage for awareness only)

**Configuration:**
- Tool: pytest-cov (configured in `pyproject.toml` lines 30-50)
- Exclusions: `tests/*`, `*/__pycache__/*`, `*/venv/*`, `if __name__ == .__main__.:`

**View Coverage:**
```bash
make test-cov          # Generate coverage report
open htmlcov/index.html  # View HTML report (if generated)
```

## Test Types

**Unit Tests:**
- Scope: Test single function/class in isolation
- Mocking: Mock external dependencies (yt-dlp, file system)
- Speed: Fast (<1s per test)
- Examples: `test_utils.py`, `test_config.py`, `test_retry.py`
- Marker: `@pytest.mark.unit` (not extensively used)

**Integration Tests:**
- Scope: Test multiple modules together (API endpoints)
- Mocking: Mock only yt-dlp calls, use real State and FastAPI
- Setup: Use httpx AsyncClient with ASGITransport
- Examples: `test_api.py`
- Marker: `@pytest.mark.integration` (not extensively used)

**Contract Tests:**
- Scope: Verify external library API contracts
- Framework: pytest with network access
- Examples: `test_ytdlp_contract.py`
- Marker: `@pytest.mark.network` (skipped in CI by default)
- Run: `pytest -m network -v`

## Common Patterns

**Async Testing:**
```python
@pytest.mark.asyncio
async def test_async_function() -> None:
    result = await async_function()
    assert result == expected
```

**Error Testing:**
```python
def test_invalid_path():
    with pytest.raises(ValueError, match="Path traversal detected"):
        resolve_task_base_dir(base_output_path="../../../etc", task_id="123")
```

**Parametrized Tests:**
```python
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("0", False),
    ],
)
def test_env_truthy(value: str, expected: bool) -> None:
    assert _env_truthy(value) == expected
```

**API Testing:**
```python
def test_api_endpoint(async_client: httpx.AsyncClient) -> None:
    response = async_client.get("/task/123")
    assert response.status_code == 404  # Not found
```

---

*Testing analysis: 2025-01-17*
*Update when test patterns change*
