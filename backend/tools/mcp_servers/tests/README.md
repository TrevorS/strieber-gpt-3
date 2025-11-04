# MCP Server Test Suite

Comprehensive test suite for MCP (Model Context Protocol) servers with pytest.

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test File

```bash
pytest tests/test_search_backend.py
pytest tests/test_search_utils.py
```

### Run with Verbose Output

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=common
```

## Test Structure

### conftest.py
- **Purpose**: Pytest configuration and shared fixtures
- **Fixtures**:
  - `mock_httpx_client`: Mocked AsyncClient for HTTP requests
  - `mock_search_result`: Sample search result data
  - `mock_weather_data`: Sample weather API response
  - `mock_openai_client`: Mocked OpenAI async client
  - `mock_docker_client`: Mocked Docker client

### test_search_backend.py
- **Tests SearchResult data class**:
  - Creation and field access
  - Text concatenation (snippet + extra_snippets)
  - Token estimation

- **Tests SearchResponse data class**:
  - Creation with results
  - Empty result handling

- **Tests Backend Factory**:
  - Backend instantiation from type parameter
  - Default backend selection
  - Environment variable reading
  - Error handling for unknown backends
  - API key validation

### test_search_utils.py
- **Tests Domain Extraction**:
  - Simple URL parsing
  - www prefix removal
  - Subdomain handling
  - Invalid URL handling

- **Tests Quality Filtering**:
  - Short snippet removal
  - Missing field handling
  - Empty snippet filtering

- **Tests Domain Deduplication**:
  - Per-domain result limits
  - Order preservation

- **Tests URL Deduplication**:
  - Duplicate URL removal
  - First occurrence preservation

- **Tests Markdown Formatting**:
  - Basic formatting
  - Metadata inclusion/exclusion
  - Empty results handling

- **Tests Result Condensing**:
  - Budget-aware trimming
  - Token estimation
  - Snippet abbreviation

- **Tests Combined Filtering**:
  - Multiple filter application
  - Quality + deduplication

## Adding New Tests

1. **Create test file**: `tests/test_<feature>.py`
2. **Import fixtures**: From `conftest.py`
3. **Organize tests**: Use test classes for logical grouping
4. **Use descriptive names**: `test_<functionality>_<scenario>()`
5. **Document with docstrings**: Explain what each test validates

### Example Test

```python
def test_my_feature(mock_httpx_client):
    """Test my feature with mocked HTTP client."""
    # Arrange
    mock_httpx_client.get = AsyncMock(return_value=MagicMock(json=lambda: {"key": "value"}))

    # Act
    result = await my_function(mock_httpx_client)

    # Assert
    assert result["key"] == "value"
```

## Mocking External APIs

All tests use mocked HTTP clients to avoid requiring real API keys or network access:

- **Brave Search API**: Mocked in search backend tests
- **Open-Meteo API**: Mocked weather data
- **OpenAI API**: Mocked LLM responses
- **Docker**: Mocked container operations

## Running Tests in Docker

```bash
# Build image with test dependencies
docker build -f Dockerfile.mcp-server --build-arg SERVER_MODULE=weather \
  -t strieber-mcp-weather:latest .

# Run tests inside container
docker run --rm strieber-mcp-weather:latest pytest /app/tests/
```

## Test Coverage Goals

- **Infrastructure** (common/): 80%+ coverage
- **Search utilities**: 85%+ coverage
- **Backend implementations**: 70%+ coverage (integration tested separately)

## Known Limitations

- Tests use mocks, not real API calls
- Integration tests (calling actual APIs) run separately
- Docker container tests require running containers
- Some async patterns may require `pytest-asyncio` fixtures

## Troubleshooting

### ModuleNotFoundError: No module named 'common'
**Solution**: Ensure `conftest.py` and test files are in the `tests/` directory, and run pytest from the repository root or with Python path correctly set.

### FixtureLookupError
**Solution**: Ensure you're importing from `conftest.py` via pytest's fixture auto-discovery. Don't import fixtures directly.

### asyncio Event Loop Errors
**Solution**: Use `pytest-asyncio` and mark async tests with `@pytest.mark.asyncio` decorator.

## CI/CD Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run MCP Server Tests
  run: pytest backend/tools/mcp_servers/tests/ -v --cov
```

## Future Improvements

- [ ] Integration tests with real APIs (in staging)
- [ ] Performance benchmarks
- [ ] Fuzz testing for input validation
- [ ] Load testing for concurrent requests
- [ ] Security testing for injection vulnerabilities
