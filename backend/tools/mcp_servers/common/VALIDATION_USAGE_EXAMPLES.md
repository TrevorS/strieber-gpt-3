# Validation Module Usage Examples

This document shows how to use the shared `validation.py` module to replace duplicate validation logic across MCP tools.

## Basic Imports

```python
from common.validation import (
    # Constants
    VALID_URL_SCHEMES,
    MAX_URL_LENGTH,
    MIN_TIMEOUT_SECONDS,
    MAX_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,

    # Pydantic field validators
    validate_url_field,
    validate_timeout_field,
    validate_string_length_field,
    validate_non_empty_string_field,

    # Standalone validators
    validate_url,
    validate_timeout,
    validate_string_length,
    validate_non_empty_string,
)
```

## Example 1: Using Pydantic Field Validators

**Before** (duplicated in jina_reader.py, reader/server.py, etc.):

```python
class FetchPageInput(BaseModel):
    url: str = Field(..., description="URL to fetch")
    timeout: int = Field(default=10, ge=1, le=120)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("URL cannot be empty")
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        try:
            parsed = urlparse(v)
            if not parsed.netloc:
                raise ValueError("URL must include a domain name")
        except Exception as e:
            raise ValueError(f"Invalid URL format: {str(e)}")
        return v
```

**After** (using shared validation):

```python
from common.validation import validate_url_field, validate_timeout_field, MAX_URL_LENGTH

class FetchPageInput(BaseModel):
    url: str = Field(
        ...,
        description="URL to fetch",
        max_length=MAX_URL_LENGTH
    )
    timeout: int = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        ge=MIN_TIMEOUT_SECONDS,
        le=MAX_TIMEOUT_SECONDS
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        return validate_url_field(v)

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        return validate_timeout_field(v)
```

## Example 2: Using Standalone Validators

**Use Case**: Validation outside of Pydantic models (e.g., in helper functions)

```python
from common.validation import validate_url, validate_timeout

def process_request(url: str, timeout: int) -> tuple[bool, str]:
    """Process a request with validation."""

    # Validate URL
    is_valid, error = validate_url(url)
    if not is_valid:
        logger.error(f"Invalid URL: {error}")
        return False, f"Validation error: {error}"

    # Validate timeout with custom limits
    is_valid, error = validate_timeout(timeout, min_val=5, max_val=120)
    if not is_valid:
        logger.error(f"Invalid timeout: {error}")
        return False, f"Validation error: {error}"

    # Proceed with processing...
    return True, "Success"
```

## Example 3: String Length Validation

```python
from common.validation import validate_string_length_field

class SearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    css_selector: str = Field(default=None, description="CSS selector")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        return validate_string_length_field(
            v,
            min_length=1,
            max_length=500,
            field_name="query"
        )

    @field_validator("css_selector")
    @classmethod
    def validate_selector(cls, v: str) -> str:
        if v is None:
            return v
        return validate_string_length_field(
            v,
            min_length=1,
            max_length=512,
            field_name="css_selector"
        )
```

## Example 4: Non-Empty String Validation

```python
from common.validation import validate_non_empty_string_field

class ConfigInput(BaseModel):
    api_key: str = Field(..., description="API key")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        return validate_non_empty_string_field(v, field_name="api_key")
```

## Example 5: Using Constants

```python
from common.validation import (
    VALID_URL_SCHEMES,
    MAX_URL_LENGTH,
    DEFAULT_TIMEOUT_SECONDS
)

# Use constants for consistency
JINA_API_BASE_URL = "https://r.jina.ai"
DEFAULT_REQUEST_TIMEOUT = DEFAULT_TIMEOUT_SECONDS

def build_url(path: str) -> str:
    """Build URL ensuring it uses valid schemes."""
    if not path.startswith(VALID_URL_SCHEMES):
        raise ValueError(f"Path must start with {' or '.join(VALID_URL_SCHEMES)}")
    return path
```

## Benefits

1. **DRY (Don't Repeat Yourself)**: Eliminate duplicate validation logic across 3+ files
2. **Consistency**: All tools use the same validation rules
3. **Maintainability**: Update validation logic in one place
4. **Type Safety**: Full type hints for better IDE support
5. **Error Messages**: Consistent, clear error messages
6. **Testability**: Shared validators can be tested once, thoroughly

## Migration Checklist

When refactoring existing code to use shared validation:

- [ ] Import validation functions from `common.validation`
- [ ] Replace duplicated validator methods with calls to shared validators
- [ ] Use shared constants instead of local definitions
- [ ] Remove old validation code
- [ ] Test thoroughly to ensure behavior is unchanged
- [ ] Update any error handling that depends on specific error messages
