"""ABOUTME: Shared validation utility module for all MCP tools.

Provides reusable validation logic for URLs, timeouts, string lengths, and other
common input validation patterns. Supports both Pydantic field validators and
standalone validation functions for use outside of Pydantic models.

Design:
- Constants for common validation limits (timeouts, URL schemes, etc.)
- Field validator functions (for @field_validator decorators)
- Standalone validator functions (return tuple[bool, Optional[str]])
- Type-safe with comprehensive type hints
- No external dependencies beyond stdlib and pydantic
"""

from typing import Optional, Tuple
from urllib.parse import urlparse


# =============================================================================
# Validation Constants
# =============================================================================

# URL validation
VALID_URL_SCHEMES: Tuple[str, str] = ("http://", "https://")
MAX_URL_LENGTH: int = 8192
MIN_URL_LENGTH: int = 1

# Timeout validation (seconds)
MIN_TIMEOUT_SECONDS: int = 1
MAX_TIMEOUT_SECONDS: int = 300
DEFAULT_TIMEOUT_SECONDS: int = 10

# String length defaults
DEFAULT_MIN_STRING_LENGTH: int = 1
DEFAULT_MAX_STRING_LENGTH: int = 65536  # 64KB


# =============================================================================
# Pydantic Field Validator Functions (for @field_validator decorators)
# =============================================================================

def validate_url_field(v: str) -> str:
    """Pydantic field validator for URL fields.

    Validates that the URL:
    - Is not empty
    - Starts with http:// or https://
    - Contains a valid domain/netloc
    - Is within length limits

    Args:
        v: URL string to validate

    Returns:
        Validated and stripped URL string

    Raises:
        ValueError: If URL is invalid

    Usage:
        @field_validator("url")
        @classmethod
        def validate_url(cls, v: str) -> str:
            return validate_url_field(v)
    """
    is_valid, error = validate_url(v)
    if not is_valid:
        raise ValueError(error)
    return v.strip()


def validate_timeout_field(
    v: int,
    min_val: int = MIN_TIMEOUT_SECONDS,
    max_val: int = MAX_TIMEOUT_SECONDS
) -> int:
    """Pydantic field validator for timeout fields.

    Validates that the timeout is within acceptable range.

    Args:
        v: Timeout value in seconds to validate
        min_val: Minimum allowed timeout (default: MIN_TIMEOUT_SECONDS)
        max_val: Maximum allowed timeout (default: MAX_TIMEOUT_SECONDS)

    Returns:
        Validated timeout value

    Raises:
        ValueError: If timeout is out of range

    Usage:
        @field_validator("timeout")
        @classmethod
        def validate_timeout(cls, v: int) -> int:
            return validate_timeout_field(v)

        # Or with custom limits:
        @field_validator("timeout")
        @classmethod
        def validate_timeout(cls, v: int) -> int:
            return validate_timeout_field(v, min_val=5, max_val=120)
    """
    is_valid, error = validate_timeout(v, min_val, max_val)
    if not is_valid:
        raise ValueError(error)
    return v


def validate_string_length_field(
    v: str,
    min_length: int = DEFAULT_MIN_STRING_LENGTH,
    max_length: int = DEFAULT_MAX_STRING_LENGTH,
    field_name: str = "value"
) -> str:
    """Pydantic field validator for string length constraints.

    Validates that a string is within acceptable length range.

    Args:
        v: String value to validate
        min_length: Minimum allowed length (default: 1)
        max_length: Maximum allowed length (default: 65536)
        field_name: Name of field for error messages (default: "value")

    Returns:
        Validated and stripped string

    Raises:
        ValueError: If string length is out of range

    Usage:
        @field_validator("prompt")
        @classmethod
        def validate_prompt(cls, v: str) -> str:
            return validate_string_length_field(v, min_length=1, max_length=2000, field_name="prompt")
    """
    is_valid, error = validate_string_length(v, min_length, max_length, field_name)
    if not is_valid:
        raise ValueError(error)
    return v.strip()


def validate_non_empty_string_field(v: str, field_name: str = "field") -> str:
    """Pydantic field validator for non-empty strings.

    Validates that a string is not empty or whitespace-only.

    Args:
        v: String value to validate
        field_name: Name of field for error messages (default: "field")

    Returns:
        Validated and stripped string

    Raises:
        ValueError: If string is empty or whitespace-only

    Usage:
        @field_validator("css_selector")
        @classmethod
        def validate_selector(cls, v: str) -> str:
            return validate_non_empty_string_field(v, field_name="css_selector")
    """
    is_valid, error = validate_non_empty_string(v, field_name)
    if not is_valid:
        raise ValueError(error)
    return v.strip()


# =============================================================================
# Standalone Validator Functions (for non-Pydantic validation)
# =============================================================================

def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """Validate URL format and return (is_valid, error_message).

    Checks that the URL:
    - Is not empty or whitespace-only
    - Starts with http:// or https://
    - Contains a valid domain/netloc
    - Is within length limits

    Args:
        url: URL string to validate

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, error_message) if invalid

    Example:
        is_valid, error = validate_url("https://example.com")
        if not is_valid:
            print(f"Invalid URL: {error}")
    """
    url = url.strip()

    if not url:
        return False, "URL cannot be empty"

    if len(url) < MIN_URL_LENGTH:
        return False, f"URL must be at least {MIN_URL_LENGTH} character(s)"

    if len(url) > MAX_URL_LENGTH:
        return False, f"URL too long (max {MAX_URL_LENGTH} characters, got {len(url)})"

    if not url.startswith(VALID_URL_SCHEMES):
        schemes_str = " or ".join(VALID_URL_SCHEMES)
        return False, f"URL must start with {schemes_str}"

    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return False, "URL must contain a valid domain name"

        # Check for invalid characters in domain
        if any(char in parsed.netloc for char in [' ', '\t', '\n', '\r']):
            return False, "URL domain contains invalid whitespace characters"

    except Exception as e:
        return False, f"Invalid URL format: {str(e)}"

    return True, None


def validate_timeout(
    timeout: int,
    min_val: int = MIN_TIMEOUT_SECONDS,
    max_val: int = MAX_TIMEOUT_SECONDS
) -> Tuple[bool, Optional[str]]:
    """Validate timeout value and return (is_valid, error_message).

    Checks that the timeout is within acceptable range.

    Args:
        timeout: Timeout value in seconds to validate
        min_val: Minimum allowed timeout (default: MIN_TIMEOUT_SECONDS)
        max_val: Maximum allowed timeout (default: MAX_TIMEOUT_SECONDS)

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, error_message) if invalid

    Example:
        is_valid, error = validate_timeout(30, min_val=1, max_val=120)
        if not is_valid:
            print(f"Invalid timeout: {error}")
    """
    if not isinstance(timeout, int):
        return False, f"Timeout must be an integer, got {type(timeout).__name__}"

    if timeout < min_val:
        return False, f"Timeout must be at least {min_val} second(s), got {timeout}"

    if timeout > max_val:
        return False, f"Timeout must be at most {max_val} second(s), got {timeout}"

    return True, None


def validate_string_length(
    value: str,
    min_length: int,
    max_length: int,
    field_name: str = "value"
) -> Tuple[bool, Optional[str]]:
    """Validate string length and return (is_valid, error_message).

    Checks that a string is within acceptable length range.

    Args:
        value: String value to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        field_name: Name of field for error messages (default: "value")

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, error_message) if invalid

    Example:
        is_valid, error = validate_string_length(prompt, 1, 2000, "prompt")
        if not is_valid:
            print(f"Invalid prompt: {error}")
    """
    if not isinstance(value, str):
        return False, f"{field_name} must be a string, got {type(value).__name__}"

    value_stripped = value.strip()
    value_len = len(value_stripped)

    if value_len < min_length:
        return False, f"{field_name} must be at least {min_length} character(s), got {value_len}"

    if value_len > max_length:
        return False, f"{field_name} too long (max {max_length} characters, got {value_len})"

    return True, None


def validate_non_empty_string(
    value: str,
    field_name: str = "field"
) -> Tuple[bool, Optional[str]]:
    """Validate that string is not empty and return (is_valid, error_message).

    Checks that a string is not empty or whitespace-only.

    Args:
        value: String value to validate
        field_name: Name of field for error messages (default: "field")

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, error_message) if invalid

    Example:
        is_valid, error = validate_non_empty_string(selector, "css_selector")
        if not is_valid:
            print(f"Invalid selector: {error}")
    """
    if not isinstance(value, str):
        return False, f"{field_name} must be a string, got {type(value).__name__}"

    if not value.strip():
        return False, f"{field_name} cannot be empty or whitespace-only"

    return True, None


# =============================================================================
# Additional Validation Helpers
# =============================================================================

def validate_positive_integer(
    value: int,
    field_name: str = "value"
) -> Tuple[bool, Optional[str]]:
    """Validate that value is a positive integer.

    Args:
        value: Integer value to validate
        field_name: Name of field for error messages (default: "value")

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, error_message) if invalid

    Example:
        is_valid, error = validate_positive_integer(count, "max_results")
        if not is_valid:
            print(f"Invalid count: {error}")
    """
    if not isinstance(value, int):
        return False, f"{field_name} must be an integer, got {type(value).__name__}"

    if value <= 0:
        return False, f"{field_name} must be positive, got {value}"

    return True, None


def validate_integer_range(
    value: int,
    min_val: int,
    max_val: int,
    field_name: str = "value"
) -> Tuple[bool, Optional[str]]:
    """Validate that integer is within range.

    Args:
        value: Integer value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        field_name: Name of field for error messages (default: "value")

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, error_message) if invalid

    Example:
        is_valid, error = validate_integer_range(page, 1, 100, "page_number")
        if not is_valid:
            print(f"Invalid page: {error}")
    """
    if not isinstance(value, int):
        return False, f"{field_name} must be an integer, got {type(value).__name__}"

    if value < min_val or value > max_val:
        return False, f"{field_name} must be between {min_val} and {max_val}, got {value}"

    return True, None
