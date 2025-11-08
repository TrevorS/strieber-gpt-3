"""ABOUTME: Shared error handling utilities for all MCP tools.

Provides standardized error codes, error result creation functions, and HTTP status
code helpers to ensure consistent error handling across all MCP servers.
"""

from typing import Optional, Dict, Any
from mcp.types import TextContent, CallToolResult


# =============================================================================
# Error Code Constants
# =============================================================================

# Input validation errors
ERROR_INVALID_INPUT: str = "invalid_input"
ERROR_INVALID_URL: str = "invalid_url"
ERROR_INVALID_PARAMETER: str = "invalid_parameter"
ERROR_VALIDATION_FAILED: str = "validation_failed"

# Network and fetch errors
ERROR_TIMEOUT: str = "timeout"
ERROR_FETCH_FAILED: str = "fetch_failed"
ERROR_NETWORK_ERROR: str = "network_error"

# Processing and extraction errors
ERROR_EXTRACTION_FAILED: str = "extraction_failed"
ERROR_JS_RENDERING_FAILED: str = "js_rendering_failed"

# Resource errors
ERROR_NOT_FOUND: str = "not_found"
ERROR_RATE_LIMITED: str = "rate_limited"

# General errors
ERROR_UNEXPECTED: str = "unexpected_error"


# =============================================================================
# HTTP Status Code Helpers
# =============================================================================

class HTTPStatusCodes:
    """Helper methods for HTTP status code checks.

    Provides semantic methods to check HTTP status codes instead of
    hardcoding numeric values throughout the codebase.
    """

    @staticmethod
    def is_rate_limit(status_code: int) -> bool:
        """Check if status code indicates rate limiting.

        Args:
            status_code: HTTP status code

        Returns:
            True if status code is 429 (Too Many Requests)

        Example:
            if HTTPStatusCodes.is_rate_limit(response.status):
                return create_rate_limit_error()
        """
        return status_code == 429

    @staticmethod
    def is_auth_error(status_code: int) -> bool:
        """Check if status code indicates authentication/authorization error.

        Args:
            status_code: HTTP status code

        Returns:
            True if status code is 401 (Unauthorized) or 403 (Forbidden)

        Example:
            if HTTPStatusCodes.is_auth_error(response.status):
                logger.error("Authentication failed")
        """
        return status_code in (401, 403)

    @staticmethod
    def is_not_found(status_code: int) -> bool:
        """Check if status code indicates resource not found.

        Args:
            status_code: HTTP status code

        Returns:
            True if status code is 404 (Not Found)

        Example:
            if HTTPStatusCodes.is_not_found(response.status):
                return create_not_found_error("page", url)
        """
        return status_code == 404

    @staticmethod
    def is_client_error(status_code: int) -> bool:
        """Check if status code indicates client error (4xx).

        Args:
            status_code: HTTP status code

        Returns:
            True if status code is in range 400-499

        Example:
            if HTTPStatusCodes.is_client_error(response.status):
                logger.warning(f"Client error: {response.status}")
        """
        return 400 <= status_code < 500

    @staticmethod
    def is_server_error(status_code: int) -> bool:
        """Check if status code indicates server error (5xx).

        Args:
            status_code: HTTP status code

        Returns:
            True if status code is in range 500-599

        Example:
            if HTTPStatusCodes.is_server_error(response.status):
                logger.error(f"Server error: {response.status}")
        """
        return 500 <= status_code < 600


# =============================================================================
# Main Error Creation Function
# =============================================================================

def create_error_result(
    error_message: str,
    error_code: str,
    error_type: str = "error",
    additional_metadata: Optional[Dict[str, Any]] = None
) -> CallToolResult:
    """Create standardized error CallToolResult.

    This is the main error creation function used by all MCP tools.
    Use the convenience wrapper functions below for common error types.

    Args:
        error_message: Human-readable error message for users and LLMs
        error_code: Machine-readable error code (use ERROR_* constants)
        error_type: Error category/type (e.g., "validation_error", "network_error")
        additional_metadata: Additional context for debugging (optional)

    Returns:
        CallToolResult with standardized error format

    Example:
        result = create_error_result(
            error_message="URL is invalid",
            error_code=ERROR_INVALID_URL,
            error_type="validation_error",
            additional_metadata={"url": "not-a-url"}
        )
    """
    metadata = {
        "error_type": error_type,
        "error_code": error_code,
    }

    if additional_metadata:
        metadata.update(additional_metadata)

    return CallToolResult(
        content=[TextContent(type="text", text=f"Error: {error_message}")],
        isError=True,
        metadata=metadata
    )


# =============================================================================
# Convenience Wrapper Functions
# =============================================================================

def create_validation_error(
    field_name: str,
    error_message: str,
    field_value: Any = None
) -> CallToolResult:
    """Create a validation error for invalid input fields.

    Args:
        field_name: Name of the field that failed validation
        error_message: Human-readable description of the validation failure
        field_value: The invalid value that was provided (optional, for debugging)

    Returns:
        CallToolResult with validation error

    Example:
        return create_validation_error(
            field_name="timeout",
            error_message="Timeout must be between 5 and 300 seconds",
            field_value=timeout
        )
    """
    metadata = {"field_name": field_name}
    if field_value is not None:
        metadata["field_value"] = field_value

    return create_error_result(
        error_message=f"{field_name}: {error_message}",
        error_code=ERROR_VALIDATION_FAILED,
        error_type="validation_error",
        additional_metadata=metadata
    )


def create_network_error(
    url: str,
    error_message: str,
    timeout_seconds: Optional[float] = None,
    status_code: Optional[int] = None
) -> CallToolResult:
    """Create a network error for failed HTTP requests.

    Args:
        url: The URL that failed
        error_message: Human-readable description of the network failure
        timeout_seconds: Timeout duration if applicable (optional)
        status_code: HTTP status code if applicable (optional)

    Returns:
        CallToolResult with network error

    Example:
        return create_network_error(
            url="https://example.com",
            error_message="Connection refused",
            timeout_seconds=30
        )
    """
    metadata = {"url": url}

    if timeout_seconds is not None:
        metadata["timeout_seconds"] = timeout_seconds

    if status_code is not None:
        metadata["status_code"] = status_code

    return create_error_result(
        error_message=f"Network error fetching {url}: {error_message}",
        error_code=ERROR_NETWORK_ERROR,
        error_type="network_error",
        additional_metadata=metadata
    )


def create_timeout_error(
    timeout_seconds: float,
    context: str = ""
) -> CallToolResult:
    """Create a timeout error for operations that exceeded time limits.

    Args:
        timeout_seconds: The timeout duration that was exceeded
        context: Additional context about what timed out (optional)

    Returns:
        CallToolResult with timeout error

    Example:
        return create_timeout_error(
            timeout_seconds=30,
            context="page load"
        )
    """
    message = f"Operation timed out after {timeout_seconds} seconds"
    if context:
        message = f"{context.capitalize()} timed out after {timeout_seconds} seconds"

    return create_error_result(
        error_message=message,
        error_code=ERROR_TIMEOUT,
        error_type="timeout_error",
        additional_metadata={
            "timeout_seconds": timeout_seconds,
            "context": context
        }
    )


def create_not_found_error(
    resource_type: str,
    resource_id: str
) -> CallToolResult:
    """Create a not found error for missing resources.

    Args:
        resource_type: Type of resource that was not found (e.g., "page", "user", "file")
        resource_id: Identifier of the resource (e.g., URL, username, filename)

    Returns:
        CallToolResult with not found error

    Example:
        return create_not_found_error(
            resource_type="page",
            resource_id="https://example.com/404"
        )
    """
    return create_error_result(
        error_message=f"{resource_type.capitalize()} not found: {resource_id}",
        error_code=ERROR_NOT_FOUND,
        error_type="not_found_error",
        additional_metadata={
            "resource_type": resource_type,
            "resource_id": resource_id
        }
    )


def create_rate_limit_error(
    retry_after_seconds: Optional[int] = None,
    limit_description: Optional[str] = None
) -> CallToolResult:
    """Create a rate limit error for throttled requests.

    Args:
        retry_after_seconds: Seconds to wait before retrying (optional)
        limit_description: Description of the rate limit (optional)

    Returns:
        CallToolResult with rate limit error

    Example:
        return create_rate_limit_error(
            retry_after_seconds=60,
            limit_description="100 requests per minute"
        )
    """
    message = "Rate limit exceeded"

    if retry_after_seconds is not None:
        message += f". Retry after {retry_after_seconds} seconds"

    if limit_description:
        message += f". Limit: {limit_description}"

    metadata = {}
    if retry_after_seconds is not None:
        metadata["retry_after_seconds"] = retry_after_seconds
    if limit_description:
        metadata["limit_description"] = limit_description

    return create_error_result(
        error_message=message,
        error_code=ERROR_RATE_LIMITED,
        error_type="rate_limit_error",
        additional_metadata=metadata if metadata else None
    )
