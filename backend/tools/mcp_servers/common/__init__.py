"""ABOUTME: Common MCP server utilities and shared infrastructure."""

from .mcp_base import MCPServerBase
from .error_handling import (
    # Error code constants
    ERROR_INVALID_INPUT,
    ERROR_INVALID_URL,
    ERROR_INVALID_PARAMETER,
    ERROR_VALIDATION_FAILED,
    ERROR_TIMEOUT,
    ERROR_FETCH_FAILED,
    ERROR_NETWORK_ERROR,
    ERROR_NOT_FOUND,
    ERROR_RATE_LIMITED,
    ERROR_UNEXPECTED,
    # HTTP status code helpers
    HTTPStatusCodes,
    # Error creation functions
    create_error_result,
    create_validation_error,
    create_network_error,
    create_timeout_error,
    create_not_found_error,
    create_rate_limit_error,
)

__all__ = [
    "MCPServerBase",
    # Error code constants
    "ERROR_INVALID_INPUT",
    "ERROR_INVALID_URL",
    "ERROR_INVALID_PARAMETER",
    "ERROR_VALIDATION_FAILED",
    "ERROR_TIMEOUT",
    "ERROR_FETCH_FAILED",
    "ERROR_NETWORK_ERROR",
    "ERROR_NOT_FOUND",
    "ERROR_RATE_LIMITED",
    "ERROR_UNEXPECTED",
    # HTTP status code helpers
    "HTTPStatusCodes",
    # Error creation functions
    "create_error_result",
    "create_validation_error",
    "create_network_error",
    "create_timeout_error",
    "create_not_found_error",
    "create_rate_limit_error",
]
