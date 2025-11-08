"""ABOUTME: HTTP client utilities for MCP tools - async HTTP operations with standard error handling."""

from typing import Optional, Dict, Any
import httpx

from .error_handling import HTTPStatusCodes

# Constants
DEFAULT_HTTP_TIMEOUT = 10.0
MIN_HTTP_TIMEOUT = 1.0
MAX_HTTP_TIMEOUT = 300.0


async def safe_http_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
    follow_redirects: bool = True
) -> httpx.Response:
    """Perform async HTTP GET with standard error handling."""
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=follow_redirects) as client:
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response


def interpret_http_error(status_code: int) -> str:
    """Map HTTP status code to MCP error code."""
    if status_code == 429:
        return "rate_limited"
    elif status_code in (401, 403):
        return "invalid_url"
    elif status_code == 404:
        return "not_found"
    elif status_code >= 500:
        return "fetch_failed"
    else:
        return "network_error"


def get_retry_after_seconds(response: httpx.Response) -> Optional[int]:
    """Extract Retry-After header from response."""
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return int(retry_after)
        except ValueError:
            return None
    return None


__all__ = [
    "DEFAULT_HTTP_TIMEOUT",
    "MIN_HTTP_TIMEOUT",
    "MAX_HTTP_TIMEOUT",
    "safe_http_get",
    "interpret_http_error",
    "get_retry_after_seconds",
    "HTTPStatusCodes",
]
