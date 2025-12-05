"""ABOUTME: HTTP client for Playwright scraper service.

Communicates with the Playwright scraper service to fetch web pages
with optional JavaScript rendering and screenshot capture.

Features:
- Exponential backoff retry for transient failures
- User-agent rotation to avoid detection
- Configurable timeouts and retry limits
"""

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 16.0
BACKOFF_MULTIPLIER = 2.0
JITTER_FACTOR = 0.25  # Add up to 25% random jitter

# HTTP status codes that warrant retry
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

# Common browser user agents for rotation
USER_AGENTS = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Chrome on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Firefox on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Safari on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    # Chrome on Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Firefox on Linux
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
]


def get_random_user_agent() -> str:
    """Get a random user agent from the rotation pool."""
    return random.choice(USER_AGENTS)


def calculate_backoff(attempt: int) -> float:
    """Calculate backoff time with exponential increase and jitter.

    Args:
        attempt: The retry attempt number (0-indexed)

    Returns:
        Backoff time in seconds
    """
    # Exponential backoff: 1s, 2s, 4s, 8s, 16s...
    backoff = min(
        INITIAL_BACKOFF_SECONDS * (BACKOFF_MULTIPLIER**attempt),
        MAX_BACKOFF_SECONDS,
    )

    # Add jitter to prevent thundering herd
    jitter = backoff * JITTER_FACTOR * random.random()
    return backoff + jitter


@dataclass
class ScrapeResult:
    """Result from scraping a page."""

    html: str
    method: str  # "http" or "playwright"
    success: bool
    screenshot: Optional[str] = None  # Base64-encoded screenshot
    retries: int = 0
    user_agent: Optional[str] = None


class ScraperClient:
    """Client for communicating with Playwright scraper service."""

    def __init__(
        self,
        endpoint: str = "http://playwright-scraper:8000",
        max_retries: int = MAX_RETRIES,
        rotate_user_agent: bool = True,
    ):
        """
        Initialize scraper client.

        Args:
            endpoint: URL to the Playwright scraper service
            max_retries: Maximum number of retry attempts for transient failures
            rotate_user_agent: Whether to use random user agents
        """
        self.endpoint = endpoint
        self.max_retries = max_retries
        self.rotate_user_agent = rotate_user_agent
        self.client = httpx.AsyncClient(timeout=120.0)  # Long timeout for slow pages

    def _get_headers(self) -> dict:
        """Get request headers with optional user-agent rotation."""
        headers = {}
        if self.rotate_user_agent:
            headers["X-Custom-User-Agent"] = get_random_user_agent()
        return headers

    async def _execute_with_retry(
        self, url: str, payload: dict
    ) -> Tuple[Optional[dict], int, Optional[str]]:
        """Execute request with exponential backoff retry.

        Args:
            url: Request URL
            payload: JSON payload

        Returns:
            Tuple of (response_data, retry_count, error_message)
        """
        last_error = None
        user_agent = None

        for attempt in range(self.max_retries + 1):
            try:
                headers = self._get_headers()
                user_agent = headers.get("X-Custom-User-Agent")

                response = await self.client.post(
                    url, json=payload, headers=headers
                )

                # Check if we should retry based on status code
                if response.status_code in RETRY_STATUS_CODES:
                    if attempt < self.max_retries:
                        backoff = calculate_backoff(attempt)
                        logger.warning(
                            f"Received {response.status_code}, retrying in {backoff:.1f}s "
                            f"(attempt {attempt + 1}/{self.max_retries + 1})"
                        )
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        last_error = f"Max retries exceeded after {response.status_code}"
                        return None, attempt, last_error

                response.raise_for_status()
                return response.json(), attempt, None

            except httpx.TimeoutException as e:
                last_error = f"Timeout: {e}"
                if attempt < self.max_retries:
                    backoff = calculate_backoff(attempt)
                    logger.warning(
                        f"Request timeout, retrying in {backoff:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await asyncio.sleep(backoff)
                    continue

            except httpx.ConnectError as e:
                last_error = f"Connection error: {e}"
                if attempt < self.max_retries:
                    backoff = calculate_backoff(attempt)
                    logger.warning(
                        f"Connection error, retrying in {backoff:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await asyncio.sleep(backoff)
                    continue

            except httpx.HTTPStatusError as e:
                # Don't retry client errors (4xx except 429)
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    last_error = f"Client error: {e.response.status_code}"
                    return None, attempt, last_error

                last_error = f"HTTP error: {e}"
                if attempt < self.max_retries:
                    backoff = calculate_backoff(attempt)
                    logger.warning(
                        f"HTTP error {e.response.status_code}, retrying in {backoff:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await asyncio.sleep(backoff)
                    continue

            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logger.error(f"Unexpected error during scrape: {e}")
                break

        return None, self.max_retries, last_error

    async def scrape(
        self,
        url: str,
        wait_for_selector: Optional[str] = None,
        timeout: int = 30,
        force_playwright: bool = False,
        capture_screenshot: bool = False,
    ) -> Tuple[str, str, bool]:
        """
        Scrape a web page.

        Args:
            url: URL to scrape
            wait_for_selector: Optional CSS selector to wait for
            timeout: Maximum time for scraping (seconds)
            force_playwright: Force Playwright even for simple pages
            capture_screenshot: Capture full-page screenshot

        Returns:
            (html_content, method_used, success)
            method_used: "http" or "playwright"
            success: True if scraping succeeded
        """
        result = await self.scrape_full(
            url, wait_for_selector, timeout, force_playwright, capture_screenshot
        )
        return result.html, result.method, result.success

    async def scrape_full(
        self,
        url: str,
        wait_for_selector: Optional[str] = None,
        timeout: int = 30,
        force_playwright: bool = False,
        capture_screenshot: bool = False,
    ) -> ScrapeResult:
        """
        Scrape a web page with full result including screenshot.

        Args:
            url: URL to scrape
            wait_for_selector: Optional CSS selector to wait for
            timeout: Maximum time for scraping (seconds)
            force_playwright: Force Playwright even for simple pages
            capture_screenshot: Capture full-page screenshot

        Returns:
            ScrapeResult with html, method, success, and optional screenshot
        """
        payload = {
            "url": url,
            "wait_for_selector": wait_for_selector,
            "timeout": timeout,
            "force_playwright": force_playwright,
            "capture_screenshot": capture_screenshot,
        }

        data, retries, error = await self._execute_with_retry(
            f"{self.endpoint}/scrape", payload
        )

        if data is None:
            logger.error(f"Scraping failed for {url}: {error}")
            return ScrapeResult(
                html="",
                method="unknown",
                success=False,
                retries=retries,
            )

        if data.get("success"):
            return ScrapeResult(
                html=data["html"],
                method=data["method"],
                success=True,
                screenshot=data.get("screenshot"),
                retries=retries,
                user_agent=self._get_headers().get("X-Custom-User-Agent"),
            )
        else:
            error_msg = data.get("error", "Unknown error")
            logger.error(f"Scraping failed for {url}: {error_msg}")
            return ScrapeResult(
                html="",
                method=data.get("method", "unknown"),
                success=False,
                retries=retries,
            )

    async def health_check(self) -> bool:
        """Check if scraper service is healthy."""
        try:
            response = await self.client.get(f"{self.endpoint}/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Scraper health check failed: {e}")
            return False

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
