"""ABOUTME: HTTP client for Playwright scraper service.

Communicates with the Playwright scraper service to fetch web pages
with optional JavaScript rendering.
"""

import logging
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class ScraperClient:
    """Client for communicating with Playwright scraper service."""

    def __init__(self, endpoint: str = "http://playwright-scraper:8000"):
        """
        Initialize scraper client.

        Args:
            endpoint: URL to the Playwright scraper service
        """
        self.endpoint = endpoint
        self.client = httpx.AsyncClient(timeout=120.0)  # Long timeout for slow pages

    async def scrape(
        self,
        url: str,
        wait_for_selector: Optional[str] = None,
        timeout: int = 30,
        force_playwright: bool = False
    ) -> Tuple[str, str, bool]:
        """
        Scrape a web page.

        Args:
            url: URL to scrape
            wait_for_selector: Optional CSS selector to wait for
            timeout: Maximum time for scraping (seconds)
            force_playwright: Force Playwright even for simple pages

        Returns:
            (html_content, method_used, success)
            method_used: "http" or "playwright"
            success: True if scraping succeeded
        """
        try:
            response = await self.client.post(
                f"{self.endpoint}/scrape",
                json={
                    "url": url,
                    "wait_for_selector": wait_for_selector,
                    "timeout": timeout,
                    "force_playwright": force_playwright
                }
            )
            response.raise_for_status()
            data = response.json()

            if data.get("success"):
                return data["html"], data["method"], True
            else:
                error_msg = data.get("error", "Unknown error")
                logger.error(f"Scraping failed for {url}: {error_msg}")
                return "", data.get("method", "unknown"), False

        except httpx.TimeoutException:
            logger.error(f"Scraper timeout for {url}")
            return "", "unknown", False
        except Exception as e:
            logger.error(f"Scraper error for {url}: {e}")
            return "", "unknown", False

    async def health_check(self) -> bool:
        """Check if scraper service is healthy."""
        try:
            response = await self.client.get(
                f"{self.endpoint}/health",
                timeout=5.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Scraper health check failed: {e}")
            return False

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
