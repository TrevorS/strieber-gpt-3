"""ABOUTME: Pipeline orchestrating web scraping and HTML-to-Markdown conversion.

Coordinates Playwright scraping and ReaderLM-v2 inference to provide
complete HTML→Markdown conversion pipeline.
"""

import logging
from typing import Optional

from scraper_client import ScraperClient
from llama_client import LlamaReaderClient

logger = logging.getLogger(__name__)


class LocalReaderPipeline:
    """Orchestrates web scraping and inference pipeline."""

    def __init__(
        self,
        scraper_endpoint: str = "http://playwright-scraper:8000",
        llama_endpoint: str = "http://llama-server-readerlm:8001"
    ):
        """Initialize pipeline with service endpoints."""
        self.scraper = ScraperClient(scraper_endpoint)
        self.llama = LlamaReaderClient(llama_endpoint)

    async def process_url(
        self,
        url: str,
        wait_for_selector: Optional[str] = None,
        timeout: int = 30,
        force_playwright: bool = False,
        max_tokens: int = 8192
    ) -> tuple[str, bool, dict]:
        """
        Complete pipeline: Scrape URL → Convert to Markdown.

        Args:
            url: URL to process
            wait_for_selector: Optional CSS selector to wait for
            timeout: Page load timeout (seconds)
            force_playwright: Force Playwright rendering
            max_tokens: Max output tokens

        Returns:
            (markdown, success, metadata)
            metadata: {method_used, processing_steps}
        """
        metadata = {
            "url": url,
            "steps": [],
            "method_used": None
        }

        try:
            # Step 1: Scrape HTML
            logger.info(f"Scraping {url}")
            html, method, success = await self.scraper.scrape(
                url,
                wait_for_selector,
                timeout,
                force_playwright
            )
            metadata["method_used"] = method
            metadata["steps"].append(f"scraped_{method}")

            if not success or not html:
                error = f"Failed to scrape {url}: {method} method returned empty HTML"
                logger.error(error)
                return "", False, metadata

            logger.info(f"Scraped {len(html)} bytes from {url} using {method}")
            metadata["steps"].append(f"html_size_{len(html)}")

            # Step 2: Convert HTML to Markdown
            logger.info(f"Converting HTML to Markdown using ReaderLM-v2")
            markdown, conv_success = await self.llama.html_to_markdown(
                html,
                max_tokens=max_tokens
            )
            metadata["steps"].append("markdown_converted")

            if not conv_success or not markdown:
                error = f"Failed to convert HTML to Markdown for {url}"
                logger.error(error)
                return "", False, metadata

            logger.info(f"Successfully converted to {len(markdown)} chars of markdown")
            metadata["steps"].append(f"markdown_size_{len(markdown)}")

            return markdown, True, metadata

        except Exception as e:
            logger.error(f"Pipeline error for {url}: {e}", exc_info=True)
            metadata["steps"].append(f"error_{str(e)}")
            return "", False, metadata

    async def process_url_with_selector(
        self,
        url: str,
        css_selector: str,
        timeout: int = 30,
        max_tokens: int = 8192
    ) -> tuple[str, bool, dict]:
        """Process URL with CSS selector targeting."""
        return await self.process_url(
            url,
            wait_for_selector=css_selector,
            timeout=timeout,
            force_playwright=True,
            max_tokens=max_tokens
        )

    async def close(self):
        """Cleanup resources."""
        await self.scraper.close()
        await self.llama.close()
