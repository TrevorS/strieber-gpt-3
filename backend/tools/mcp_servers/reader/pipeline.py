"""ABOUTME: Pipeline orchestrating web scraping and HTML-to-Markdown conversion.

Coordinates Playwright scraping and ReaderLM-v2 inference to provide
complete HTML→Markdown conversion pipeline with optional instruction-based extraction.
"""

import logging
import time
from typing import Optional

from scraper_client import ScraperClient
from llama_client import LlamaReaderClient

logger = logging.getLogger(__name__)


class ReaderPipeline:
    """Orchestrates web scraping and inference pipeline."""

    def __init__(
        self,
        scraper_endpoint: str = "http://playwright-scraper:8000",
        llama_endpoint: str = "http://llama-server-readerlm:8000"
    ):
        """Initialize pipeline with service endpoints."""
        self.scraper = ScraperClient(scraper_endpoint)
        self.llama = LlamaReaderClient(llama_endpoint)

    async def process_url(
        self,
        url: str,
        instruction: Optional[str] = None,
        timeout: int = 30,
        force_playwright: bool = False,
        max_tokens: int = 8192
    ) -> tuple[str, bool, dict]:
        """
        Complete pipeline: Scrape URL → Convert to Markdown with optional extraction.

        Args:
            url: URL to process
            instruction: Optional instruction for extraction (e.g., "Extract the price")
                        If None, uses default markdown conversion
            timeout: Page load timeout (seconds)
            force_playwright: Force Playwright rendering
            max_tokens: Max output tokens

        Returns:
            (content, success, metadata)
            metadata: {method_used, html_size, scrape_time_ms, inference_time_ms}
        """
        metadata = {
            "url": url,
            "method_used": None,
            "html_size": 0,
            "scrape_time_ms": 0,
            "inference_time_ms": 0
        }

        try:
            # Step 1: Scrape HTML
            logger.info(f"Scraping {url}")
            scrape_start = time.time()

            html, method, success = await self.scraper.scrape(
                url,
                None,  # wait_for_selector
                timeout,
                force_playwright
            )

            scrape_time_ms = int((time.time() - scrape_start) * 1000)
            metadata["method_used"] = method
            metadata["html_size"] = len(html) if html else 0
            metadata["scrape_time_ms"] = scrape_time_ms

            if not success or not html:
                error = f"Failed to scrape {url}: {method} method returned empty HTML"
                logger.error(error)
                return "", False, metadata

            logger.info(f"Scraped {len(html)} bytes from {url} using {method} ({scrape_time_ms}ms)")

            # Step 2: Convert HTML to Markdown (with optional extraction)
            logger.info(f"Converting HTML to Markdown using ReaderLM-v2")
            inference_start = time.time()

            content, conv_success = await self.llama.html_to_markdown(
                html,
                instruction=instruction,
                max_tokens=max_tokens
            )

            inference_time_ms = int((time.time() - inference_start) * 1000)
            metadata["inference_time_ms"] = inference_time_ms

            if not conv_success or not content:
                error = f"Failed to convert HTML to Markdown for {url}"
                logger.error(error)
                return "", False, metadata

            logger.info(f"Successfully converted to {len(content)} chars ({inference_time_ms}ms)")
            return content, True, metadata

        except Exception as e:
            logger.error(f"Pipeline error for {url}: {e}", exc_info=True)
            return "", False, metadata

    async def close(self):
        """Cleanup resources."""
        await self.scraper.close()
        await self.llama.close()
