"""ABOUTME: Pipeline orchestrating web scraping and HTML-to-Markdown conversion.

Provides two conversion paths:
1. Fast path (default): Trafilatura + markdownify (~100-200ms, no LLM)
2. LLM path (opt-in): Trafilatura + ReaderLM-v2 (~2-3s, higher quality)

The fast path handles 90%+ of pages well. Use LLM path for complex/broken HTML.
"""

import logging
import time
from typing import Optional

from scraper_client import ScraperClient
from llama_client import LlamaReaderClient
from html_preprocessor import (
    extract_and_convert_to_markdown,
    preprocess_html
)

logger = logging.getLogger(__name__)


class ReaderPipeline:
    """Orchestrates web scraping and conversion pipeline."""

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
        max_tokens: int = 8192,
        use_llm: bool = False
    ) -> tuple[str, bool, dict]:
        """
        Complete pipeline: Scrape URL → Extract content → Convert to Markdown.

        Two paths available:
        - Fast path (default): Trafilatura + markdownify (~100-200ms)
        - LLM path (use_llm=True): Trafilatura + ReaderLM-v2 (~2-3s)

        Args:
            url: URL to process
            instruction: Optional instruction for LLM extraction (only used with use_llm=True)
            timeout: Operation timeout (seconds)
            force_playwright: Force Playwright rendering
            max_tokens: Max output tokens (only used with use_llm=True)
            use_llm: Use ReaderLM-v2 for conversion (default False - uses fast path)

        Returns:
            (content, success, metadata)
            metadata: {method_used, html_size, scrape_time_ms, inference_time_ms, preprocessing, pipeline}
        """
        metadata = {
            "url": url,
            "method_used": None,
            "html_size": 0,
            "scrape_time_ms": 0,
            "inference_time_ms": 0,
            "pipeline": "fast_path" if not use_llm else "llm_path"
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

            # Step 2 & 3: Extract content and convert to Markdown
            if use_llm:
                # LLM Path: Trafilatura extraction → ReaderLM-v2 conversion
                content, conv_success, conv_metadata = await self._convert_with_llm(
                    html, instruction, max_tokens, timeout
                )
            else:
                # Fast Path: Trafilatura extraction → markdownify conversion
                content, conv_success, conv_metadata = self._convert_fast(html)

            metadata["conversion"] = conv_metadata
            metadata["inference_time_ms"] = conv_metadata.get("inference_time_ms", 0)

            if not conv_success or not content:
                error = f"Failed to convert HTML to Markdown for {url}"
                logger.error(error)
                return "", False, metadata

            logger.info(
                f"Successfully converted to {len(content)} chars "
                f"({metadata['pipeline']}, {metadata.get('inference_time_ms', 0)}ms)"
            )
            return content, True, metadata

        except Exception as e:
            logger.error(f"Pipeline error for {url}: {e}", exc_info=True)
            return "", False, metadata

    def _convert_fast(self, html: str) -> tuple[str, bool, dict]:
        """Fast path: Trafilatura + markdownify (no LLM).

        Args:
            html: Raw HTML content

        Returns:
            (markdown, success, metadata)
        """
        start_time = time.time()

        content, success, metadata = extract_and_convert_to_markdown(html)

        elapsed_ms = int((time.time() - start_time) * 1000)
        metadata["inference_time_ms"] = elapsed_ms  # For consistency with LLM path
        metadata["pipeline"] = "fast_path"

        logger.info(f"Fast path conversion completed in {elapsed_ms}ms")

        return content, success, metadata

    async def _convert_with_llm(
        self,
        html: str,
        instruction: Optional[str],
        max_tokens: int,
        timeout: int
    ) -> tuple[str, bool, dict]:
        """LLM path: Trafilatura extraction → ReaderLM-v2 conversion.

        Args:
            html: Raw HTML content
            instruction: Optional extraction instruction
            max_tokens: Max output tokens
            timeout: Timeout in seconds

        Returns:
            (markdown, success, metadata)
        """
        metadata = {
            "pipeline": "llm_path",
            "inference_time_ms": 0
        }

        # Step 1: Extract with Trafilatura (output as HTML for ReaderLM)
        logger.info("Preprocessing HTML with Trafilatura for ReaderLM-v2")
        html_processed, preprocess_metadata = preprocess_html(html, use_trafilatura=True, output_format="html")
        metadata["preprocessing"] = preprocess_metadata

        # Step 2: Convert with ReaderLM-v2
        logger.info("Converting HTML to Markdown using ReaderLM-v2")
        inference_start = time.time()

        content, conv_success = await self.llama.html_to_markdown(
            html_processed,
            instruction=instruction,
            max_tokens=max_tokens,
            timeout=timeout
        )

        inference_time_ms = int((time.time() - inference_start) * 1000)
        metadata["inference_time_ms"] = inference_time_ms

        return content, conv_success, metadata

    async def close(self):
        """Cleanup resources."""
        await self.scraper.close()
        await self.llama.close()
