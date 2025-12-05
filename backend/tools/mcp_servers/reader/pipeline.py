"""ABOUTME: Pipeline orchestrating web scraping and HTML-to-Markdown conversion.

Provides two conversion paths:
1. Fast path (default): Trafilatura + markdownify (~100-200ms, no LLM)
2. LLM path (opt-in): Trafilatura + ReaderLM-v2 (~2-3s, higher quality)

Additional features:
- Multiple output formats: markdown, html, text, screenshot
- Metadata extraction: title, author, date, language, sitename
- Links extraction: internal/external URLs
- BM25 query filtering: extract only content relevant to a query
- Wait for CSS selector before scraping
"""

import logging
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

from scraper_client import ScraperClient, ScrapeResult
from llama_client import LlamaReaderClient
from html_preprocessor import (
    extract_and_convert_to_markdown,
    preprocess_html,
    extract_metadata,
    extract_links,
    filter_content_by_query,
    extract_with_trafilatura,
    PageMetadata,
    ExtractedLink,
    ExtractionOptions,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    """Full result from processing a URL."""
    content: str
    success: bool
    metadata: Dict[str, Any]
    page_metadata: Optional[Dict] = None
    links: Optional[List[Dict]] = None
    screenshot: Optional[str] = None

    def to_tuple(self):
        """Convert to legacy tuple format (content, success, metadata)."""
        return self.content, self.success, self.metadata


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
        Legacy method: Complete pipeline with basic options.

        For full feature access, use process_url_full() instead.
        """
        result = await self.process_url_full(
            url=url,
            timeout=timeout,
            force_playwright=force_playwright,
            use_llm=use_llm,
            output_format="markdown"
        )
        return result.to_tuple()

    async def process_url_full(
        self,
        url: str,
        timeout: int = 30,
        force_playwright: bool = False,
        use_llm: bool = False,
        output_format: str = "markdown",
        include_metadata: bool = False,
        include_links: bool = False,
        query: Optional[str] = None,
        query_top_k: int = 10,
        wait_for_selector: Optional[str] = None,
        capture_screenshot: bool = False,
        extraction_options: Optional[ExtractionOptions] = None,
    ) -> ProcessResult:
        """
        Full-featured pipeline with all options.

        Args:
            url: URL to process
            timeout: Operation timeout (seconds)
            force_playwright: Force Playwright rendering
            use_llm: Use ReaderLM-v2 for conversion (default False - uses fast path)
            output_format: "markdown", "html", "text", or "screenshot"
            include_metadata: Extract page metadata (title, author, date, etc.)
            include_links: Extract all links from the page
            query: BM25 query to filter content by relevance
            query_top_k: Max paragraphs when filtering by query
            wait_for_selector: CSS selector to wait for before scraping
            capture_screenshot: Capture full-page screenshot
            extraction_options: Advanced extraction options (precision/recall, language, etc.)

        Returns:
            ProcessResult with content, success, metadata, and optional extras
        """
        pipeline_name = "llm_path" if use_llm else "fast_path"

        metadata = {
            "url": url,
            "method_used": None,
            "html_size": 0,
            "scrape_time_ms": 0,
            "inference_time_ms": 0,
            "pipeline": pipeline_name,
            "output_format": output_format
        }

        result = ProcessResult(
            content="",
            success=False,
            metadata=metadata
        )

        try:
            # Step 1: Scrape HTML
            logger.info(f"Scraping {url} (format={output_format}, llm={use_llm})")
            scrape_start = time.time()

            # Screenshot output format forces screenshot capture
            needs_screenshot = capture_screenshot or output_format == "screenshot"

            scrape_result = await self.scraper.scrape_full(
                url,
                wait_for_selector=wait_for_selector,
                timeout=timeout,
                force_playwright=force_playwright or needs_screenshot,
                capture_screenshot=needs_screenshot
            )

            scrape_time_ms = int((time.time() - scrape_start) * 1000)
            metadata["method_used"] = scrape_result.method
            metadata["html_size"] = len(scrape_result.html) if scrape_result.html else 0
            metadata["scrape_time_ms"] = scrape_time_ms

            if scrape_result.screenshot:
                result.screenshot = scrape_result.screenshot

            # If screenshot-only mode, return early
            if output_format == "screenshot":
                if scrape_result.screenshot:
                    result.content = f"Screenshot captured ({len(scrape_result.screenshot)} bytes base64)"
                    result.success = True
                else:
                    result.content = "Screenshot capture failed"
                return result

            if not scrape_result.success or not scrape_result.html:
                error = f"Failed to scrape {url}: {scrape_result.method} method returned empty HTML"
                logger.error(error)
                return result

            html = scrape_result.html
            logger.info(f"Scraped {len(html)} bytes from {url} using {scrape_result.method} ({scrape_time_ms}ms)")

            # Step 2: Extract metadata if requested
            if include_metadata:
                page_meta = extract_metadata(html)
                result.page_metadata = page_meta.to_dict()

            # Step 3: Extract links if requested
            if include_links:
                links = extract_links(html, base_url=url)
                result.links = [link.to_dict() for link in links]

            # Step 4: Convert content based on output format and use_llm flag
            conversion_start = time.time()

            if output_format == "text":
                # Plain text extraction (no markdown conversion)
                content, success, conv_meta = extract_with_trafilatura(
                    html,
                    output_format="txt",
                    include_links=False,
                    include_images=False,
                    options=extraction_options,
                )
                result.content = content or ""
                result.success = success
                metadata["conversion"] = conv_meta

            elif output_format == "html":
                # Cleaned HTML
                content, success, conv_meta = extract_with_trafilatura(
                    html,
                    output_format="html",
                    options=extraction_options,
                )
                result.content = content or ""
                result.success = success
                metadata["conversion"] = conv_meta

            elif use_llm:
                # LLM Path: Trafilatura extraction → ReaderLM-v2 conversion
                content, conv_success, conv_meta = await self._convert_with_llm(
                    html, None, max_tokens=8192, timeout=timeout
                )
                result.content = content
                result.success = conv_success
                metadata["conversion"] = conv_meta

            else:
                # Fast Path: Trafilatura extraction → markdownify conversion
                content, conv_success, conv_meta = self._convert_fast(
                    html, query, query_top_k, extraction_options
                )
                result.content = content
                result.success = conv_success
                metadata["conversion"] = conv_meta

            conversion_time_ms = int((time.time() - conversion_start) * 1000)
            metadata["inference_time_ms"] = conversion_time_ms

            # Step 5: Apply BM25 query filter for non-markdown formats if needed
            if query and output_format != "markdown" and result.content:
                filtered, bm25_meta = filter_content_by_query(
                    result.content,
                    query,
                    top_k=query_top_k
                )
                result.content = filtered
                metadata["bm25_filter"] = bm25_meta

            if result.success:
                logger.info(
                    f"Successfully processed {url}: {len(result.content)} chars "
                    f"({pipeline_name}, {conversion_time_ms}ms)"
                )

            return result

        except Exception as e:
            logger.error(f"Pipeline error for {url}: {e}", exc_info=True)
            return result

    def _convert_fast(
        self,
        html: str,
        query: Optional[str] = None,
        query_top_k: int = 10,
        extraction_options: Optional[ExtractionOptions] = None,
    ) -> tuple[str, bool, dict]:
        """Fast path: Trafilatura + markdownify (no LLM).

        Args:
            html: Raw HTML content
            query: Optional BM25 query for filtering
            query_top_k: Max paragraphs when filtering
            extraction_options: Advanced extraction options

        Returns:
            (markdown, success, metadata)
        """
        start_time = time.time()

        content, success, metadata = extract_and_convert_to_markdown(
            html,
            query=query,
            query_top_k=query_top_k,
            options=extraction_options,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)
        metadata["inference_time_ms"] = elapsed_ms
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
