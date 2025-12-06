"""ABOUTME: Reader MCP Server - Privacy-first web content extraction.

Provides completely self-hosted web-to-Markdown conversion with:
- Fast path: Trafilatura + markdownify (no LLM, ~100ms)
- LLM path: Trafilatura + ReaderLM-v2 (higher quality, ~2-3s)

Features:
- Multiple output formats: markdown, html, text, screenshot
- Metadata extraction: title, author, date, language, sitename
- Links extraction: internal/external URLs
- BM25 query filtering: extract only content relevant to a query
- CSS selector waiting for dynamic content

Zero external API calls. All URLs and content processed locally.
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import Context
from mcp.types import TextContent, CallToolResult
from pydantic import BaseModel, Field, field_validator

# Import common base class
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.mcp_base import MCPServerBase
from common.validation import (
    validate_url_field,
    validate_timeout_field,
    MAX_URL_LENGTH,
)
from common.error_handling import (
    ERROR_TIMEOUT,
    ERROR_FETCH_FAILED,
    ERROR_EXTRACTION_FAILED,
    ERROR_JS_RENDERING_FAILED,
    ERROR_UNEXPECTED,
    create_error_result,
)

from pipeline import ReaderPipeline
from html_preprocessor import ExtractionOptions

# =============================================================================
# Module-Level Constants
# =============================================================================

TIMEOUT_MIN: int = 5
TIMEOUT_MAX: int = 300
TIMEOUT_DEFAULT: int = 30
CONTENT_SIZE_WARNING_THRESHOLD: int = 1_000_000  # 1MB
QUERY_MAX_LENGTH: int = 500
SELECTOR_MAX_LENGTH: int = 200

# Valid output formats
VALID_OUTPUT_FORMATS = ["markdown", "html", "text", "screenshot"]

# =============================================================================
# Pydantic Models for Input/Output Schemas
# =============================================================================


class FetchPageInput(BaseModel):
    """Input schema for fetch_page tool."""

    url: str = Field(
        description="The URL to fetch (must include http:// or https://)",
        min_length=1,
        max_length=MAX_URL_LENGTH,
    )
    timeout: int = Field(
        default=TIMEOUT_DEFAULT,
        ge=TIMEOUT_MIN,
        le=TIMEOUT_MAX,
        description=f"Maximum page load time in seconds (range: {TIMEOUT_MIN}-{TIMEOUT_MAX})",
    )
    force_js_rendering: bool = Field(
        default=False,
        description="Force Playwright rendering for JavaScript-heavy SPAs",
    )
    use_llm: bool = Field(
        default=False,
        description="Use ReaderLM-v2 for higher quality conversion (~2-3s vs ~100ms)",
    )
    output_format: str = Field(
        default="markdown",
        description="Output format: 'markdown' (default), 'html', 'text', or 'screenshot'",
    )
    include_metadata: bool = Field(
        default=False,
        description="Extract page metadata (title, author, date, language, sitename)",
    )
    include_links: bool = Field(
        default=False,
        description="Extract all links from the page with internal/external classification",
    )
    query: Optional[str] = Field(
        default=None,
        max_length=QUERY_MAX_LENGTH,
        description="BM25 query to filter content - returns only paragraphs relevant to query (great for RAG)",
    )
    wait_for_selector: Optional[str] = Field(
        default=None,
        max_length=SELECTOR_MAX_LENGTH,
        description="CSS selector to wait for before extracting content (for dynamic pages)",
    )
    # Advanced extraction options
    favor_precision: bool = Field(
        default=False,
        description="Prefer less text but more accurate extraction (vs favor_recall)",
    )
    favor_recall: bool = Field(
        default=False,
        description="Prefer more text even when uncertain (vs favor_precision)",
    )
    deduplicate: bool = Field(
        default=False, description="Remove duplicate text segments from output"
    )
    target_language: Optional[str] = Field(
        default=None,
        max_length=5,
        description="Filter content by language (ISO 639-1 code, e.g., 'en', 'de', 'fr')",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        return validate_url_field(v)

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        return validate_timeout_field(v, min_val=TIMEOUT_MIN, max_val=TIMEOUT_MAX)

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        if v not in VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Invalid output_format: {v}. Must be one of: {VALID_OUTPUT_FORMATS}"
            )
        return v


class FetchPageOutput(BaseModel):
    """Output schema for fetch_page tool."""

    content: str = Field(description="Extracted content or error message")
    method: str = Field(description="Scraping method used (http or playwright)")
    pipeline: str = Field(
        description="Conversion pipeline used (fast_path or llm_path)"
    )
    output_format: str = Field(description="Output format used")
    html_size: int = Field(description="Size of fetched HTML in bytes")
    content_size: int = Field(description="Size of extracted content in bytes")
    scrape_time_ms: int = Field(description="Time spent scraping in milliseconds")
    conversion_time_ms: int = Field(description="Time spent converting in milliseconds")
    total_time_ms: int = Field(description="Total processing time in milliseconds")


# =============================================================================
# Server Initialization
# =============================================================================

server = MCPServerBase("reader")
mcp = server.get_mcp()
logger = server.get_logger()

pipeline = ReaderPipeline(
    scraper_endpoint=os.getenv("SCRAPER_ENDPOINT", "http://playwright-scraper:8000"),
    llama_endpoint=os.getenv("LLAMA_ENDPOINT", "http://llama-server-readerlm:8000"),
)


def get_mcp():
    """Get the MCP server instance for launcher compatibility."""
    return mcp


# =============================================================================
# MCP Tools
# =============================================================================


@mcp.tool()
async def fetch_page(
    url: str,
    timeout: int = TIMEOUT_DEFAULT,
    force_js_rendering: bool = False,
    use_llm: bool = False,
    output_format: str = "markdown",
    include_metadata: bool = False,
    include_links: bool = False,
    query: Optional[str] = None,
    wait_for_selector: Optional[str] = None,
    favor_precision: bool = False,
    favor_recall: bool = False,
    deduplicate: bool = False,
    target_language: Optional[str] = None,
    ctx: Context = None,
) -> CallToolResult:
    """Fetch and extract web page content with advanced options.

    Privacy-first: URLs and content never leave your infrastructure.

    **Conversion Paths**:
    • Fast path (default): Trafilatura + markdownify (~100-200ms)
    • LLM path: Trafilatura + ReaderLM-v2 (~2-3s, higher quality)

    **Output Formats**:
    • markdown: Clean markdown (default)
    • html: Cleaned HTML
    • text: Plain text only
    • screenshot: Full-page screenshot (base64)

    **Advanced Features**:
    • Metadata extraction: title, author, date, language, sitename
    • Links extraction: all page links with internal/external classification
    • BM25 query filtering: extract only content relevant to your query
    • CSS selector waiting: wait for dynamic content before extraction
    • Ensemble extraction: Trafilatura + readability-lxml fallback (F1: 0.981)
    • Language filtering: only extract content in specified language
    • Deduplication: remove duplicate text segments

    Args:
        url: URL to fetch (must include http:// or https://)
        timeout: Max page load time (5-300 seconds, default: 30)
        force_js_rendering: Force Playwright for JS-heavy SPAs
        use_llm: Use ReaderLM-v2 for higher quality (slower)
        output_format: "markdown", "html", "text", or "screenshot"
        include_metadata: Extract page metadata (title, author, date, etc.)
        include_links: Extract all links from the page
        query: BM25 query to filter content by relevance (great for RAG)
        wait_for_selector: CSS selector to wait for before scraping
        favor_precision: Prefer less text but more accurate extraction
        favor_recall: Prefer more text even when uncertain
        deduplicate: Remove duplicate text segments
        target_language: Filter by language (ISO 639-1 code: "en", "de", etc.)

    Returns:
        Content with metadata. If include_metadata/include_links, returns JSON with:
        - content: The extracted content
        - metadata: Page metadata (if include_metadata=True)
        - links: List of links (if include_links=True)
        - screenshot: Base64 screenshot (if output_format="screenshot")

    Examples:
        # Basic fetch
        fetch_page("https://example.com")

        # Get only content about pricing
        fetch_page("https://example.com/docs", query="pricing plans")

        # Extract with metadata
        fetch_page("https://news.site.com/article", include_metadata=True)

        # Get all links for crawling
        fetch_page("https://example.com", include_links=True)

        # High precision extraction (less noise)
        fetch_page("https://example.com", favor_precision=True)

        # Only English content
        fetch_page("https://example.com", target_language="en")

        # Screenshot
        fetch_page("https://example.com", output_format="screenshot")
    """
    pipeline_name = "llm_path" if use_llm else "fast_path"
    logger.debug(
        f"fetch_page: url={url}, format={output_format}, llm={use_llm}, query={query}"
    )

    try:
        url_preview = url[:50] + ("..." if len(url) > 50 else "")

        if ctx:
            await ctx.report_progress(1, 4, f"Scraping: {url_preview}")

        logger.info(
            f"Processing URL: {url} (format={output_format}, pipeline={pipeline_name})"
        )

        start_time = time.time()

        # Build extraction options from parameters
        extraction_options = ExtractionOptions(
            favor_precision=favor_precision,
            favor_recall=favor_recall,
            deduplicate=deduplicate,
            target_language=target_language,
        )

        # Process URL through full pipeline
        result = await pipeline.process_url_full(
            url=url,
            timeout=timeout,
            force_playwright=force_js_rendering,
            use_llm=use_llm,
            output_format=output_format,
            include_metadata=include_metadata,
            include_links=include_links,
            query=query,
            wait_for_selector=wait_for_selector,
            extraction_options=extraction_options,
        )

        total_time_ms = int((time.time() - start_time) * 1000)
        metadata = result.metadata

        # Handle failure
        if not result.success:
            method_used = metadata.get("method_used", "unknown")
            error_msg = f"Failed to process {url}"

            if method_used == "playwright" and force_js_rendering:
                error_code = ERROR_JS_RENDERING_FAILED
            elif metadata.get("scrape_time_ms", 0) >= timeout * 1000:
                error_code = ERROR_TIMEOUT
            elif metadata.get("html_size", 0) == 0:
                error_code = ERROR_FETCH_FAILED
            else:
                error_code = ERROR_EXTRACTION_FAILED

            logger.error(
                f"{error_msg} - Method: {method_used}, HTML: {metadata.get('html_size', 0)} bytes"
            )

            if ctx:
                await ctx.error(error_msg)

            return create_error_result(
                error_message=error_msg,
                error_code=error_code,
                error_type="ProcessingError",
                additional_metadata={
                    "url": url,
                    "method_used": method_used,
                    "pipeline": pipeline_name,
                    "output_format": output_format,
                },
            )

        # Report progress
        if ctx:
            method = metadata.get("method_used", "unknown")
            await ctx.report_progress(2, 4, f"Scraped ({method})")
            converter = "ReaderLM-v2" if use_llm else "markdownify"
            await ctx.report_progress(3, 4, f"Converting with {converter}...")
            await ctx.report_progress(4, 4, f"Complete: {len(result.content)} chars")

        # Log warning for large content
        if len(result.content) > CONTENT_SIZE_WARNING_THRESHOLD:
            logger.warning(f"Large content from {url}: {len(result.content)} chars")

        logger.info(
            f"Processed {url}: {len(result.content)} chars ({pipeline_name}) - "
            f"Scrape: {metadata.get('scrape_time_ms', 0)}ms, "
            f"Convert: {metadata.get('inference_time_ms', 0)}ms, "
            f"Total: {total_time_ms}ms"
        )

        # Build response content
        # If extras requested, return structured JSON; otherwise just content
        has_extras = include_metadata or include_links or output_format == "screenshot"

        if has_extras:
            response_data = {"content": result.content}
            if include_metadata and result.page_metadata:
                response_data["metadata"] = result.page_metadata
            if include_links and result.links:
                response_data["links"] = result.links
            if result.screenshot:
                response_data["screenshot"] = result.screenshot

            content_text = json.dumps(response_data, indent=2)
        else:
            content_text = result.content

        return CallToolResult(
            content=[TextContent(type="text", text=content_text)],
            isError=False,
            metadata={
                "method": metadata.get("method_used", "unknown"),
                "pipeline": pipeline_name,
                "output_format": output_format,
                "html_size": metadata.get("html_size", 0),
                "content_size": len(result.content),
                "scrape_time_ms": metadata.get("scrape_time_ms", 0),
                "conversion_time_ms": metadata.get("inference_time_ms", 0),
                "total_time_ms": total_time_ms,
                "url": url,
                "has_metadata": result.page_metadata is not None,
                "has_links": result.links is not None,
                "has_screenshot": result.screenshot is not None,
                "query_used": query,
            },
        )

    except Exception as e:
        logger.error(f"Unexpected error processing {url}: {e}", exc_info=True)
        if ctx:
            await ctx.error(str(e))

        return create_error_result(
            error_message=str(e),
            error_code=ERROR_UNEXPECTED,
            error_type="UnexpectedError",
            additional_metadata={
                "url": url,
                "exception_type": type(e).__name__,
                "output_format": output_format,
            },
        )


if __name__ == "__main__":
    logger.info("Starting Reader MCP server (Streamable HTTP)...")
    logger.info("Configuration:")
    logger.info(
        f"  Scraper endpoint: {os.getenv('SCRAPER_ENDPOINT', 'http://playwright-scraper:8000')}"
    )
    logger.info(
        f"  Llama endpoint: {os.getenv('LLAMA_ENDPOINT', 'http://llama-server-readerlm:8000')}"
    )
    logger.info("")
    logger.info("Features:")
    logger.info("  - Output formats: markdown, html, text, screenshot")
    logger.info("  - Metadata extraction: title, author, date, language")
    logger.info("  - Links extraction: internal/external URLs")
    logger.info("  - BM25 query filtering: extract relevant content only")
    logger.info("  - CSS selector waiting: for dynamic content")
    logger.info("")
    logger.info("Privacy Notice:")
    logger.info("  All URL fetching and content processing occurs locally.")
    logger.info("  No URLs or content are transmitted to external services.")
    logger.info("")
    server.run(transport="streamable-http")
