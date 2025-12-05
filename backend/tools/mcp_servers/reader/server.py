"""ABOUTME: Reader MCP Server - Privacy-first web content extraction.

Provides completely self-hosted web-to-Markdown conversion with optional
instruction-based extraction using:
- Playwright: For web scraping with JavaScript rendering
- ReaderLM-v2: For HTML-to-Markdown conversion via llama-server

Zero external API calls. All URLs and content processed locally.
"""

import logging
import sys
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

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
    ERROR_INVALID_URL,
    ERROR_TIMEOUT,
    ERROR_FETCH_FAILED,
    ERROR_EXTRACTION_FAILED,
    ERROR_JS_RENDERING_FAILED,
    ERROR_VALIDATION_FAILED,
    ERROR_UNEXPECTED,
    create_error_result,
    create_validation_error,
)

from pipeline import ReaderPipeline

# =============================================================================
# Module-Level Constants
# =============================================================================

# Tool-specific timeout constraints (seconds) - overrides common defaults
TIMEOUT_MIN: int = 5  # Higher min for web scraping (need JS rendering time)
TIMEOUT_MAX: int = 300  # Allow longer timeouts for large pages
TIMEOUT_DEFAULT: int = 30

# Content size limits
CONTENT_SIZE_WARNING_THRESHOLD: int = 1_000_000  # 1MB

# =============================================================================
# Pydantic Models for Input/Output Schemas
# =============================================================================

class FetchPageInput(BaseModel):
    """Input schema for fetch_page tool."""
    url: str = Field(
        description="The URL to fetch (must include http:// or https://)",
        min_length=1,
        max_length=MAX_URL_LENGTH
    )
    timeout: int = Field(
        default=TIMEOUT_DEFAULT,
        ge=TIMEOUT_MIN,
        le=TIMEOUT_MAX,
        description=f"Maximum page load time in seconds (range: {TIMEOUT_MIN}-{TIMEOUT_MAX})"
    )
    force_js_rendering: bool = Field(
        default=False,
        description=(
            "Force Playwright even for simple pages (default: False). "
            "Use True for JavaScript-heavy SPAs like Twitter, Reddit, etc."
        )
    )
    use_llm: bool = Field(
        default=False,
        description=(
            "Use ReaderLM-v2 for HTML-to-Markdown conversion (default: False). "
            "When False, uses fast rule-based conversion (Trafilatura + markdownify, ~100ms). "
            "When True, uses ReaderLM-v2 LLM for higher quality on complex pages (~2-3s). "
            "The fast path handles 90%+ of pages well."
        )
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format and scheme using shared validator."""
        return validate_url_field(v)

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout range using shared validator."""
        return validate_timeout_field(v, min_val=TIMEOUT_MIN, max_val=TIMEOUT_MAX)


class FetchPageOutput(BaseModel):
    """Output schema for fetch_page tool."""
    content: str = Field(description="Extracted Markdown content or error message")
    method: str = Field(description="Scraping method used (http or playwright)")
    pipeline: str = Field(description="Conversion pipeline used (fast_path or llm_path)")
    html_size: int = Field(description="Size of fetched HTML in bytes")
    content_size: int = Field(description="Size of extracted/markdown content in bytes")
    scrape_time_ms: int = Field(description="Time spent scraping in milliseconds")
    conversion_time_ms: int = Field(description="Time spent converting HTML to Markdown in milliseconds")
    total_time_ms: int = Field(description="Total processing time in milliseconds")


class GetReaderInfoOutput(BaseModel):
    """Output schema for get_reader_info tool."""
    info_text: str = Field(description="Comprehensive information about Reader capabilities and configuration")


# =============================================================================
# Server Initialization
# =============================================================================

# Initialize MCP server
server = MCPServerBase("reader")
mcp = server.get_mcp()
logger = server.get_logger()

# Initialize pipeline
pipeline = ReaderPipeline(
    scraper_endpoint=os.getenv("SCRAPER_ENDPOINT", "http://playwright-scraper:8000"),
    llama_endpoint=os.getenv("LLAMA_ENDPOINT", "http://llama-server-readerlm:8000")
)


# =============================================================================
# Module-Level Functions
# =============================================================================


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
    ctx: Context = None
) -> CallToolResult:
    """Fetch and convert web page to clean, optimized Markdown.

    Completely private: URLs and content never leave your infrastructure.

    **Two conversion paths**:
    1. **Fast path (default)**: Trafilatura + markdownify (~100-200ms, no LLM)
       - Uses Trafilatura for content extraction (F1: 0.958, used by HuggingFace/IBM/Microsoft)
       - Rule-based markdown conversion - fast and deterministic
       - Handles 90%+ of pages well

    2. **LLM path (use_llm=True)**: Trafilatura + ReaderLM-v2 (~2-3s)
       - Uses ReaderLM-v2 (1.5B params) for higher quality conversion
       - Better for complex/broken HTML structures
       - Slightly higher accuracy but 10-20x slower

    **How it works**:
    - Automatically extracts main content (removes navigation, ads, sidebars)
    - Preserves structure: headings, lists, tables, code blocks
    - Handles large documents efficiently

    Args:
        url: The URL to fetch (must include http:// or https://)
        timeout: Maximum page load time in seconds (range: 5-300, default: 30)
        force_js_rendering: Force Playwright even for simple pages (default: False).
                          Use True for JavaScript-heavy sites: Twitter, Reddit, Medium, etc.
        use_llm: Use ReaderLM-v2 for conversion (default: False).
                 When False, uses fast rule-based conversion (~100ms).
                 When True, uses LLM for higher quality on complex pages (~2-3s).

    Returns:
        CallToolResult with clean Markdown content and metadata:
        - method: Scraping method used (http or playwright)
        - pipeline: Conversion path used (fast_path or llm_path)
        - html_size: Size of raw HTML fetched
        - scrape_time_ms: Time to scrape the page
        - conversion_time_ms: Time for HTML→Markdown conversion

    **Performance**:
    • Fast path: ~200-500ms total (static page), ~2-3s (JS-heavy page)
    • LLM path: ~2-4s total (static page), ~4-6s (JS-heavy page)

    Examples:
        # Fast fetch (default) - good for most pages
        fetch_page("https://example.com")

        # Force JavaScript rendering for SPAs
        fetch_page("https://github.com/anthropics/claude-code", force_js_rendering=True)

        # Use LLM for complex/broken HTML
        fetch_page("https://complex-site.com", use_llm=True)
    """
    # Step 1: Log request details (Pydantic validation already done by framework)
    pipeline_name = "llm_path" if use_llm else "fast_path"
    logger.debug(f"fetch_page called with url={url}, timeout={timeout}, force_js_rendering={force_js_rendering}, use_llm={use_llm}")

    # Step 2: Process request
    try:
        url_preview = url[:50] + ("..." if len(url) > 50 else "")

        if ctx:
            await ctx.report_progress(1, 4, f"Scraping: {url_preview}")

        logger.info(f"Processing URL: {url} (pipeline: {pipeline_name})")

        start_time = time.time()

        # Process URL through pipeline
        content, success, metadata = await pipeline.process_url(
            url,
            instruction=None,
            timeout=timeout,
            force_playwright=force_js_rendering,
            use_llm=use_llm
        )

        total_time_ms = int((time.time() - start_time) * 1000)

        # Step 3: Handle pipeline failure
        if not success:
            method_used = metadata.get('method_used', 'unknown')
            error_msg = f"Failed to process {url}"

            # Determine specific error code based on method and context
            if method_used == 'playwright' and force_js_rendering:
                error_code = ERROR_JS_RENDERING_FAILED
            elif metadata.get('scrape_time_ms', 0) >= timeout * 1000:
                error_code = ERROR_TIMEOUT
            elif metadata.get('html_size', 0) == 0:
                error_code = ERROR_FETCH_FAILED
            elif metadata.get('inference_time_ms', 0) > 0:
                error_code = ERROR_EXTRACTION_FAILED
            else:
                error_code = ERROR_FETCH_FAILED

            logger.error(
                f"{error_msg} - Method: {method_used}, "
                f"HTML size: {metadata.get('html_size', 0)}, "
                f"Scrape time: {metadata.get('scrape_time_ms', 0)}ms"
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
                    "html_size": metadata.get('html_size', 0),
                    "scrape_time_ms": metadata.get('scrape_time_ms', 0),
                    "conversion_time_ms": metadata.get('inference_time_ms', 0),
                    "timeout": timeout,
                    "force_js_rendering": force_js_rendering,
                    "use_llm": use_llm
                }
            )

        # Step 4: Report progress for successful processing
        if ctx:
            method = metadata.get('method_used', 'unknown')
            await ctx.report_progress(2, 4, f"Scraped ({method})")
            converter = "ReaderLM-v2" if use_llm else "markdownify"
            await ctx.report_progress(3, 4, f"Converting with {converter}...")
            await ctx.report_progress(4, 4, f"Complete: {len(content)} chars ({pipeline_name})")

        # Log warning for very large content
        if len(content) > CONTENT_SIZE_WARNING_THRESHOLD:
            logger.warning(
                f"Large content extracted from {url}: {len(content)} chars "
                f"({len(content) / 1_000_000:.2f} MB)"
            )

        logger.info(
            f"Successfully processed {url}: {len(content)} chars "
            f"({metadata.get('method_used', 'unknown')}, {pipeline_name}) - "
            f"Scrape: {metadata.get('scrape_time_ms', 0)}ms, "
            f"Conversion: {metadata.get('inference_time_ms', 0)}ms, "
            f"Total: {total_time_ms}ms"
        )

        # Step 5: Return successful result with comprehensive metadata
        return CallToolResult(
            content=[TextContent(type="text", text=content)],
            isError=False,
            metadata={
                "method": metadata.get('method_used', 'unknown'),
                "pipeline": pipeline_name,
                "html_size": metadata.get('html_size', 0),
                "content_size": len(content),
                "scrape_time_ms": metadata.get('scrape_time_ms', 0),
                "conversion_time_ms": metadata.get('inference_time_ms', 0),
                "total_time_ms": total_time_ms,
                "url": url,
                "timeout": timeout,
                "force_js_rendering": force_js_rendering,
                "use_llm": use_llm
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error processing {url}: {e}", exc_info=True)
        error_msg = str(e)
        if ctx:
            await ctx.error(error_msg)

        return create_error_result(
            error_message=error_msg,
            error_code=ERROR_UNEXPECTED,
            error_type="UnexpectedError",
            additional_metadata={
                "url": url,
                "exception_type": type(e).__name__,
                "timeout": timeout,
                "force_js_rendering": force_js_rendering,
                "use_llm": use_llm
            }
        )


if __name__ == "__main__":
    logger.info("Starting Reader MCP server (Streamable HTTP)...")
    logger.info("Configuration:")
    logger.info(f"  Scraper endpoint: {os.getenv('SCRAPER_ENDPOINT', 'http://playwright-scraper:8000')}")
    logger.info(f"  Llama endpoint: {os.getenv('LLAMA_ENDPOINT', 'http://llama-server-readerlm:8000')}")
    logger.info("")
    logger.info("Privacy Notice:")
    logger.info("  All URL fetching and content processing occurs locally.")
    logger.info("  No URLs or content are transmitted to external services.")
    logger.info("")
    server.run(transport="streamable-http")
