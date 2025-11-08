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
    validate_string_length_field,
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

# Prompt/instruction constraints
PROMPT_MAX_LENGTH: int = 2000

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
    prompt: Optional[str] = Field(
        default=None,
        description=(
            "Optional extraction instruction (e.g., 'Extract the main headline and price'). "
            "If not provided, returns full markdown. ReaderLM-v2 supports custom instructions."
        ),
        max_length=PROMPT_MAX_LENGTH
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

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: Optional[str]) -> Optional[str]:
        """Validate prompt length if provided."""
        if v is None:
            return v
        return validate_string_length_field(v, min_length=1, max_length=PROMPT_MAX_LENGTH, field_name="prompt")


class FetchPageOutput(BaseModel):
    """Output schema for fetch_page tool."""
    content: str = Field(description="Extracted Markdown content or error message")
    method: str = Field(description="Scraping method used (http or playwright)")
    html_size: int = Field(description="Size of fetched HTML in bytes")
    content_size: int = Field(description="Size of extracted/markdown content in bytes")
    scrape_time_ms: int = Field(description="Time spent scraping in milliseconds")
    inference_time_ms: int = Field(description="Time spent in ReaderLM-v2 in milliseconds")
    total_time_ms: int = Field(description="Total processing time in milliseconds")
    extraction_mode: bool = Field(description="Whether prompt-based extraction was used")
    instruction_used: Optional[str] = Field(description="The instruction/prompt that was used, if any")


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

@mcp.tool(inputSchema=FetchPageInput, outputSchema=FetchPageOutput)
async def fetch_page(
    url: str,
    prompt: Optional[str] = None,
    timeout: int = TIMEOUT_DEFAULT,
    force_js_rendering: bool = False,
    ctx: Context = None
) -> CallToolResult:
    """Fetch and convert web page to clean Markdown with optional instruction-based extraction.

    Completely private: URLs and content never leave your infrastructure.
    Uses Playwright for web scraping + ReaderLM-v2 for conversion/extraction.

    Args:
        url: The URL to fetch (must include http:// or https://)
        prompt: Optional extraction instruction (e.g., "Extract the main headline and price").
               If not provided, returns full markdown. ReaderLM-v2 supports custom instructions.
        timeout: Maximum page load time in seconds (range: 5-300, default: 30)
        force_js_rendering: Force Playwright even for simple pages (default: False).
                          Use True for JavaScript-heavy SPAs like Twitter, Reddit, etc.

    Returns:
        CallToolResult with extracted Markdown content and comprehensive metadata including:
        - method: Scraping method used (http or playwright)
        - html_size: Size of fetched HTML in bytes
        - content_size: Size of extracted/markdown content in bytes
        - scrape_time_ms: Time spent scraping in milliseconds
        - inference_time_ms: Time spent in ReaderLM-v2 in milliseconds
        - total_time_ms: Total processing time in milliseconds
        - extraction_mode: Whether prompt-based extraction was used
        - instruction_used: The instruction/prompt that was used, if any

    Features:
        • Smart HTTP/Playwright fallback (fast for static pages)
        • Full JavaScript rendering for SPAs
        • Optional instruction-based extraction (prompt parameter)
        • 20% better quality than previous ReaderLM (v2)
        • No rate limits, unlimited usage
        • Complete privacy (all processing local)

    Examples:
        # Full page content
        fetch_page("https://example.com")

        # Extract specific information
        fetch_page("https://example.com", prompt="Extract the main headline and author")

        # Force JavaScript rendering
        fetch_page("https://github.com/anthropics/claude-code", force_js_rendering=True)

        # Extract with custom instruction
        fetch_page("https://store.example.com/product", prompt="Extract the price and availability")
    """
    # Step 1: Log request details (Pydantic validation already done by framework)
    logger.debug(f"fetch_page called with url={url}, prompt={prompt}, timeout={timeout}, force_js_rendering={force_js_rendering}")

    # Step 2: Process request
    try:
        url_preview = url[:50] + ("..." if len(url) > 50 else "")

        if ctx:
            await ctx.report_progress(1, 4, f"Scraping: {url_preview}")

        logger.info(
            f"Processing URL: {url}" +
            (f" (extraction: {prompt[:50]}{'...' if len(prompt) > 50 else ''})" if prompt else " (full content)")
        )

        start_time = time.time()

        # Process URL through pipeline with optional instruction
        content, success, metadata = await pipeline.process_url(
            url,
            instruction=prompt,
            timeout=timeout,
            force_playwright=force_js_rendering
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
                    "html_size": metadata.get('html_size', 0),
                    "scrape_time_ms": metadata.get('scrape_time_ms', 0),
                    "inference_time_ms": metadata.get('inference_time_ms', 0),
                    "timeout": timeout,
                    "force_js_rendering": force_js_rendering
                }
            )

        # Step 4: Report progress for successful processing
        if ctx:
            method = metadata.get('method_used', 'unknown')
            await ctx.report_progress(2, 4, f"Scraped ({method})")
            mode_label = "Extracting" if prompt else "Converting"
            await ctx.report_progress(3, 4, f"{mode_label} with ReaderLM-v2...")
            await ctx.report_progress(4, 4, f"Complete: {len(content)} chars")

        # Log warning for very large content
        if len(content) > CONTENT_SIZE_WARNING_THRESHOLD:
            logger.warning(
                f"Large content extracted from {url}: {len(content)} chars "
                f"({len(content) / 1_000_000:.2f} MB)"
            )

        logger.info(
            f"Successfully processed {url}: {len(content)} chars "
            f"({metadata.get('method_used', 'unknown')}) - "
            f"Scrape: {metadata.get('scrape_time_ms', 0)}ms, "
            f"Inference: {metadata.get('inference_time_ms', 0)}ms, "
            f"Total: {total_time_ms}ms"
        )

        # Step 5: Return successful result with comprehensive metadata
        return CallToolResult(
            content=[TextContent(type="text", text=content)],
            isError=False,
            metadata={
                "method": metadata.get('method_used', 'unknown'),
                "html_size": metadata.get('html_size', 0),
                "content_size": len(content),
                "scrape_time_ms": metadata.get('scrape_time_ms', 0),
                "inference_time_ms": metadata.get('inference_time_ms', 0),
                "total_time_ms": total_time_ms,
                "extraction_mode": prompt is not None,
                "instruction_used": prompt if prompt else None,
                "url": url,
                "timeout": timeout,
                "force_js_rendering": force_js_rendering
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
                "force_js_rendering": force_js_rendering
            }
        )


@mcp.tool(outputSchema=GetReaderInfoOutput)
async def get_reader_info(ctx: Context = None) -> CallToolResult:
    """Get information about Reader capabilities and configuration.

    Returns:
        CallToolResult with comprehensive information about Reader capabilities,
        configuration, service health status, and usage guidelines.
    """
    logger.debug("get_reader_info called")

    try:
        # Check service health
        scraper_health = await pipeline.scraper.health_check()
        llama_health = await pipeline.llama.health_check()

        logger.info(
            f"Reader info requested - "
            f"Scraper: {'healthy' if scraper_health else 'unavailable'}, "
            f"ReaderLM: {'healthy' if llama_health else 'unavailable'}"
        )

        info = f"""
Reader - Privacy-First Web Content Extraction

**Architecture**:
✓ Playwright: Web scraping with JavaScript rendering
✓ ReaderLM-v2: HTML→Markdown/extraction via llama-server (1.5B params, Q4_K_M quantized)
✓ llama.cpp: Optimized inference engine on GPU

**Service Status**:
• Playwright Scraper: {"🟢 Healthy" if scraper_health else "🔴 Unavailable"}
• ReaderLM-v2 Inference: {"🟢 Healthy" if llama_health else "🔴 Unavailable"}

**Features**:
• Complete privacy: All processing on-premises
• Smart fallback: HTTP first, Playwright for JS-heavy sites
• Superior quality: ReaderLM-v2 outperforms prior versions
• Instruction-based extraction: Pass a prompt to extract specific information
• Configurable timeouts for slow pages (range: {TIMEOUT_MIN}-{TIMEOUT_MAX}s)
• Streaming progress reporting

**Modes**:
1. Full Content Mode (no prompt):
   - Returns complete page content as Markdown
   - Useful for knowledge bases, documentation, full articles

2. Extraction Mode (with prompt):
   - Pass custom instruction to ReaderLM-v2
   - Examples:
     - "Extract the main headline and author"
     - "Extract price and availability"
     - "List all links on this page"
   - Faster, more focused results
   - Uses ReaderLM-v2's native instruction support
   - Max prompt length: {PROMPT_MAX_LENGTH} characters

**Supported Content**:
• Web pages (HTML, JavaScript-rendered SPAs)
• News articles and blogs
• Documentation sites
• Academic papers and research
• Twitter threads, Reddit discussions (fully rendered)
• Medium articles and similar platforms

**Performance Characteristics**:
• Typical latency: 2-5 seconds per page
  - Static page (HTTP): ~1-2 seconds
  - JS-heavy page (Playwright): ~4-5 seconds
  - Inference: ~1-2 seconds
• No rate limits
• Unlimited concurrent requests (GPU limited)
• Local GPU acceleration via llama.cpp

**Best Practices**:
• Use force_js_rendering=true for known SPAs (Twitter, Reddit, etc.)
• Use prompt for targeted extraction from large pages
• Set appropriate timeout for very large documents (30s+ for 100MB+ pages)
• Combine with web_search: search for snippets, then fetch_page for full content

**Cost**:
• Infrastructure: Zero per-request (uses owned GPU)
• Token usage: Zero (local inference)
• Data privacy: Maximum (no external API calls)

**Always Privacy-First**: Unlike cloud-based APIs, this reader keeps all your data local.
No third-party APIs, no rate limits, no usage costs. Complete control.
"""

        return CallToolResult(
            content=[TextContent(type="text", text=info.strip())],
            isError=False,
            metadata={
                "scraper_healthy": scraper_health,
                "llama_healthy": llama_health,
                "all_services_healthy": scraper_health and llama_health,
                "timeout_min": TIMEOUT_MIN,
                "timeout_max": TIMEOUT_MAX,
                "timeout_default": TIMEOUT_DEFAULT,
                "prompt_max_length": PROMPT_MAX_LENGTH,
                "url_max_length": URL_MAX_LENGTH,
                "scraper_endpoint": os.getenv("SCRAPER_ENDPOINT", "http://playwright-scraper:8000"),
                "llama_endpoint": os.getenv("LLAMA_ENDPOINT", "http://llama-server-readerlm:8000")
            }
        )

    except Exception as e:
        logger.error(f"Error getting reader info: {e}", exc_info=True)
        error_msg = f"Failed to retrieve reader information: {str(e)}"

        if ctx:
            await ctx.error(error_msg)

        return create_error_result(
            error_message=error_msg,
            error_code=ERROR_UNEXPECTED,
            error_type="UnexpectedError",
            additional_metadata={"exception_type": type(e).__name__}
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
