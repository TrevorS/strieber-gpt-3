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
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import Context
from mcp.types import TextContent, CallToolResult

# Import common base class
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.mcp_base import MCPServerBase

from pipeline import ReaderPipeline

# Initialize MCP server
server = MCPServerBase("reader")
mcp = server.get_mcp()
logger = server.get_logger()

# Initialize pipeline
pipeline = ReaderPipeline(
    scraper_endpoint=os.getenv("SCRAPER_ENDPOINT", "http://playwright-scraper:8000"),
    llama_endpoint=os.getenv("LLAMA_ENDPOINT", "http://llama-server-readerlm:8000")
)


# Module-level function for launcher.py
def get_mcp():
    """Get the MCP server instance for launcher compatibility."""
    return mcp


@mcp.tool()
async def fetch_page(
    url: str,
    prompt: Optional[str] = None,
    timeout: int = 30,
    force_js_rendering: bool = False,
    ctx: Context = None
) -> CallToolResult:
    """Fetch and convert web page to clean Markdown with optional instruction-based extraction.

    Completely private: URLs and content never leave your infrastructure.
    Uses Playwright for web scraping + ReaderLM-v2 for conversion/extraction.

    Args:
        url: The URL to fetch (must include http:// or https://)
        prompt: Optional extraction instruction (e.g., "Extract the main headline and price")
               If not provided, returns full markdown. ReaderLM-v2 supports custom instructions.
        timeout: Maximum page load time in seconds (default: 30)
        force_js_rendering: Force Playwright even for simple pages (default: False)
        ctx: MCP context for progress/logging (auto-injected)

    Returns:
        CallToolResult with content and structured metadata:
        - method: Scraping method used (http or playwright)
        - html_size: Size of fetched HTML in bytes
        - content_size: Size of extracted/markdown content in bytes
        - scrape_time_ms: Time spent scraping in milliseconds
        - inference_time_ms: Time spent in ReaderLM-v2 in milliseconds
        - extraction_mode: bool indicating if prompt was used

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
    if not url:
        if ctx:
            await ctx.error("URL is required")
        return CallToolResult(
            content=[TextContent(type="text", text="Error: URL is required")],
            isError=True
        )

    if not url.startswith(("http://", "https://")):
        if ctx:
            await ctx.error("URL must start with http:// or https://")
        return CallToolResult(
            content=[TextContent(type="text", text="Error: URL must start with http:// or https://")],
            isError=True
        )

    try:
        url_preview = url[:50] + ("..." if len(url) > 50 else "")

        if ctx:
            await ctx.report_progress(1, 4, f"Scraping: {url_preview}")

        logger.info(f"Processing URL: {url}" + (f" (extraction: {prompt[:50]}...)" if prompt else ""))

        start_time = time.time()

        # Process URL through pipeline with optional instruction
        content, success, metadata = await pipeline.process_url(
            url,
            instruction=prompt,
            timeout=timeout,
            force_playwright=force_js_rendering
        )

        total_time_ms = int((time.time() - start_time) * 1000)

        if not success:
            error_msg = f"Failed to process {url}"
            logger.error(error_msg)
            if ctx:
                await ctx.error(error_msg)
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {error_msg}")],
                isError=True
            )

        if ctx:
            method = metadata.get('method_used', 'unknown')
            await ctx.report_progress(2, 4, f"Scraped ({method})")
            mode_label = "Extracting" if prompt else "Converting"
            await ctx.report_progress(3, 4, f"{mode_label} with ReaderLM-v2...")
            await ctx.report_progress(4, 4, f"Complete: {len(content)} chars")

        logger.info(
            f"Successfully processed {url}: {len(content)} chars "
            f"({metadata.get('method_used', 'unknown')}) - "
            f"Scrape: {metadata.get('scrape_time_ms', 0)}ms, "
            f"Inference: {metadata.get('inference_time_ms', 0)}ms"
        )

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
                "instruction_used": prompt if prompt else None
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error processing {url}: {e}", exc_info=True)
        error_msg = str(e)
        if ctx:
            await ctx.error(error_msg)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {error_msg}")],
            isError=True
        )


@mcp.tool()
async def get_reader_info(ctx: Context = None) -> list:
    """Get information about Reader capabilities and configuration.

    Returns:
        MCP content array format: [TextContent(type="text", text="info text")]
    """
    scraper_health = await pipeline.scraper.health_check()
    llama_health = await pipeline.llama.health_check()

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
• Configurable timeouts for slow pages
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

**Comparison to Cloud APIs**:
• Jina Reader: 1-2s latency, 3x token cost, rate limits (500 RPM)
• Reader (local): 2-5s latency, $0 cost, unlimited usage, complete privacy

For more info, see: https://jina.ai/reader/
"""
    return [TextContent(type="text", text=info.strip())]


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
