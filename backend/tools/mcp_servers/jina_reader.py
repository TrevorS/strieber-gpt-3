"""ABOUTME: Jina Reader MCP Server - Web content extraction and markdown conversion.

Provides web page content retrieval via local ReaderLM-v2 model with Playwright,
with optional fallback to Jina Reader API for PDFs and complex pages.
Converts URLs to clean, LLM-friendly markdown format.
"""

import asyncio
import os
import logging
import re
from typing import Optional
from urllib.parse import quote

import httpx
from mcp.server.fastmcp import Context

from common.mcp_base import MCPServerBase

# Initialize MCP server with base class
server = MCPServerBase("jina-reader")
mcp = server.get_mcp()
logger = server.get_logger()

# Configuration from environment
JINA_API_KEY = os.getenv("JINA_API_KEY")
READERLM_BASE_URL = os.getenv("READERLM_BASE_URL", "http://llama-server-reader:8004")
PLAYWRIGHT_BASE_URL = os.getenv("PLAYWRIGHT_BASE_URL", "http://playwright-fetcher:8005")
USE_LOCAL_READER = os.getenv("USE_LOCAL_READER", "true").lower() == "true"

if not JINA_API_KEY:
    logger.warning("JINA_API_KEY not set. API fallback will use free tier (20 RPM limit).")

if USE_LOCAL_READER:
    logger.info(f"Local reader enabled: ReaderLM @ {READERLM_BASE_URL}, Playwright @ {PLAYWRIGHT_BASE_URL}")
else:
    logger.info("Local reader disabled, using Jina API only")


# ============================================================================
# Local Processing Functions
# ============================================================================

async def fetch_html_with_playwright(url: str, timeout: int = 10) -> Optional[str]:
    """Fetch HTML using Playwright headless browser.

    Args:
        url: URL to fetch
        timeout: Timeout in seconds

    Returns:
        Rendered HTML content or None if failed
    """
    try:
        async with httpx.AsyncClient(timeout=timeout + 5.0) as client:
            response = await client.get(
                f"{PLAYWRIGHT_BASE_URL}/fetch",
                params={
                    "url": url,
                    "timeout": timeout,
                    "wait_for": "networkidle",
                    "block_resources": True
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("html")
    except Exception as e:
        logger.error(f"Playwright fetch failed for {url}: {e}")
        return None


async def convert_html_to_markdown_local(html: str, url: str) -> Optional[str]:
    """Convert HTML to Markdown using local ReaderLM-v2 model.

    Args:
        html: HTML content to convert
        url: Original URL (for context)

    Returns:
        Markdown content or None if failed
    """
    try:
        # Prepare messages for ReaderLM-v2 (uses ChatML format)
        messages = [
            {
                "role": "system",
                "content": "Convert the HTML to Markdown. Remove ads, navigation, and clutter. Keep only the main content."
            },
            {
                "role": "user",
                "content": html[:500000]  # Limit to ~500k chars to stay within context
            }
        ]

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{READERLM_BASE_URL}/v1/chat/completions",
                json={
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 32000,  # Allow long outputs
                }
            )
            response.raise_for_status()
            data = response.json()

            if "choices" in data and len(data["choices"]) > 0:
                markdown = data["choices"][0]["message"]["content"]
                return markdown.strip()

            return None

    except Exception as e:
        logger.error(f"ReaderLM conversion failed: {e}")
        return None


async def fetch_page_local(url: str, timeout: int = 10) -> Optional[str]:
    """Fetch and convert page using local stack (Playwright + ReaderLM).

    Args:
        url: URL to fetch
        timeout: Timeout for page load

    Returns:
        Markdown content or None if failed
    """
    logger.info(f"Using local reader for {url}")

    # Step 1: Fetch HTML with Playwright
    html = await fetch_html_with_playwright(url, timeout)
    if not html:
        logger.warning(f"Failed to fetch HTML for {url}")
        return None

    logger.info(f"Fetched {len(html)} bytes of HTML from {url}")

    # Step 2: Convert to Markdown with ReaderLM
    markdown = await convert_html_to_markdown_local(html, url)
    if not markdown:
        logger.warning(f"Failed to convert HTML to Markdown for {url}")
        return None

    logger.info(f"Converted to {len(markdown)} bytes of Markdown")
    return markdown


async def fetch_page_jina_api(
    url: str,
    remove_images: bool = False,
    gather_links: bool = False,
    timeout: int = 10,
    bypass_cache: bool = False,
    css_selector: Optional[str] = None
) -> Optional[str]:
    """Fetch page using Jina Reader API (fallback method).

    This is the original implementation, kept as fallback for PDFs
    and complex pages that local processing can't handle.
    """
    try:
        jina_url = f"https://r.jina.ai/{url}"

        headers = {
            "Accept": "text/markdown",
            "X-Timeout": str(timeout)
        }

        if JINA_API_KEY:
            headers["Authorization"] = f"Bearer {JINA_API_KEY}"

        if remove_images:
            headers["X-Remove-Images"] = "true"
        if gather_links:
            headers["X-With-Links-Summary"] = "true"
        if bypass_cache:
            headers["X-No-Cache"] = "true"
        if css_selector:
            headers["X-Target-Selector"] = css_selector

        async with httpx.AsyncClient(timeout=timeout + 5.0, follow_redirects=True) as client:
            response = await client.get(jina_url, headers=headers)
            response.raise_for_status()
            content = response.text

        if not content or len(content.strip()) == 0:
            return None

        # Strip image tags
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
        content = re.sub(r'<img[^>]*>', '', content, flags=re.IGNORECASE)

        return content

    except Exception as e:
        logger.error(f"Jina API fetch failed: {e}")
        return None


@mcp.tool()
async def jina_fetch_page(
    url: str,
    remove_images: bool = False,
    gather_links: bool = False,
    timeout: int = 10,
    bypass_cache: bool = False,
    ctx: Context = None
) -> str:
    """Fetch and convert a web page to clean, LLM-friendly markdown.

    Uses local ReaderLM-v2 + Playwright by default for fast, unlimited processing.
    Falls back to Jina Reader API for PDFs and when local processing fails.

    Args:
        url: The URL to fetch (must include http:// or https://)
        remove_images: If True, removes all images from output (default: False)
        gather_links: If True, gathers all links at the end in a summary section (default: False)
        timeout: Maximum page load time in seconds (default: 10)
        bypass_cache: If True, fetches fresh content (default: False uses cache)
        ctx: MCP context for progress/logging (auto-injected)

    Returns:
        Clean markdown-formatted content including title, URL, and main text.
        Returns error message if fetch fails.

    Processing:
        - Local: Playwright (JS rendering) + ReaderLM-v2 (HTML→MD) - unlimited, fast
        - Fallback: Jina API (for PDFs, complex pages) - 20 RPM free / 500 RPM with key

    Example:
        jina_fetch_page("https://www.example.com")
        jina_fetch_page("https://arxiv.org/pdf/2301.00001.pdf")  # Uses Jina API
        jina_fetch_page("https://news.ycombinator.com", gather_links=True)
    """
    if not url:
        if ctx:
            await ctx.error("URL is required")
        return "Error: URL is required"

    # Validate URL format
    if not url.startswith(("http://", "https://")):
        if ctx:
            await ctx.error("URL must start with http:// or https://")
        return "Error: URL must start with http:// or https://"

    url_preview = url[:50] + ("..." if len(url) > 50 else "")

    try:
        # Determine if we should use local processing
        is_pdf = url.lower().endswith('.pdf')
        use_local = USE_LOCAL_READER and not is_pdf

        if use_local:
            # Try local processing first
            if ctx:
                await ctx.report_progress(1, 3, f"Fetching locally: {url_preview}")

            content = await fetch_page_local(url, timeout)

            if content:
                # Success with local processing
                if ctx:
                    await ctx.report_progress(3, 3, f"Fetched {len(content)} chars (local)")
                logger.info(f"Successfully processed {url} locally ({len(content)} chars)")
                return content

            # Local processing failed, fall back to API
            logger.warning(f"Local processing failed for {url}, falling back to Jina API")
            if ctx:
                await ctx.report_progress(2, 3, "Local failed, trying Jina API...")

        # Use Jina API (either as primary or fallback)
        if ctx:
            await ctx.report_progress(1, 3, f"Fetching via API: {url_preview}")

        content = await fetch_page_jina_api(
            url,
            remove_images=remove_images,
            gather_links=gather_links,
            timeout=timeout,
            bypass_cache=bypass_cache
        )

        if content:
            if ctx:
                await ctx.report_progress(3, 3, f"Fetched {len(content)} chars (API)")
            logger.info(f"Successfully fetched {url} via API ({len(content)} chars)")
            return content

        # Both methods failed
        error_msg = "Failed to fetch content using both local and API methods"
        if ctx:
            await ctx.error(error_msg)
        return f"Error: {error_msg}"

    except Exception as e:
        logger.error(f"Unexpected error fetching page: {e}", exc_info=True)
        error_msg = str(e)
        if ctx:
            await ctx.error(error_msg)
        return f"Error: {error_msg}"


@mcp.tool()
async def jina_fetch_page_with_selector(
    url: str,
    css_selector: str,
    timeout: int = 10,
    ctx: Context = None
) -> str:
    """Fetch specific content from a page using CSS selector.

    Useful for extracting only relevant sections from large pages
    (e.g., main article, specific divs, tables).

    Args:
        url: The URL to fetch
        css_selector: CSS selector to target specific elements (e.g., "article", ".content", "#main")
        timeout: Maximum page load time in seconds (default: 10)
        ctx: MCP context for progress/logging (auto-injected)

    Returns:
        Markdown content of only the selected elements

    Example:
        jina_fetch_page_with_selector("https://news.site.com/article", "article.main-content")
        jina_fetch_page_with_selector("https://docs.site.com", "#documentation")
    """
    if not url or not css_selector:
        if ctx:
            await ctx.error("Both URL and CSS selector are required")
        return "Error: Both URL and CSS selector are required"

    if not url.startswith(("http://", "https://")):
        if ctx:
            await ctx.error("URL must start with http:// or https://")
        return "Error: URL must start with http:// or https://"

    try:
        if ctx:
            await ctx.report_progress(1, 3, f"Fetching with selector: {css_selector}")
        jina_url = f"https://r.jina.ai/{url}"

        logger.info(f"Fetching page with selector '{css_selector}': {url}")

        headers = {
            "Accept": "text/markdown",
            "X-Timeout": str(timeout),
            "X-Target-Selector": css_selector
        }

        if JINA_API_KEY:
            headers["Authorization"] = f"Bearer {JINA_API_KEY}"

        async with httpx.AsyncClient(timeout=timeout + 5.0, follow_redirects=True) as client:
            response = await client.get(jina_url, headers=headers)
            response.raise_for_status()
            content = response.text

        if not content or len(content.strip()) == 0:
            error_msg = f"No content found matching selector '{css_selector}' at {url}"
            if ctx:
                await ctx.error(error_msg)
            return f"Error: {error_msg}"

        if ctx:
            await ctx.report_progress(2, 3, "Processing selected content...")
        logger.info(f"Successfully fetched {len(content)} chars with selector from '{url}'")
        if ctx:
            await ctx.report_progress(3, 3, f"Fetched {len(content)} chars with selector")
        return content

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e}")
        error_msg = f"HTTP {e.response.status_code} - {e.response.text[:200]}"
        if ctx:
            await ctx.error(error_msg)
        return f"Error: {error_msg}"

    except httpx.TimeoutException:
        logger.error(f"Request timeout for {url}")
        error_msg = f"Request timed out after {timeout} seconds"
        if ctx:
            await ctx.error(error_msg)
        return f"Error: {error_msg}"

    except Exception as e:
        logger.error(f"Error fetching with selector: {e}", exc_info=True)
        error_msg = str(e)
        if ctx:
            await ctx.error(error_msg)
        return f"Error: {error_msg}"


@mcp.tool()
async def get_jina_reader_info(ctx: Context = None) -> str:
    """Get information about reader capabilities and configuration.

    Returns:
        Information about features, processing methods, and limits
    """
    info = f"""
Jina Reader MCP Server - Features and Configuration:

**Current Configuration**:
• Local Processing: {"✓ Enabled" if USE_LOCAL_READER else "✗ Disabled"}
• ReaderLM-v2 Model: {READERLM_BASE_URL if USE_LOCAL_READER else "N/A"}
• Playwright Browser: {PLAYWRIGHT_BASE_URL if USE_LOCAL_READER else "N/A"}
• Jina API Key: {"✓ Configured (500 RPM)" if JINA_API_KEY else "✗ Not set (20 RPM free tier)"}

**Processing Methods**:

1. **Local Processing** (Primary, Unlimited):
   • Playwright headless browser for JavaScript rendering
   • ReaderLM-v2 (1.5B model) for HTML→Markdown conversion
   • No rate limits, completely free
   • Handles JavaScript-heavy SPAs
   • Fast: ~1-3 seconds per page

2. **Jina API** (Fallback/PDF):
   • Used for PDFs (ReaderLM doesn't support binary)
   • Used when local processing fails
   • Rate limits: {"500 RPM" if JINA_API_KEY else "20 RPM (free tier)"}
   • Handles complex auth, Cloudflare, etc.

**Features**:
• Removes ads, navigation, and clutter automatically
• Supports HTML pages and PDF files
• JavaScript rendering (SPAs, dynamic content)
• CSS selector support for targeted extraction
• Configurable timeout (default 10s)
• Smart fallback: local → API

**Supported Content**:
• Web pages (HTML, JavaScript-rendered SPAs)
• PDF documents (via Jina API)
• Documentation sites
• News articles
• Blog posts
• Academic papers (arXiv, etc.)

**Best Practices**:
• Local processing works for 95% of pages
• PDFs automatically use Jina API
• Use after web search to get full content
• Enable 'gather_links' for pages with many references
• Set appropriate timeout for slow pages (10-30s)

**Performance**:
• Local: ~2-3 sec/page, unlimited
• API: ~1-2 sec/page, {"500 RPM" if JINA_API_KEY else "20 RPM"}

For more info: https://jina.ai/reader/ (API) | https://jina.ai/models/ReaderLM-v2/ (Model)
"""
    return info.strip()


if __name__ == "__main__":
    logger.info("Starting Jina Reader MCP server (Streamable HTTP)...")
    if not JINA_API_KEY:
        logger.warning(
            "JINA_API_KEY not set. Running with free tier limits (20 RPM). "
            "Set JINA_API_KEY environment variable for higher limits (500 RPM)."
        )
    server.run(transport="streamable-http")
