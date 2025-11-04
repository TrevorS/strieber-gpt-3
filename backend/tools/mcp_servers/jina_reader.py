"""ABOUTME: Jina Reader MCP Server - Web content extraction and markdown conversion.

Provides web page content retrieval via Jina Reader API.
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

# Get API key from environment
JINA_API_KEY = os.getenv("JINA_API_KEY")
if not JINA_API_KEY:
    logger.warning("JINA_API_KEY not set. Jina Reader tool will use free tier (20 RPM limit).")


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

    Uses Jina Reader API to extract main content from URLs, removing ads,
    navigation, and other clutter. Supports HTML pages and PDFs.

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

    Rate Limits:
        - With API key: 500 requests per minute
        - Without API key: 20 requests per minute

    Example:
        jina_fetch_page("https://www.example.com")
        jina_fetch_page("https://arxiv.org/pdf/2301.00001.pdf")
        jina_fetch_page("https://news.ycombinator.com", gather_links=True)
    """
    if not url:
        if ctx:
            await ctx.error("URL is required")
        return {"content": "Error: URL is required", "error": True}

    # Validate URL format
    if not url.startswith(("http://", "https://")):
        if ctx:
            await ctx.error("URL must start with http:// or https://")
        return {"content": "Error: URL must start with http:// or https://", "error": True}

    try:
        url_preview = url[:50] + ("..." if len(url) > 50 else "")
        if ctx:
            await ctx.report_progress(1, 3, f"Fetching page: {url_preview}")
        # Build Jina Reader URL
        # Format: https://r.jina.ai/{url}
        jina_url = f"https://r.jina.ai/{url}"

        logger.info(f"Fetching page via Jina Reader: '{url}'")

        # Build headers
        headers = {
            "Accept": "text/markdown",
            "X-Timeout": str(timeout)
        }

        # Add API key if available (for higher rate limits)
        if JINA_API_KEY:
            headers["Authorization"] = f"Bearer {JINA_API_KEY}"

        # Add optional headers
        if remove_images:
            headers["X-Remove-Images"] = "true"

        if gather_links:
            headers["X-With-Links-Summary"] = "true"

        if bypass_cache:
            headers["X-No-Cache"] = "true"

        # Make API request
        async with httpx.AsyncClient(timeout=timeout + 5.0, follow_redirects=True) as client:
            response = await client.get(jina_url, headers=headers)
            response.raise_for_status()

            content = response.text

        if not content or len(content.strip()) == 0:
            if ctx:
                await ctx.error("No content retrieved")
            return {"content": f"Error: No content retrieved from URL: {url}", "error": True}

        # Strip image tags from content to avoid empty src errors
        # Remove markdown images: ![alt text](url)
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
        # Remove HTML images: <img ...>
        content = re.sub(r'<img[^>]*>', '', content, flags=re.IGNORECASE)

        if ctx:
            await ctx.report_progress(2, 3, f"Processing {len(content)} chars...")

        logger.info(f"Successfully fetched {len(content)} chars from '{url}'")

        if ctx:
            await ctx.report_progress(3, 3, f"Fetched {len(content)} chars")

        return content

    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        error_msg = ""
        if status_code == 429:
            logger.warning("Jina Reader rate limit exceeded")
            if JINA_API_KEY:
                error_msg = "Rate limit exceeded (500 RPM with API key). Please retry in a moment."
            else:
                error_msg = "Rate limit exceeded (20 RPM without API key). Set JINA_API_KEY for higher limits."
        elif status_code == 401:
            logger.error("Invalid Jina API key")
            error_msg = "Invalid JINA_API_KEY"
        elif status_code == 404:
            logger.error(f"Page not found: {url}")
            error_msg = f"Page not found (404): {url}"
        elif status_code == 403:
            logger.error(f"Access forbidden: {url}")
            error_msg = f"Access forbidden (403). Page may require authentication: {url}"
        else:
            logger.error(f"HTTP error {status_code}: {e}")
            error_msg = f"HTTP {status_code} - {e.response.text[:200]}"

        if ctx:
            await ctx.error(error_msg)
        return {"content": f"Error: {error_msg}", "error": True}

    except httpx.TimeoutException:
        logger.error(f"Request timeout after {timeout}s for {url}")
        error_msg = f"Request timed out after {timeout} seconds. Page may be very large or slow to load."
        if ctx:
            await ctx.error(error_msg)
        return {"content": f"Error: {error_msg}", "error": True}

    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        error_msg = f"Failed to connect to Jina Reader API - {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return {"content": f"Error: {error_msg}", "error": True}

    except Exception as e:
        logger.error(f"Unexpected error fetching page: {e}", exc_info=True)
        error_msg = str(e)
        if ctx:
            await ctx.error(error_msg)
        return {"content": f"Error: {error_msg}", "error": True}


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
        return {"content": "Error: Both URL and CSS selector are required", "error": True}

    if not url.startswith(("http://", "https://")):
        if ctx:
            await ctx.error("URL must start with http:// or https://")
        return {"content": "Error: URL must start with http:// or https://", "error": True}

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
            return {"content": f"Error: {error_msg}", "error": True}

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
        return {"content": f"Error: {error_msg}", "error": True}

    except httpx.TimeoutException:
        logger.error(f"Request timeout for {url}")
        error_msg = f"Request timed out after {timeout} seconds"
        if ctx:
            await ctx.error(error_msg)
        return {"content": f"Error: {error_msg}", "error": True}

    except Exception as e:
        logger.error(f"Error fetching with selector: {e}", exc_info=True)
        error_msg = str(e)
        if ctx:
            await ctx.error(error_msg)
        return {"content": f"Error: {error_msg}", "error": True}


@mcp.tool()
async def get_jina_reader_info(ctx: Context = None) -> str:
    """Get information about Jina Reader API capabilities and limits.

    Returns:
        Information about features, rate limits, and usage guidelines
    """
    info = f"""
Jina Reader API - Features and Limits:

**Current Configuration**:
• API Key: {"✓ Configured" if JINA_API_KEY else "✗ Not set (using free tier)"}
• Rate Limit: {"500 RPM" if JINA_API_KEY else "20 RPM"}

**Features**:
• Converts URLs to clean markdown (removes ads, nav, clutter)
• Supports HTML pages and PDF files
• Image captioning (automatic alt text generation)
• CSS selector support for targeted extraction
• Configurable timeout (default 10s)
• Caching enabled by default for faster repeated access

**Supported Content**:
• Web pages (HTML, JavaScript-rendered)
• PDF documents (including image-heavy PDFs)
• Documentation sites
• News articles
• Blog posts
• Academic papers (arXiv, etc.)

**Best Practices**:
• Use after web search to get full content (vs snippets)
• Enable 'gather_links' for pages with many references
• Use CSS selectors to extract only relevant sections from large pages
• Bypass cache for time-sensitive content (news, stock prices)
• Set appropriate timeout for slow pages (docs, large PDFs)

**Cost**:
• Free tier: 10 million tokens included
• Pricing: ~$0.05 per 1M output tokens
• Typical article: 1-5k tokens

For more info, see: https://jina.ai/reader/
"""
    return info.strip()


if __name__ == "__main__":
    logger.info("Starting Jina Reader MCP server (Streamable HTTP)...")
    if not JINA_API_KEY:
        logger.warning(
            "JINA_API_KEY not set. Running with free tier limits (20 RPM). "
            "Set JINA_API_KEY environment variable for higher limits (500 RPM)."
        )
    mcp.run(transport="streamable-http")
