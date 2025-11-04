"""ABOUTME: Page Reader MCP Server - Local HTML-to-Markdown conversion with content extraction.

Provides web page content retrieval via local ReaderLM-v2 model with Playwright.
Uses Mozilla Readability for clean content extraction, optimized for LLM consumption.
Returns structured data similar to web_search tool for consistency.
"""

import asyncio
import os
import logging
import re
from typing import Optional, Dict, List
from urllib.parse import urlparse
from dataclasses import dataclass, asdict

import httpx
from bs4 import BeautifulSoup
from readabilipy import simple_json_from_html_string
from mcp.server.fastmcp import Context

from common.mcp_base import MCPServerBase

# Initialize MCP server with base class
server = MCPServerBase("page-reader")
mcp = server.get_mcp()
logger = server.get_logger()

# Configuration from environment
READERLM_BASE_URL = os.getenv("READERLM_BASE_URL", "http://llama-server-reader:8004")
PLAYWRIGHT_BASE_URL = os.getenv("PLAYWRIGHT_BASE_URL", "http://playwright-fetcher:8005")

logger.info(f"Page Reader initialized: ReaderLM @ {READERLM_BASE_URL}, Playwright @ {PLAYWRIGHT_BASE_URL}")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PageMetadata:
    """Metadata extracted from a web page"""
    url: str
    title: Optional[str] = None
    author: Optional[str] = None
    site_name: Optional[str] = None
    published_date: Optional[str] = None
    word_count: int = 0
    reading_time_minutes: int = 0
    language: Optional[str] = None
    excerpt: Optional[str] = None


@dataclass
class PageLink:
    """A link extracted from the page"""
    text: str
    url: str
    is_external: bool = False


@dataclass
class PageImage:
    """An image extracted from the page"""
    url: str
    alt: Optional[str] = None
    title: Optional[str] = None


@dataclass
class PageContent:
    """Complete structured page content optimized for LLM consumption"""
    content: str  # Main content in markdown
    metadata: PageMetadata
    links: List[PageLink]
    images: List[PageImage]
    sections: List[str]  # Heading structure


# ============================================================================
# HTML Fetching
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


# ============================================================================
# Content Extraction
# ============================================================================

def extract_with_readability(html: str, url: str) -> Dict:
    """Extract clean content using Mozilla Readability algorithm.

    Args:
        html: Raw HTML content
        url: Source URL for link resolution

    Returns:
        Readability extraction result with title, content, etc.
    """
    try:
        # Use Mozilla's Readability.js (requires Node.js, but much better quality)
        result = simple_json_from_html_string(html, use_readability=True)
        return result
    except Exception as e:
        logger.error(f"Readability extraction failed: {e}")
        return {}


def extract_metadata(soup: BeautifulSoup, url: str, readability_data: Dict) -> PageMetadata:
    """Extract comprehensive metadata from HTML and readability results.

    Args:
        soup: BeautifulSoup parsed HTML
        url: Page URL
        readability_data: Readability extraction results

    Returns:
        PageMetadata object with all available metadata
    """
    # Try to get title from multiple sources
    title = readability_data.get("title")
    if not title:
        title_tag = soup.find("title")
        title = title_tag.text.strip() if title_tag else None

    # Extract author
    author = readability_data.get("byline")
    if not author:
        author_meta = soup.find("meta", attrs={"name": "author"}) or \
                      soup.find("meta", attrs={"property": "article:author"})
        author = author_meta.get("content") if author_meta else None

    # Extract site name
    site_name_meta = soup.find("meta", attrs={"property": "og:site_name"})
    site_name = site_name_meta.get("content") if site_name_meta else None

    # Extract publication date
    date_meta = soup.find("meta", attrs={"property": "article:published_time"}) or \
                soup.find("meta", attrs={"name": "date"})
    published_date = date_meta.get("content") if date_meta else None

    # Extract language
    html_tag = soup.find("html")
    language = html_tag.get("lang") if html_tag else None

    # Extract excerpt/description
    desc_meta = soup.find("meta", attrs={"name": "description"}) or \
                soup.find("meta", attrs={"property": "og:description"})
    excerpt = desc_meta.get("content") if desc_meta else readability_data.get("excerpt")

    # Calculate word count and reading time from plain text
    plain_text = readability_data.get("plain_text", [])
    if isinstance(plain_text, list):
        plain_text = " ".join(plain_text)

    word_count = len(plain_text.split()) if plain_text else 0
    reading_time_minutes = max(1, word_count // 200)  # Assume 200 WPM reading speed

    return PageMetadata(
        url=url,
        title=title,
        author=author,
        site_name=site_name,
        published_date=published_date,
        word_count=word_count,
        reading_time_minutes=reading_time_minutes,
        language=language,
        excerpt=excerpt
    )


def extract_links(soup: BeautifulSoup, base_url: str) -> List[PageLink]:
    """Extract all meaningful links from the page.

    Args:
        soup: BeautifulSoup parsed HTML
        base_url: Base URL for resolving relative links

    Returns:
        List of PageLink objects
    """
    links = []
    base_domain = urlparse(base_url).netloc

    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get("href", "").strip()
        text = a_tag.get_text(strip=True)

        # Skip empty links, anchors, javascript
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue

        # Resolve relative URLs
        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            parsed_base = urlparse(base_url)
            href = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
        elif not href.startswith("http"):
            # Relative path
            href = base_url.rstrip("/") + "/" + href.lstrip("/")

        # Determine if external
        link_domain = urlparse(href).netloc
        is_external = link_domain != base_domain if link_domain else False

        if text:  # Only keep links with text
            links.append(PageLink(text=text, url=href, is_external=is_external))

    # Deduplicate by URL
    seen_urls = set()
    unique_links = []
    for link in links:
        if link.url not in seen_urls:
            seen_urls.add(link.url)
            unique_links.append(link)

    logger.debug(f"Extracted {len(unique_links)} unique links")
    return unique_links


def extract_images(soup: BeautifulSoup) -> List[PageImage]:
    """Extract all images from the page.

    Args:
        soup: BeautifulSoup parsed HTML

    Returns:
        List of PageImage objects
    """
    images = []

    for img_tag in soup.find_all("img", src=True):
        src = img_tag.get("src", "").strip()
        alt = img_tag.get("alt", "").strip()
        title = img_tag.get("title", "").strip()

        # Skip data URIs and tiny images
        if src.startswith("data:") or not src:
            continue

        images.append(PageImage(url=src, alt=alt or None, title=title or None))

    logger.debug(f"Extracted {len(images)} images")
    return images


def extract_sections(soup: BeautifulSoup) -> List[str]:
    """Extract heading structure from the page.

    Args:
        soup: BeautifulSoup parsed HTML

    Returns:
        List of headings in document order
    """
    sections = []

    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        text = heading.get_text(strip=True)
        if text:
            sections.append(text)

    logger.debug(f"Extracted {len(sections)} section headings")
    return sections


# ============================================================================
# Markdown Conversion
# ============================================================================

async def convert_html_to_markdown(html: str, url: str) -> str:
    """Convert HTML to Markdown using local ReaderLM-v2 model.

    Args:
        html: HTML content to convert
        url: Original URL (for context)

    Returns:
        Markdown content
    """
    try:
        # Prepare messages for ReaderLM-v2 (uses ChatML format)
        messages = [
            {
                "role": "system",
                "content": "Convert the HTML to Markdown. Preserve structure, headings, links, and formatting. Remove ads and navigation."
            },
            {
                "role": "user",
                "content": html[:500000]  # Limit to ~500k chars
            }
        ]

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{READERLM_BASE_URL}/v1/chat/completions",
                json={
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 32000,
                }
            )
            response.raise_for_status()
            data = response.json()

            if "choices" in data and len(data["choices"]) > 0:
                markdown = data["choices"][0]["message"]["content"]
                return markdown.strip()

            return ""

    except Exception as e:
        logger.error(f"ReaderLM conversion failed: {e}")
        return ""


# ============================================================================
# Main Processing Pipeline
# ============================================================================

async def process_page(url: str, timeout: int = 10, include_links: bool = False, include_images: bool = False) -> Optional[PageContent]:
    """Complete pipeline: Fetch → Extract → Convert → Structure

    Args:
        url: URL to process
        timeout: Fetch timeout in seconds
        include_links: Extract all links from page
        include_images: Extract all images from page

    Returns:
        PageContent object with structured data or None if failed
    """
    logger.info(f"Processing page: {url}")

    # Step 1: Fetch HTML
    html = await fetch_html_with_playwright(url, timeout)
    if not html:
        logger.error(f"Failed to fetch HTML for {url}")
        return None

    logger.debug(f"Fetched {len(html)} bytes of HTML")

    # Step 2: Parse with BeautifulSoup
    soup = BeautifulSoup(html, "lxml")

    # Step 3: Extract clean content with Readability
    readability_data = extract_with_readability(html, url)

    # Step 4: Extract metadata
    metadata = extract_metadata(soup, url, readability_data)

    # Step 5: Extract links and images (optional)
    links = extract_links(soup, url) if include_links else []
    images = extract_images(soup) if include_images else []

    # Step 6: Extract section structure
    sections = extract_sections(soup)

    # Step 7: Get clean HTML from readability
    clean_html = readability_data.get("content", "") or html

    # Step 8: Convert to Markdown with ReaderLM
    markdown = await convert_html_to_markdown(clean_html, url)
    if not markdown:
        logger.error(f"Failed to convert HTML to Markdown for {url}")
        return None

    logger.info(f"Successfully processed {url}: {metadata.word_count} words, {len(markdown)} chars markdown")

    return PageContent(
        content=markdown,
        metadata=metadata,
        links=links,
        images=images,
        sections=sections
    )


# ============================================================================
# MCP Tool Definitions
# ============================================================================

@mcp.tool()
async def fetch_page(
    url: str,
    timeout: int = 10,
    include_links: bool = False,
    include_images: bool = False,
    ctx: Context = None
) -> dict:
    """Fetch and convert a web page to clean markdown with structured metadata.

    Returns structured data optimized for LLM consumption, similar to web_search tool.
    Uses local processing (Playwright + Readability + ReaderLM-v2) for unlimited, free usage.

    Args:
        url: The URL to fetch (must include http:// or https://)
        timeout: Maximum page load time in seconds (default: 10)
        include_links: Extract all links from the page (default: False)
        include_images: Extract all images from the page (default: False)
        ctx: MCP context for progress/logging (auto-injected)

    Returns:
        Dict with:
        - content: Main page content in markdown format
        - metadata: Page metadata (title, author, word count, etc.)
        - links: List of links (if include_links=True)
        - images: List of images (if include_images=True)
        - sections: List of section headings

    Processing Pipeline:
        URL → Playwright (JS rendering) → Readability (content extraction) →
        ReaderLM-v2 (HTML→MD) → Structured output

    Performance:
        - Average: 2-5 seconds per page
        - No rate limits, completely free
        - Handles JavaScript-heavy sites

    Examples:
        fetch_page("https://example.com/article")
        fetch_page("https://docs.python.org/3/tutorial/", include_links=True)
        fetch_page("https://blog.example.com", include_images=True, include_links=True)
    """
    if not url:
        if ctx:
            await ctx.error("URL is required")
        return {"error": "URL is required"}

    if not url.startswith(("http://", "https://")):
        if ctx:
            await ctx.error("URL must start with http:// or https://")
        return {"error": "URL must start with http:// or https://"}

    url_preview = url[:50] + ("..." if len(url) > 50 else "")

    try:
        if ctx:
            await ctx.report_progress(1, 4, f"Fetching: {url_preview}")

        result = await process_page(url, timeout, include_links, include_images)

        if not result:
            error_msg = "Failed to process page"
            if ctx:
                await ctx.error(error_msg)
            return {"error": error_msg}

        if ctx:
            await ctx.report_progress(4, 4, f"Processed {result.metadata.word_count} words")

        # Convert to dict for JSON serialization
        return {
            "content": result.content,
            "metadata": asdict(result.metadata),
            "links": [asdict(link) for link in result.links],
            "images": [asdict(image) for image in result.images],
            "sections": result.sections
        }

    except Exception as e:
        logger.error(f"Error processing page: {e}", exc_info=True)
        error_msg = str(e)
        if ctx:
            await ctx.error(error_msg)
        return {"error": error_msg}


if __name__ == "__main__":
    logger.info("Starting Page Reader MCP server (Streamable HTTP)...")
    server.run(transport="streamable-http")
