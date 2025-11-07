"""ABOUTME: Playwright web scraper FastAPI server.

Provides REST API for fetching web pages with automatic JavaScript rendering.
Implements smart fallback: tries simple HTTP first, falls back to Playwright if needed.

Privacy-first: All fetching happens locally, URLs never transmitted externally.
"""

import asyncio
import logging
import re
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global playwright instance
playwright_instance = None


class ScrapeRequest(BaseModel):
    """Request model for scraping."""
    url: str
    wait_for_selector: Optional[str] = None
    timeout: int = 30  # seconds
    force_playwright: bool = False  # Force Playwright even for simple pages


class ScrapeResponse(BaseModel):
    """Response model for scraping."""
    url: str
    html: str
    method: str  # "http" or "playwright"
    success: bool
    error: Optional[str] = None


async def requires_js_rendering(url: str, html: str = None) -> bool:
    """
    Detect if a page likely requires JavaScript rendering.

    Heuristics:
    - Known SPA frameworks (Twitter, Reddit, Medium, etc.)
    - Empty body or minimal content
    - React/Vue/Angular markers in HTML
    """
    spa_domains = [
        "twitter.com",
        "reddit.com",
        "medium.com",
        "linkedin.com",
        "facebook.com",
        "instagram.com",
        "youtube.com",
        "google.com/search",
    ]

    if any(domain in url for domain in spa_domains):
        return True

    # Check HTML content if provided
    if html:
        # Look for empty body or SPA markers
        if len(html) < 500:
            return True

        spa_markers = [
            r'<div id="app"[^>]*>',
            r'<div id="root"[^>]*>',
            r'<noscript>',
            r'__NEXT_DATA__',
            r'__NUXT__',
        ]

        if any(re.search(marker, html, re.IGNORECASE) for marker in spa_markers):
            return True

    return False


async def scrape_with_http(url: str, timeout: int = 30) -> tuple[str, bool]:
    """
    Attempt simple HTTP fetch.

    Returns: (html_content, success)
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
                },
                follow_redirects=True
            )
            response.raise_for_status()
            return response.text, True
    except Exception as e:
        logger.warning(f"HTTP fetch failed for {url}: {e}")
        return None, False


async def scrape_with_playwright(
    url: str,
    wait_for_selector: Optional[str] = None,
    timeout: int = 30
) -> tuple[str, bool]:
    """
    Scrape with Playwright for JavaScript rendering.

    Returns: (html_content, success)
    """
    global playwright_instance

    if playwright_instance is None:
        logger.error("Playwright not initialized")
        return None, False

    browser = None
    try:
        browser = await playwright_instance.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        page = await context.new_page()
        page.set_default_timeout(timeout * 1000)

        logger.info(f"Navigating to {url}")
        try:
            await page.goto(url, wait_until="networkidle")
        except PlaywrightTimeoutError:
            logger.warning(f"Navigation timeout for {url}, trying with domcontentloaded")
            try:
                await page.goto(url, wait_until="domcontentloaded")
            except Exception as e:
                logger.error(f"Failed to load {url}: {e}")
                return None, False

        # Wait for optional selector
        if wait_for_selector:
            try:
                await page.wait_for_selector(wait_for_selector, timeout=timeout * 1000)
                logger.info(f"Found selector: {wait_for_selector}")
            except Exception as e:
                logger.warning(f"Selector {wait_for_selector} not found: {e}")

        html = await page.content()
        await context.close()
        await browser.close()

        return html, True

    except Exception as e:
        logger.error(f"Playwright scrape failed for {url}: {e}")
        if browser:
            try:
                await browser.close()
            except:
                pass
        return None, False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage Playwright lifecycle.
    """
    global playwright_instance

    logger.info("Initializing Playwright")
    playwright_context = await async_playwright().start()
    playwright_instance = playwright_context

    yield

    logger.info("Shutting down Playwright")
    await playwright_context.stop()


app = FastAPI(
    title="Playwright Scraper",
    description="Web scraping API with JavaScript rendering support",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/scrape", response_model=ScrapeResponse)
async def scrape(request: ScrapeRequest) -> ScrapeResponse:
    """
    Scrape a web page with smart fallback strategy.

    Strategy:
    1. Try simple HTTP first (faster)
    2. If HTML looks empty or requires JS, fall back to Playwright
    3. If force_playwright=true, skip HTTP and use Playwright directly
    """
    logger.info(f"Scrape request: {request.url}")

    if request.force_playwright:
        logger.info(f"Force Playwright mode for {request.url}")
        html, success = await scrape_with_playwright(
            request.url,
            request.wait_for_selector,
            request.timeout
        )
        if success:
            return ScrapeResponse(
                url=request.url,
                html=html,
                method="playwright",
                success=True
            )
        else:
            return ScrapeResponse(
                url=request.url,
                html="",
                method="playwright",
                success=False,
                error="Playwright scrape failed"
            )

    # Try HTTP first
    html, http_success = await scrape_with_http(request.url, request.timeout)

    if http_success:
        # Check if we actually need JS rendering
        needs_js = await requires_js_rendering(request.url, html)

        if not needs_js:
            logger.info(f"HTTP fetch successful for {request.url}")
            return ScrapeResponse(
                url=request.url,
                html=html,
                method="http",
                success=True
            )
        else:
            logger.info(f"HTTP content appears incomplete, trying Playwright for {request.url}")

    # Fall back to Playwright
    html, pw_success = await scrape_with_playwright(
        request.url,
        request.wait_for_selector,
        request.timeout
    )

    if pw_success:
        return ScrapeResponse(
            url=request.url,
            html=html,
            method="playwright",
            success=True
        )
    else:
        return ScrapeResponse(
            url=request.url,
            html="",
            method="playwright",
            success=False,
            error="Both HTTP and Playwright scraping failed"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "playwright-scraper",
        "playwright_ready": playwright_instance is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
