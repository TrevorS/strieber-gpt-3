"""Playwright HTML Fetcher Service

Provides headless browser automation for fetching JavaScript-rendered web pages.
Used by the jina-reader MCP server for local HTML-to-Markdown conversion.

Features:
- Headless Chromium browser
- JavaScript execution support
- Resource blocking (images, CSS) for speed
- Configurable wait strategies
- Simple HTTP API
"""

import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager

from playwright.async_api import async_playwright, Browser, TimeoutError as PlaywrightTimeoutError
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global browser instance
browser: Optional[Browser] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage browser lifecycle"""
    global browser
    logger.info("Starting Playwright browser...")
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        headless=True,
        args=[
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu',
        ]
    )
    logger.info("Browser started successfully")
    yield
    logger.info("Shutting down browser...")
    await browser.close()
    await playwright.stop()


app = FastAPI(
    title="Playwright HTML Fetcher",
    description="Headless browser service for fetching JavaScript-rendered web pages",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "browser": "running" if browser else "stopped"}


@app.get("/fetch")
async def fetch_html(
    url: str = Query(..., description="URL to fetch"),
    timeout: int = Query(10, description="Page load timeout in seconds", ge=1, le=60),
    wait_for: str = Query("networkidle", description="Wait strategy: load, domcontentloaded, networkidle"),
    block_resources: bool = Query(True, description="Block images and CSS for faster loading"),
    user_agent: Optional[str] = Query(None, description="Custom user agent")
):
    """Fetch HTML content from a URL using headless browser

    Args:
        url: The URL to fetch
        timeout: Maximum time to wait for page load (seconds)
        wait_for: Wait strategy - 'load', 'domcontentloaded', or 'networkidle'
        block_resources: If True, blocks images and CSS for faster loading
        user_agent: Optional custom user agent string

    Returns:
        JSON with 'html' field containing rendered HTML content
    """
    if not browser:
        raise HTTPException(status_code=503, detail="Browser not initialized")

    # Validate URL
    if not url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    # Validate wait_for parameter
    if wait_for not in ['load', 'domcontentloaded', 'networkidle']:
        raise HTTPException(status_code=400, detail="wait_for must be 'load', 'domcontentloaded', or 'networkidle'")

    try:
        logger.info(f"Fetching URL: {url} (timeout={timeout}s, wait_for={wait_for}, block_resources={block_resources})")

        # Create browser context with optional user agent
        context_options = {}
        if user_agent:
            context_options['user_agent'] = user_agent
        else:
            # Default user agent to avoid bot detection
            context_options['user_agent'] = (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )

        context = await browser.new_context(**context_options)
        page = await context.new_page()

        # Block unnecessary resources if requested
        if block_resources:
            async def route_handler(route):
                if route.request.resource_type in ['image', 'stylesheet', 'font', 'media']:
                    await route.abort()
                else:
                    await route.continue_()

            await page.route('**/*', route_handler)

        # Navigate to URL with timeout
        await page.goto(
            url,
            wait_until=wait_for,
            timeout=timeout * 1000  # Convert to milliseconds
        )

        # Get rendered HTML
        html = await page.content()

        # Close context
        await context.close()

        logger.info(f"Successfully fetched {len(html)} bytes from {url}")

        return JSONResponse(content={
            "url": url,
            "html": html,
            "length": len(html)
        })

    except PlaywrightTimeoutError:
        logger.error(f"Timeout fetching {url} after {timeout}s")
        raise HTTPException(
            status_code=504,
            detail=f"Page load timeout after {timeout} seconds. Try increasing timeout or changing wait_for strategy."
        )

    except Exception as e:
        logger.error(f"Error fetching {url}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch page: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8005,
        log_level="info"
    )
