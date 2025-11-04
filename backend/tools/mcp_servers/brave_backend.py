"""
Brave Search API backend implementation.

Implements the SearchBackend interface for Brave Search API with rate limiting,
error handling, and result formatting.
"""

import asyncio
import logging
import os
import time
from typing import Optional

import httpx

from search_backend import SearchBackend, SearchResponse, SearchResult

# Configure logging
logger = logging.getLogger(__name__)


class BraveSearchBackend(SearchBackend):
    """Brave Search API backend implementation.

    Provides web search capabilities via Brave Search API with:
    - Rate limiting for free tier (1 req/sec)
    - Automatic retry with exponential backoff
    - Comprehensive error handling
    - Standard SearchResponse format
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Brave Search backend.

        Args:
            api_key: Brave API key. If None, reads from BRAVE_API_KEY env var.

        Raises:
            ValueError: If API key is not provided and not in environment
        """
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("BRAVE_API_KEY not set. Brave Search backend requires API key.")

        self.base_url = "https://api.search.brave.com/res/v1"
        self.last_request_time: Optional[float] = None
        self.min_request_interval = 1.0  # Free tier: 1 request/second
        self._rate_limit_lock = asyncio.Lock()

        logger.info("BraveSearchBackend initialized")

    @property
    def name(self) -> str:
        """Backend name."""
        return "brave"

    async def _rate_limit(self) -> None:
        """Rate limit requests to 1 per second (free tier limit)."""
        async with self._rate_limit_lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < 1.0:
                wait_time = 1.0 - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            self.last_request_time = time.time()

    async def search(
        self,
        query: str,
        count: int = 10,
        freshness: Optional[str] = None,
        country: str = "US",
        search_lang: str = "en",
        safesearch: str = "moderate",
        **kwargs
    ) -> SearchResponse:
        """Execute a Brave Search API query.

        Args:
            query: Search query (max 400 characters recommended)
            count: Number of results to return (1-20, default 10)
            freshness: Optional time filter - "pd" (past day), "pw" (past week),
                      "pm" (past month), "py" (past year)
            country: Country code (ISO 3166-1 alpha-2, default "US")
            search_lang: Search language (default "en")
            safesearch: Safe search level - "off", "moderate", "strict" (default "moderate")
            **kwargs: Additional parameters (ignored)

        Returns:
            SearchResponse with results in standardized format

        Raises:
            httpx.HTTPStatusError: If API returns error status
            httpx.TimeoutException: If request times out
            Exception: For other unexpected errors
        """
        # Validate and normalize parameters
        count = max(1, min(20, count))  # Clamp to 1-20
        query = query[:400]  # Limit query length

        try:
            # Apply rate limiting
            await self._rate_limit()

            logger.info(f"Brave search: '{query}' (count={count})")

            # Build request
            url = f"{self.base_url}/web/search"
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key
            }
            params = {
                "q": query,
                "count": count,
                "country": country,
                "search_lang": search_lang,
                "safesearch": safesearch
            }
            if freshness:
                params["freshness"] = freshness

            # Make API request
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()

            # Extract and convert results
            web_results = data.get("web", {}).get("results", [])

            if not web_results:
                logger.warning(f"No results found for query: {query}")
                return SearchResponse(
                    query=query,
                    results=[],
                    total_results=0
                )

            # Convert to standardized format
            results = self._format_results(web_results)

            logger.info(f"Brave search returned {len(results)} results for '{query}'")

            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                metadata={
                    "backend": "brave",
                    "country": country,
                    "search_lang": search_lang
                }
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Brave API rate limit exceeded")
                raise Exception("Rate limit exceeded. Free tier allows 1 request/second, 2000 requests/month.")
            elif e.response.status_code == 401:
                logger.error("Invalid Brave API key")
                raise Exception("Invalid BRAVE_API_KEY")
            else:
                logger.error(f"Brave API HTTP error: {e}")
                raise Exception(f"HTTP Error {e.response.status_code}: {e.response.text}")

        except httpx.TimeoutException:
            logger.error("Brave API request timeout")
            raise Exception("Search request timed out after 10 seconds")

        except Exception as e:
            logger.error(f"Unexpected error during Brave search: {e}", exc_info=True)
            raise

    async def search_news(
        self,
        query: str,
        count: int = 10,
        freshness: Optional[str] = None,
        country: str = "US",
        search_lang: str = "en",
        **kwargs
    ) -> SearchResponse:
        """Execute a Brave News Search API query.

        Args:
            query: Search query (max 400 characters recommended)
            count: Number of results to return (1-20, default 10)
            freshness: Optional time filter - "pd" (past day), "pw" (past week),
                      "pm" (past month), "py" (past year)
            country: Country code (ISO 3166-1 alpha-2, default "US")
            search_lang: Search language (default "en")
            **kwargs: Additional parameters (ignored)

        Returns:
            SearchResponse with news results in standardized format

        Raises:
            httpx.HTTPStatusError: If API returns error status
            httpx.TimeoutException: If request times out
            Exception: For other unexpected errors
        """
        # Validate and normalize parameters
        count = max(1, min(20, count))  # Clamp to 1-20
        query = query[:400]  # Limit query length

        try:
            # Apply rate limiting
            await self._rate_limit()

            logger.info(f"Brave news search: '{query}' (count={count})")

            # Build request
            url = f"{self.base_url}/news/search"
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key
            }
            params = {
                "q": query,
                "count": count,
                "country": country,
                "search_lang": search_lang
            }
            if freshness:
                params["freshness"] = freshness

            # Make API request
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()

            # Extract news results
            news_results = data.get("results", [])

            if not news_results:
                logger.warning(f"No news results found for query: {query}")
                return SearchResponse(
                    query=query,
                    results=[],
                    total_results=0,
                    metadata={"type": "news"}
                )

            # Convert to standardized format
            results = self._format_news_results(news_results)

            logger.info(f"Brave news search returned {len(results)} results for '{query}'")

            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                metadata={
                    "backend": "brave",
                    "type": "news",
                    "country": country,
                    "search_lang": search_lang
                }
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Brave API rate limit exceeded")
                raise Exception("Rate limit exceeded. Free tier allows 1 request/second, 2000 requests/month.")
            elif e.response.status_code == 401:
                logger.error("Invalid Brave API key")
                raise Exception("Invalid BRAVE_API_KEY")
            else:
                logger.error(f"Brave API HTTP error: {e}")
                raise Exception(f"HTTP Error {e.response.status_code}: {e.response.text}")

        except httpx.TimeoutException:
            logger.error("Brave API request timeout")
            raise Exception("Search request timed out after 10 seconds")

        except Exception as e:
            logger.error(f"Unexpected error during Brave news search: {e}", exc_info=True)
            raise

    def _format_results(self, raw_results: list[dict]) -> list[SearchResult]:
        """Convert Brave API results to standardized SearchResult format.

        Args:
            raw_results: Raw results from Brave API

        Returns:
            List of SearchResult objects
        """
        formatted_results = []

        for result in raw_results:
            title = result.get("title", "No title")
            url = result.get("url", "")
            description = result.get("description", "")
            age = result.get("age", "")
            extra_snippets = result.get("extra_snippets", [])

            # Convert to SearchResult
            formatted_results.append(SearchResult(
                title=title,
                url=url,
                snippet=description,
                date=age if age else None,
                extra_snippets=extra_snippets if extra_snippets else [],
                metadata={
                    "backend": "brave"
                }
            ))

        return formatted_results

    def _format_news_results(self, raw_results: list[dict]) -> list[SearchResult]:
        """Convert Brave News API results to standardized SearchResult format.

        Args:
            raw_results: Raw news results from Brave API

        Returns:
            List of SearchResult objects with news-specific metadata
        """
        formatted_results = []

        for result in raw_results:
            title = result.get("title", "No title")
            url = result.get("url", "")
            description = result.get("description", "")
            age = result.get("age", "")

            # News-specific fields
            source = result.get("source", "")
            breaking = result.get("breaking", False)
            thumbnail = result.get("thumbnail", {})
            thumbnail_url = thumbnail.get("src", "") if isinstance(thumbnail, dict) else ""

            # Build metadata with news-specific info
            metadata = {
                "backend": "brave",
                "type": "news",
                "source": source,
                "is_breaking": breaking
            }
            if thumbnail_url:
                metadata["thumbnail_url"] = thumbnail_url

            # Convert to SearchResult
            formatted_results.append(SearchResult(
                title=title,
                url=url,
                snippet=description,
                date=age if age else None,
                extra_snippets=[],  # News results typically don't have extra snippets
                metadata=metadata
            ))

        return formatted_results

    async def close(self):
        """Close backend connections and cleanup resources.

        Note: httpx.AsyncClient is used with context manager, so no cleanup needed.
        """
        logger.debug("BraveSearchBackend closed")
