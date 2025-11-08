"""ABOUTME: Abstract interface for web search backends.

Defines the SearchBackend interface that all search implementations must follow,
along with standardized data classes for search results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchResult:
    """Standardized search result format across all backends.

    Attributes:
        title: Page title
        url: Page URL
        snippet: Main content snippet/description
        date: Publication or discovery date (human-readable age, e.g., "2 days ago")
        extra_snippets: Additional contextual snippets (optional)
        metadata: Backend-specific metadata (optional)
        source_name: Clean source name (e.g., "W3Schools", "React") (optional)
        source_favicon: Favicon URL for source (optional)
        content_type: Content type classification ("article", "generic", "video") (optional)
        thumbnail_url: Preview image URL (optional)
        thumbnail_is_logo: Whether thumbnail is a logo vs content image (optional)
        language: Content language code (e.g., "en") (optional)
        page_timestamp: ISO 8601 timestamp for precise date sorting (optional)
        is_live: Whether content is live/streaming (optional)
    """
    title: str
    url: str
    snippet: str
    date: Optional[str] = None
    extra_snippets: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    source_name: Optional[str] = None
    source_favicon: Optional[str] = None
    content_type: Optional[str] = None
    thumbnail_url: Optional[str] = None
    thumbnail_is_logo: Optional[bool] = None
    language: Optional[str] = None
    page_timestamp: Optional[str] = None
    is_live: Optional[bool] = None

    def get_all_text(self) -> str:
        """Get all text content (snippet + extra_snippets combined)."""
        all_snippets = [self.snippet] + self.extra_snippets
        return " ... ".join(filter(None, all_snippets))

    def estimate_tokens(self) -> int:
        """Rough token estimate for this result.

        Approximate tokens:
        - Title: ~10 tokens
        - URL: ~15 tokens
        - Snippets: ~1 token per 4 characters
        - Formatting overhead: ~10 tokens
        """
        title_tokens = len(self.title.split())
        url_tokens = 15  # URLs are usually ~15 tokens
        snippet_tokens = len(self.get_all_text()) // 4
        overhead = 10
        return title_tokens + url_tokens + snippet_tokens + overhead


@dataclass
class SearchResponse:
    """Standardized search response from backend.

    Attributes:
        query: Original search query
        results: List of search results
        total_results: Total number of results available (if known)
        metadata: Backend-specific metadata (e.g., search suggestions, related queries)
    """
    query: str
    results: list[SearchResult]
    total_results: Optional[int] = None
    metadata: dict = field(default_factory=dict)


class SearchBackend(ABC):
    """Abstract interface for web search backends.

    All search backend implementations (Brave, Google, DuckDuckGo, etc.)
    must implement this interface to ensure compatibility with the web_search tool.
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        count: int = 10,
        **kwargs
    ) -> SearchResponse:
        """Execute a web search query.

        Args:
            query: Search query string
            count: Number of results to return (1-20)
            **kwargs: Backend-specific parameters (e.g., freshness, country, language)

        Returns:
            SearchResponse with standardized results

        Raises:
            Exception: If search fails (rate limit, API error, etc.)
        """
        pass

    @abstractmethod
    async def close(self):
        """Close backend connections and cleanup resources.

        Should be called when the backend is no longer needed.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'brave', 'google', 'duckduckgo')."""
        pass
