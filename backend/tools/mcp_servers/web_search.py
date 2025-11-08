"""ABOUTME: Web Search MCP Server - Intelligent web search with pluggable backends.

Provides intelligent web search with pluggable backends and dual modes:
- Quick mode: Fast search with garbage filtering
- Deep mode: Multi-step research with query planning and result condensing

Currently implements Brave Search backend, with architecture supporting
future backends (Google, DuckDuckGo, etc.).
"""

import asyncio
import json
import logging
import os
import re
from typing import Optional, Any

from mcp.server.fastmcp import Context
from mcp.types import TextContent, CallToolResult
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from common.mcp_base import MCPServerBase
from common.error_handling import (
    ERROR_INVALID_INPUT,
    ERROR_RATE_LIMITED,
    create_error_result,
    create_validation_error,
    create_rate_limit_error
)
from common.search import SearchBackend, SearchResult, SearchResponse, get_search_backend
from common.search.utils import apply_quality_filters, format_as_markdown, deduplicate_by_url, condense_results

# Initialize MCP server with base class
server = MCPServerBase("web-search")
mcp = server.get_mcp()
logger = server.get_logger()

# ============================================================================
# MODULE-LEVEL CONSTANTS
# ============================================================================

# Query validation constraints
MIN_QUERY_LENGTH: int = 3
MAX_QUERY_LENGTH: int = 256

# Result count constraints
MIN_RESULT_COUNT: int = 1
MAX_RESULT_COUNT: int = 50
DEFAULT_RESULT_COUNT: int = 10

# Token budget constraints
DEFAULT_MAX_TOKENS: int = 2000
MIN_MAX_TOKENS: int = 500
MAX_MAX_TOKENS: int = 10000

# Quality filter parameters
MIN_SNIPPET_LENGTH_WEB: int = 50
MAX_PER_DOMAIN_WEB: int = 3
MIN_SNIPPET_LENGTH_NEWS: int = 30
MAX_PER_DOMAIN_NEWS: int = 5

# Query expansion parameters
QUERY_VARIANTS_COUNT: int = 2
QUERY_VARIANTS_MAX_TOKENS: int = 150
QUERY_VARIANTS_TEMPERATURE: float = 0.7

# Metadata truncation
MAX_SNIPPET_METADATA_LENGTH: int = 200

# Valid freshness values
VALID_FRESHNESS_VALUES: set[str] = {"pd", "pw", "pm", "py"}

# Web search-specific error codes
ERROR_SEARCH_FAILED: str = "search_failed"
ERROR_BACKEND_INIT: str = "backend_init_failed"
ERROR_VARIANT_GENERATION: str = "variant_generation_failed"
ERROR_NEWS_UNSUPPORTED: str = "news_unsupported"

# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class WebSearchInput(BaseModel):
    """Input schema for web_search tool."""
    query: str = Field(
        ...,
        description="Search query string",
        min_length=MIN_QUERY_LENGTH,
        max_length=MAX_QUERY_LENGTH
    )
    count: int = Field(
        default=DEFAULT_RESULT_COUNT,
        description="Results per search query variation",
        ge=MIN_RESULT_COUNT,
        le=MAX_RESULT_COUNT
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        description="Maximum tokens for output",
        ge=MIN_MAX_TOKENS,
        le=MAX_MAX_TOKENS
    )
    freshness: Optional[str] = Field(
        default=None,
        description="Time filter: 'pd' (past day), 'pw' (past week), 'pm' (past month), 'py' (past year)"
    )


class NewsSearchInput(BaseModel):
    """Input schema for news_search tool."""
    query: str = Field(
        ...,
        description="News search query string",
        min_length=MIN_QUERY_LENGTH,
        max_length=MAX_QUERY_LENGTH
    )
    count: int = Field(
        default=DEFAULT_RESULT_COUNT,
        description="Number of news articles to return",
        ge=MIN_RESULT_COUNT,
        le=MAX_RESULT_COUNT
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        description="Maximum tokens for output",
        ge=MIN_MAX_TOKENS,
        le=MAX_MAX_TOKENS
    )
    freshness: Optional[str] = Field(
        default=None,
        description="Time filter: 'pd' (past day), 'pw' (past week), 'pm' (past month), 'py' (past year)"
    )
    country: str = Field(
        default="US",
        description="Country code for localized news",
        min_length=2,
        max_length=2
    )


class SearchSourceMetadata(BaseModel):
    """Metadata for a single search result source."""
    title: str
    url: str
    snippet: str


class WebSearchOutput(BaseModel):
    """Output schema for web_search tool."""
    markdown: str = Field(description="Formatted markdown with citations")
    sources: list[SearchSourceMetadata] = Field(description="List of source metadata")
    metadata: dict[str, Any] = Field(description="Structured search metadata")


class NewsSearchOutput(BaseModel):
    """Output schema for news_search tool."""
    markdown: str = Field(description="Formatted news markdown with breaking indicators")
    sources: list[SearchSourceMetadata] = Field(description="List of news source metadata")
    metadata: dict[str, Any] = Field(description="Structured news search metadata")


class SearchInfoOutput(BaseModel):
    """Output schema for get_search_info tool."""
    capabilities: str = Field(description="Search capabilities and configuration text")
    metadata: dict[str, Any] = Field(description="Structured capabilities metadata")


# ============================================================================
# GLOBAL STATE
# ============================================================================

# Global backend instance (initialized on first use)
_backend: Optional[SearchBackend] = None


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_query(query: str) -> tuple[bool, Optional[str]]:
    """Validate search query.

    Args:
        query: Search query string

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Query cannot be empty"

    if len(query) < MIN_QUERY_LENGTH:
        return False, f"Query must be at least {MIN_QUERY_LENGTH} characters"

    if len(query) > MAX_QUERY_LENGTH:
        return False, f"Query must not exceed {MAX_QUERY_LENGTH} characters"

    return True, None


def validate_freshness(freshness: Optional[str]) -> tuple[bool, Optional[str]]:
    """Validate freshness parameter.

    Args:
        freshness: Freshness filter value

    Returns:
        Tuple of (is_valid, error_message)
    """
    if freshness is None:
        return True, None

    if freshness not in VALID_FRESHNESS_VALUES:
        valid_values = ", ".join(sorted(VALID_FRESHNESS_VALUES))
        return False, f"Invalid freshness value '{freshness}'. Must be one of: {valid_values}"

    return True, None


def validate_count(count: int) -> tuple[bool, Optional[str]]:
    """Validate result count parameter.

    Args:
        count: Number of results

    Returns:
        Tuple of (is_valid, error_message)
    """
    if count < MIN_RESULT_COUNT:
        return False, f"Count must be at least {MIN_RESULT_COUNT}"

    if count > MAX_RESULT_COUNT:
        return False, f"Count must not exceed {MAX_RESULT_COUNT}"

    return True, None


def validate_max_tokens(max_tokens: int) -> tuple[bool, Optional[str]]:
    """Validate max_tokens parameter.

    Args:
        max_tokens: Maximum token budget

    Returns:
        Tuple of (is_valid, error_message)
    """
    if max_tokens < MIN_MAX_TOKENS:
        return False, f"max_tokens must be at least {MIN_MAX_TOKENS}"

    if max_tokens > MAX_MAX_TOKENS:
        return False, f"max_tokens must not exceed {MAX_MAX_TOKENS}"

    return True, None


# ============================================================================
# BACKEND MANAGEMENT
# ============================================================================

def get_backend() -> SearchBackend:
    """Get or initialize the search backend.

    Returns:
        Initialized search backend

    Raises:
        ValueError: If backend cannot be initialized
    """
    global _backend

    if _backend is None:
        # Use factory for pluggable backend selection
        _backend = get_search_backend()

    return _backend


# ============================================================================
# QUERY EXPANSION AND SEARCH EXECUTION
# ============================================================================

async def generate_query_variants(query: str) -> list[str]:
    """Generate search query variations using LLM.

    Creates complementary search angles to get broader coverage.
    Falls back to original query if generation fails.

    Args:
        query: Original search query

    Returns:
        List of query variations (length=QUERY_VARIANTS_COUNT)
    """
    try:
        llm_client = AsyncOpenAI(
            base_url=os.getenv("LLAMA_BASE_URL", "http://llama-server:8000"),
            api_key="not-needed"
        )

        prompt = f"""Generate {QUERY_VARIANTS_COUNT} different search query variations for the topic below. Cover different angles or aspects. Return ONLY a JSON array of {QUERY_VARIANTS_COUNT} strings, no other text.

Topic: {query}

Format: ["variant 1", "variant 2"]"""

        response = await llm_client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-oss-20b"),
            messages=[{"role": "user", "content": prompt}],
            temperature=QUERY_VARIANTS_TEMPERATURE,
            max_tokens=QUERY_VARIANTS_MAX_TOKENS
        )

        content = response.choices[0].message.content.strip()

        # Try to extract JSON even if there's extra text
        # Look for JSON array pattern
        json_match = re.search(r'\[.*?\]', content, re.DOTALL)
        if json_match:
            variants = json.loads(json_match.group())
            if isinstance(variants, list) and len(variants) >= QUERY_VARIANTS_COUNT:
                logger.info(f"Generated query variants: {variants[:QUERY_VARIANTS_COUNT]}")
                return variants[:QUERY_VARIANTS_COUNT]

        # Fallback: use original query multiple times
        logger.warning("Could not parse query variants, using original query")
        return [query] * QUERY_VARIANTS_COUNT

    except Exception as e:
        logger.warning(f"Query variant generation failed: {e}, using original query")
        return [query] * QUERY_VARIANTS_COUNT


async def execute_parallel_searches(
    backend: SearchBackend,
    queries: list[str],
    count: int,
    freshness: Optional[str]
) -> list[SearchResult]:
    """Execute multiple searches in parallel and combine results.

    Args:
        backend: Search backend to use
        queries: List of search queries
        count: Results per query
        freshness: Optional time filter

    Returns:
        Combined list of search results from all queries
    """
    logger.debug(f"Executing {len(queries)} parallel searches with count={count}, freshness={freshness}")

    # Create search tasks
    tasks = [
        backend.search(q, count=count, freshness=freshness)
        for q in queries
    ]

    # Execute in parallel
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Combine all results
    all_results = []
    for i, response in enumerate(responses):
        if isinstance(response, SearchResponse):
            logger.debug(f"Search {i+1} returned {len(response.results)} results")
            all_results.extend(response.results)
        elif isinstance(response, Exception):
            logger.warning(f"Search {i+1} failed: {response}")

    logger.info(f"Parallel searches: {len(queries)} queries → {len(all_results)} total results")
    return all_results


# ============================================================================
# MCP TOOL IMPLEMENTATIONS
# ============================================================================

@mcp.tool()
async def web_search(
    query: str,
    count: int = DEFAULT_RESULT_COUNT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    freshness: Optional[str] = None,
    ctx: Context = None
) -> CallToolResult:
    """Search the web with automatic query expansion and intelligent filtering.

    Automatically generates 3 query variations, executes searches in parallel,
    combines and deduplicates results, condenses to fit token budget, and returns
    formatted markdown with citations.

    **Process**:
    1. Generate 3 query variations using LLM
    2. Execute searches in parallel
    3. Combine and filter (min 50 char snippets, max 2 results/domain)
    4. Condense results to fit token budget
    5. Format with citations

    For multi-step research, call this tool multiple times with different queries
    based on previous results. Each call is one research iteration controlled by the LLM.

    **Freshness Options**: pd (day), pw (week), pm (month), py (year)
    **Rate Limit**: 2000 queries/month, 1 request/second

    Args:
        query: Search query (3-256 characters)
        count: Results per search query variation (1-50, default 10)
        max_tokens: Maximum tokens for output (500-10000, default 2000)
        freshness: Time filter - "pd" (day), "pw" (week), "pm" (month), "py" (year)

    Returns:
        CallToolResult with markdown content, sources metadata, and search metadata
    """
    logger.info(f"Web search: query='{query}', count={count}, max_tokens={max_tokens}, freshness={freshness}")

    # ========================================================================
    # INPUT VALIDATION
    # ========================================================================

    # Validate query
    is_valid, error_msg = validate_query(query)
    if not is_valid:
        logger.warning(f"Invalid query: {error_msg}")
        if ctx:
            await ctx.error(f"Invalid query: {error_msg}")
        return create_validation_error(
            field_name="query",
            error_message=error_msg,
            field_value=query
        )

    # Validate count
    is_valid, error_msg = validate_count(count)
    if not is_valid:
        logger.warning(f"Invalid count: {error_msg}")
        if ctx:
            await ctx.error(f"Invalid count: {error_msg}")
        return create_validation_error(
            field_name="count",
            error_message=error_msg,
            field_value=count
        )

    # Validate max_tokens
    is_valid, error_msg = validate_max_tokens(max_tokens)
    if not is_valid:
        logger.warning(f"Invalid max_tokens: {error_msg}")
        if ctx:
            await ctx.error(f"Invalid max_tokens: {error_msg}")
        return create_validation_error(
            field_name="max_tokens",
            error_message=error_msg,
            field_value=max_tokens
        )

    # Validate freshness
    is_valid, error_msg = validate_freshness(freshness)
    if not is_valid:
        logger.warning(f"Invalid freshness: {error_msg}")
        if ctx:
            await ctx.error(f"Invalid freshness: {error_msg}")
        return create_validation_error(
            field_name="freshness",
            error_message=error_msg,
            field_value=freshness
        )

    # ========================================================================
    # BACKEND INITIALIZATION
    # ========================================================================

    try:
        backend = get_backend()
    except Exception as e:
        error_msg = f"Failed to initialize search backend: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if ctx:
            await ctx.error(error_msg)
        return create_error_result(
            error_message=error_msg,
            error_code=ERROR_BACKEND_INIT,
            error_type="backend_error"
        )

    # ========================================================================
    # SEARCH EXECUTION
    # ========================================================================

    if ctx:
        await ctx.info(f"Searching: {query}")

    try:
        # Step 1: Generate query variations
        if ctx:
            await ctx.report_progress(1, 5, "Generating search query variations...")
        queries = await generate_query_variants(query)
        logger.debug(f"Query variants: {queries}")

        # Step 2: Execute searches in parallel
        if ctx:
            await ctx.report_progress(2, 5, f"Searching: '{queries[0]}' & '{queries[1]}'")
        all_results = await execute_parallel_searches(backend, queries, count, freshness)

        if not all_results:
            logger.info("No search results returned")
            return CallToolResult(
                content=[TextContent(type="text", text=f"No results found for: {query}")],
                metadata={
                    "search_backend": backend.name,
                    "query_used": query,
                    "query_variants": queries,
                    "results_count": 0,
                    "freshness_filter_used": freshness
                }
            )

        # Step 3: Combine and filter
        if ctx:
            await ctx.report_progress(3, 5, f"Filtering {len(all_results)} combined results...")
        filtered = apply_quality_filters(
            all_results,
            min_snippet_length=MIN_SNIPPET_LENGTH_WEB,
            max_per_domain=MAX_PER_DOMAIN_WEB
        )
        filtered = deduplicate_by_url(filtered)
        logger.debug(f"After filtering: {len(filtered)} results")

        # Step 4: Condense to fit token budget
        if ctx:
            await ctx.report_progress(4, 5, f"Condensing {len(filtered)} results to ~{max_tokens} tokens...")
        condensed = condense_results(filtered, max_tokens)
        logger.debug(f"After condensing: {len(condensed)} results")

        # Step 5: Format with citations
        if ctx:
            await ctx.report_progress(5, 5, "Formatting results with citations...")
        markdown = format_as_markdown(condensed, query, include_metadata=True)

        # Estimate final size
        final_tokens = len(markdown) // 4  # Rough estimate

        if ctx:
            await ctx.report_progress(5, 5, f"Found {len(condensed)} results (~{final_tokens} tokens)")
        logger.info(f"Web search completed: {len(condensed)} results, ~{final_tokens} tokens")

        # Build sources metadata
        sources = [
            {
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet[:MAX_SNIPPET_METADATA_LENGTH] if result.snippet else ""
            }
            for result in condensed
        ]

        # Build structured metadata
        metadata = {
            "search_backend": backend.name,
            "query_used": query,
            "query_variants": queries,
            "results_count": len(condensed),
            "results_raw_count": len(all_results),
            "results_filtered_count": len(filtered),
            "freshness_filter_used": freshness,
            "estimated_tokens": final_tokens,
            "max_tokens_requested": max_tokens
        }

        return CallToolResult(
            content=[TextContent(type="text", text=markdown)],
            metadata=metadata
        )

    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if ctx:
            await ctx.error(error_msg)

        # Check for rate limiting
        if "rate limit" in str(e).lower() or "429" in str(e):
            return create_rate_limit_error(limit_description="Web search rate limit")

        return create_error_result(
            error_message=error_msg,
            error_code=ERROR_SEARCH_FAILED,
            error_type="search_error",
            additional_metadata={"query": query}
        )


@mcp.tool()
async def news_search(
    query: str,
    count: int = DEFAULT_RESULT_COUNT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    freshness: Optional[str] = None,
    country: str = "US",
    ctx: Context = None
) -> CallToolResult:
    """Search for recent news articles with source attribution and breaking news indicators.

    Optimized for news and current events - uses focused queries without expansion (no query variations).
    Returns news-specific metadata including source, publication time, and breaking news status.

    **Features**:
    • Time filters: pd (day), pw (week), pm (month), py (year)
    • Country-specific news available
    • Breaking news indicator in metadata (is_breaking field)
    • Dedicated news endpoint for current events
    • Min 50 char snippets, max 2 results/domain

    Args:
        query: News search query (3-256 characters)
        count: Number of news articles to return (1-50, default 10)
        max_tokens: Maximum tokens for output (500-10000, default 2000)
        freshness: Time filter - "pd" (day), "pw" (week), "pm" (month), "py" (year)
        country: Country code for localized news (default "US")

    Returns:
        CallToolResult with markdown content, sources metadata, and news metadata
    """
    logger.info(f"News search: query='{query}', count={count}, max_tokens={max_tokens}, freshness={freshness}, country={country}")

    # ========================================================================
    # INPUT VALIDATION
    # ========================================================================

    # Validate query
    is_valid, error_msg = validate_query(query)
    if not is_valid:
        logger.warning(f"Invalid query: {error_msg}")
        if ctx:
            await ctx.error(f"Invalid query: {error_msg}")
        return create_validation_error(
            field_name="query",
            error_message=error_msg,
            field_value=query
        )

    # Validate count
    is_valid, error_msg = validate_count(count)
    if not is_valid:
        logger.warning(f"Invalid count: {error_msg}")
        if ctx:
            await ctx.error(f"Invalid count: {error_msg}")
        return create_validation_error(
            field_name="count",
            error_message=error_msg,
            field_value=count
        )

    # Validate max_tokens
    is_valid, error_msg = validate_max_tokens(max_tokens)
    if not is_valid:
        logger.warning(f"Invalid max_tokens: {error_msg}")
        if ctx:
            await ctx.error(f"Invalid max_tokens: {error_msg}")
        return create_validation_error(
            field_name="max_tokens",
            error_message=error_msg,
            field_value=max_tokens
        )

    # Validate freshness
    is_valid, error_msg = validate_freshness(freshness)
    if not is_valid:
        logger.warning(f"Invalid freshness: {error_msg}")
        if ctx:
            await ctx.error(f"Invalid freshness: {error_msg}")
        return create_validation_error(
            field_name="freshness",
            error_message=error_msg,
            field_value=freshness
        )

    # ========================================================================
    # BACKEND INITIALIZATION
    # ========================================================================

    try:
        backend = get_backend()
    except Exception as e:
        error_msg = f"Failed to initialize search backend: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if ctx:
            await ctx.error(error_msg)
        return create_error_result(
            error_message=error_msg,
            error_code=ERROR_BACKEND_INIT,
            error_type="backend_error"
        )

    # Validate backend supports news search
    if not hasattr(backend, 'search_news'):
        error_msg = f"Backend '{backend.name}' does not support news search"
        logger.error(error_msg)
        if ctx:
            await ctx.error(error_msg)
        return create_error_result(
            error_message=error_msg,
            error_code=ERROR_NEWS_UNSUPPORTED,
            error_type="feature_error",
            additional_metadata={"backend": backend.name}
        )

    # ========================================================================
    # NEWS SEARCH EXECUTION
    # ========================================================================

    if ctx:
        await ctx.info(f"Searching news: {query}")

    try:
        # Step 1: Execute news search (no query expansion for news)
        if ctx:
            await ctx.report_progress(1, 4, f"Searching news: '{query}'")
        response = await backend.search_news(
            query=query,
            count=count,
            freshness=freshness,
            country=country
        )

        if not response.results:
            logger.info("No news results returned")
            return CallToolResult(
                content=[TextContent(type="text", text=f"No news articles found for: {query}")],
                metadata={
                    "search_backend": backend.name,
                    "query_used": query,
                    "results_count": 0,
                    "freshness_filter_used": freshness,
                    "country": country,
                    "breaking_count": 0
                }
            )

        # Step 2: Filter results
        if ctx:
            await ctx.report_progress(2, 4, f"Filtering {len(response.results)} news results...")
        filtered = apply_quality_filters(
            response.results,
            min_snippet_length=MIN_SNIPPET_LENGTH_NEWS,
            max_per_domain=MAX_PER_DOMAIN_NEWS
        )
        filtered = deduplicate_by_url(filtered)
        logger.debug(f"After filtering: {len(filtered)} news results")

        # Step 3: Condense to fit token budget
        if ctx:
            await ctx.report_progress(3, 4, f"Condensing {len(filtered)} results to ~{max_tokens} tokens...")
        condensed = condense_results(filtered, max_tokens)
        logger.debug(f"After condensing: {len(condensed)} news results")

        # Step 4: Format with news-specific styling
        if ctx:
            await ctx.report_progress(4, 4, "Formatting news results with source attribution...")
        markdown, breaking_count = _format_news_markdown(condensed, query)

        # Estimate final size
        final_tokens = len(markdown) // 4  # Rough estimate

        if ctx:
            await ctx.report_progress(4, 4, f"Found {len(condensed)} news articles (~{final_tokens} tokens)")
        logger.info(f"News search completed: {len(condensed)} results, {breaking_count} breaking, ~{final_tokens} tokens")

        # Build sources metadata with breaking status
        sources = [
            {
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet[:MAX_SNIPPET_METADATA_LENGTH] if result.snippet else "",
                "is_breaking": result.metadata.get("is_breaking", False),
                "source": result.metadata.get("source", "Unknown source"),
                "published": result.date if result.date else "Recent"
            }
            for result in condensed
        ]

        # Build structured metadata
        metadata = {
            "search_backend": backend.name,
            "query_used": query,
            "results_count": len(condensed),
            "results_raw_count": len(response.results),
            "results_filtered_count": len(filtered),
            "freshness_filter_used": freshness,
            "country": country,
            "breaking_count": breaking_count,
            "estimated_tokens": final_tokens,
            "max_tokens_requested": max_tokens
        }

        return CallToolResult(
            content=[TextContent(type="text", text=markdown)],
            metadata=metadata
        )

    except Exception as e:
        error_msg = f"News search failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if ctx:
            await ctx.error(error_msg)

        # Check for rate limiting
        if "rate limit" in str(e).lower() or "429" in str(e):
            return create_rate_limit_error(limit_description="News search rate limit")

        return create_error_result(
            error_message=error_msg,
            error_code=ERROR_SEARCH_FAILED,
            error_type="search_error",
            additional_metadata={"query": query, "country": country}
        )


# ============================================================================
# NEWS FORMATTING
# ============================================================================

def _format_news_markdown(results: list[SearchResult], query: str) -> tuple[str, int]:
    """Format news results with source attribution.

    Breaking news indicator is moved to metadata for UI consumption.
    Does NOT include emoji in markdown text - UI should render based on metadata.

    Args:
        results: List of news SearchResult objects
        query: Original search query

    Returns:
        Tuple of (markdown_text, breaking_count)
    """
    if not results:
        return f"No news articles found for: {query}", 0

    lines = [f"# News: {query}", ""]
    breaking_count = 0

    for i, result in enumerate(results, 1):
        # Count breaking news (metadata only, not displayed in markdown)
        breaking = result.metadata.get("is_breaking", False)
        if breaking:
            breaking_count += 1

        # Source attribution
        source = result.metadata.get("source", "Unknown source")

        # Date/age
        age = result.date if result.date else "Recent"

        # Format result (NO emoji - UI renders based on metadata)
        lines.append(f"## {i}. {result.title}")
        lines.append(f"**Source:** {source} | **Published:** {age}")
        lines.append(f"**URL:** {result.url}")
        lines.append("")
        lines.append(result.snippet)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).strip(), breaking_count


if __name__ == "__main__":
    logger.info("Starting Web Search MCP server (Streamable HTTP)...")

    # Verify backend can be initialized
    try:
        backend = get_backend()
        logger.info(f"Backend initialized successfully: {backend.name}")
    except Exception as e:
        logger.error(f"Failed to initialize backend: {e}")
        logger.error("Server will start but searches will fail until backend is configured")

    server.run(transport="streamable-http")
