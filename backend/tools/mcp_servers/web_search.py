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
from typing import Optional

from mcp.server.fastmcp import Context
from openai import AsyncOpenAI

from common.mcp_base import MCPServerBase
from common.search import SearchBackend, SearchResult, SearchResponse, get_search_backend
from common.search.utils import apply_quality_filters, format_as_markdown, deduplicate_by_url, condense_results

# Initialize MCP server with base class
server = MCPServerBase("web-search")
mcp = server.get_mcp()
logger = server.get_logger()

# Global backend instance (initialized on first use)
_backend: Optional[SearchBackend] = None


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


async def generate_query_variants(query: str) -> list[str]:
    """Generate 2 search query variations using LLM.

    Creates complementary search angles to get broader coverage.
    Falls back to original query if generation fails.

    Args:
        query: Original search query

    Returns:
        List of 2 query variations
    """
    try:
        llm_client = AsyncOpenAI(
            base_url=os.getenv("LLAMA_BASE_URL", "http://llama-server:8000"),
            api_key="not-needed"
        )

        prompt = f"""Generate 2 different search query variations for the topic below. Cover different angles or aspects. Return ONLY a JSON array of 2 strings, no other text.

Topic: {query}

Format: ["variant 1", "variant 2"]"""

        response = await llm_client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-oss-20b"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )

        content = response.choices[0].message.content.strip()

        # Try to extract JSON even if there's extra text
        # Look for JSON array pattern
        json_match = re.search(r'\[.*?\]', content, re.DOTALL)
        if json_match:
            variants = json.loads(json_match.group())
            if isinstance(variants, list) and len(variants) >= 2:
                logger.info(f"Generated query variants: {variants[:2]}")
                return variants[:2]

        # Fallback: use original query twice
        logger.warning("Could not parse query variants, using original query")
        return [query, query]

    except Exception as e:
        logger.warning(f"Query variant generation failed: {e}, using original query")
        return [query, query]


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


@mcp.tool()
async def web_search(
    query: str,
    count: int = 10,
    max_tokens: int = 2000,
    freshness: Optional[str] = None,
    ctx: Context = None
) -> dict:
    """Search the web with automatic query expansion and intelligent filtering.

    Automatically generates 2 query variations, executes both searches in parallel,
    combines and deduplicates results, condenses to fit token budget, and returns
    formatted markdown with citations.

    For multi-step research, the main LLM should call this tool multiple times
    with different queries based on previous results.

    Args:
        query: Search query
        count: Results per search query (default 10, so ~20 total from 2 queries)
        max_tokens: Maximum tokens for output (default 2000)
        freshness: Optional time filter - "pd" (past day), "pw" (past week),
                  "pm" (past month), "py" (past year)
        tool_call_id: Internal progress tracking ID

    Returns:
        Dict with markdown-formatted search results, titles, URLs, and snippets

    Raises:
        ValueError: If backend initialization fails
        Exception: If search fails (rate limit, API error, network error)

    Examples:
        # Basic search
        web_search("Python async programming", count=10)

        # Recent news only
        web_search("AI developments", freshness="pw")

        # Large result set
        web_search("transformer architecture", count=15, max_tokens=3000)
    """
    logger.info(f"Web search: {query}")
    if ctx:
        await ctx.info(f"Searching: {query}")

    backend = get_backend()

    try:
        # Step 1: Generate 2 query variations
        if ctx:
            await ctx.report_progress(1, 5, "Generating search query variations...")
        queries = await generate_query_variants(query)

        # Step 2: Execute both searches in parallel
        if ctx:
            await ctx.report_progress(2, 5, f"Searching: '{queries[0]}' & '{queries[1]}'")
        all_results = await execute_parallel_searches(backend, queries, count, freshness)

        # Step 3: Combine and filter
        if ctx:
            await ctx.report_progress(3, 5, f"Filtering {len(all_results)} combined results...")
        filtered = apply_quality_filters(
            all_results,
            min_snippet_length=50,
            max_per_domain=3
        )
        filtered = deduplicate_by_url(filtered)

        # Step 4: Condense to fit token budget
        if ctx:
            await ctx.report_progress(4, 5, f"Condensing {len(filtered)} results to ~{max_tokens} tokens...")
        condensed = condense_results(filtered, max_tokens)

        # Step 5: Format with citations
        if ctx:
            await ctx.report_progress(5, 5, "Formatting results with citations...")
        markdown = format_as_markdown(condensed, query, include_metadata=True)

        # Estimate final size
        final_tokens = len(markdown) // 4  # Rough estimate

        if ctx:
            await ctx.report_progress(5, 5, f"Found {len(condensed)} results (~{final_tokens} tokens)")
        logger.info(f"Web search completed: {len(condensed)} results, ~{final_tokens} tokens")

        # Return structured response with sources for annotation creation
        sources = [
            {
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet[:200] if result.snippet else ""  # Truncate for metadata
            }
            for result in condensed
        ]

        response_data = {
            "text": markdown,
            "sources": sources
        }

        return response_data

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        if ctx:
            await ctx.error(f"Search failed: {str(e)}")
        raise


@mcp.tool()
async def news_search(
    query: str,
    count: int = 10,
    max_tokens: int = 2000,
    freshness: Optional[str] = None,
    country: str = "US",
    ctx: Context = None
) -> dict:
    """Search for recent news articles with source attribution and breaking news indicators.

    Optimized for news and current events - uses focused queries without expansion.
    Returns news-specific metadata including source, publication time, and breaking status.

    Args:
        query: News search query (e.g., "AI developments", "tech industry news")
        count: Number of news articles to return (default 10, max 20)
        max_tokens: Maximum tokens for output (default 2000)
        freshness: Optional time filter - "pd" (past day), "pw" (past week),
                  "pm" (past month), "py" (past year). Defaults to None (all recent).
        country: Country code for localized news (default "US")
        ctx: MCP context for progress/logging (auto-injected)

    Returns:
        Dict with markdown-formatted news results, breaking indicators, sources, and timestamps

    Raises:
        ValueError: If backend initialization fails
        Exception: If search fails (rate limit, API error, network error)

    Examples:
        # Recent AI news (past week)
        news_search("artificial intelligence", freshness="pw")

        # Breaking tech news
        news_search("technology industry")

        # US election news from past day
        news_search("US elections", freshness="pd", country="US")
    """
    logger.info(f"News search: {query}")
    if ctx:
        await ctx.info(f"Searching news: {query}")

    backend = get_backend()

    try:
        # Validate backend supports news search
        if not hasattr(backend, 'search_news'):
            raise ValueError(f"Backend '{backend.name}' does not support news search")

        # Step 1: Execute news search (no query expansion for news)
        if ctx:
            await ctx.report_progress(1, 4, f"Searching news: '{query}'")
        response = await backend.search_news(
            query=query,
            count=count,
            freshness=freshness,
            country=country
        )

        # Step 2: Filter results
        if ctx:
            await ctx.report_progress(2, 4, f"Filtering {len(response.results)} news results...")
        filtered = apply_quality_filters(
            response.results,
            min_snippet_length=30,  # News snippets are often shorter
            max_per_domain=5  # Allow more from news sources
        )
        filtered = deduplicate_by_url(filtered)

        # Step 3: Condense to fit token budget
        if ctx:
            await ctx.report_progress(3, 4, f"Condensing {len(filtered)} results to ~{max_tokens} tokens...")
        condensed = condense_results(filtered, max_tokens)

        # Step 4: Format with news-specific styling
        if ctx:
            await ctx.report_progress(4, 4, "Formatting news results with source attribution...")
        markdown = _format_news_markdown(condensed, query)

        # Estimate final size
        final_tokens = len(markdown) // 4  # Rough estimate

        if ctx:
            await ctx.report_progress(4, 4, f"Found {len(condensed)} news articles (~{final_tokens} tokens)")
        logger.info(f"News search completed: {len(condensed)} results, ~{final_tokens} tokens")

        # Return structured response with sources for annotation creation
        sources = [
            {
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet[:200] if result.snippet else ""  # Truncate for metadata
            }
            for result in condensed
        ]

        response_data = {
            "text": markdown,
            "sources": sources
        }

        return response_data

    except Exception as e:
        logger.error(f"News search failed: {e}", exc_info=True)
        if ctx:
            await ctx.error(f"News search failed: {str(e)}")
        raise


def _format_news_markdown(results: list[SearchResult], query: str) -> str:
    """Format news results with breaking indicators and source attribution.

    Args:
        results: List of news SearchResult objects
        query: Original search query

    Returns:
        Markdown-formatted news results with presentation instructions
    """
    if not results:
        return f"No news articles found for: {query}"

    lines = [f"# News: {query}", ""]

    for i, result in enumerate(results, 1):
        # Breaking news indicator
        breaking = result.metadata.get("is_breaking", False)
        breaking_indicator = "🔴 " if breaking else ""

        # Source attribution
        source = result.metadata.get("source", "Unknown source")

        # Date/age
        age = result.date if result.date else "Recent"

        # Format result
        lines.append(f"## {i}. {breaking_indicator}{result.title}")
        lines.append(f"**Source:** {source} | **Published:** {age}")
        lines.append(f"**URL:** {result.url}")
        lines.append("")
        lines.append(result.snippet)
        lines.append("")
        lines.append("---")
        lines.append("")

    # Add UI implementation instructions for the orchestrator LLM
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**UI Implementation Guide:**")
    lines.append("")
    lines.append("Transform the above news data into a user-friendly presentation:")
    lines.append("")
    lines.append("**Breaking News Indicator:**")
    lines.append("- Each article has metadata field `is_breaking` (set by Brave News API)")
    lines.append("- If true: Display 🔴 emoji + \"Breaking:\" label before the title")
    lines.append("- If false: Display title without indicator")
    lines.append("- Note: Most articles will NOT be breaking news - API determines this")
    lines.append("")
    lines.append("**Article Rendering Structure:**")
    lines.append("- Title: Clickable link format `[Title](URL)` or bold text with URL below")
    lines.append("- Source: Display in italics or secondary color (e.g., *Source Name*)")
    lines.append("- Published: Show timestamp next to source (e.g., \"2 hours ago\")")
    lines.append("- Description: Render as paragraph text below metadata")
    lines.append("")
    lines.append("**Layout Requirements:**")
    lines.append("- Use list format (NOT tables) for better readability on all devices")
    lines.append("- Each article = one list item or card")
    lines.append("- Breaking news articles should appear first")
    lines.append("- Then sort remaining by publication time (newest first)")
    lines.append("")
    lines.append("**Example Transformations:**")
    lines.append("")
    lines.append("Breaking article (is_breaking = true):")
    lines.append("```")
    lines.append("🔴 Breaking: [Article Title](URL)")
    lines.append("*Source Name* • 2 hours ago")
    lines.append("Article description text here")
    lines.append("```")
    lines.append("")
    lines.append("Regular article (is_breaking = false):")
    lines.append("```")
    lines.append("[Article Title](URL)")
    lines.append("*Source Name* • 1 day ago")
    lines.append("Article description text here")
    lines.append("```")

    return "\n".join(lines).strip()


@mcp.tool()
async def get_search_info(ctx: Context = None) -> str:
    """Get information about web search capabilities and configuration.

    Returns:
        Information about current backend and search behavior
    """
    try:
        backend = get_backend()
        backend_name = backend.name
    except Exception:
        backend_name = "not initialized"

    info = f"""
Web Search Configuration:
- Backend: {backend_name}
- Default results per query variation: 10 (total ~20 from 2 queries)
- Automatic query expansion: 2 variations per search
- Filtering: min 50 char snippets, max 3 results/domain, URL deduplication
- Result condensing: ~2000 token default budget

Search Process:
1. Generate 2 query variations using LLM
2. Execute both searches in parallel
3. Combine and filter results (quality + deduplication)
4. Condense to fit token budget
5. Format with citations

News Search (news_search tool):
- Dedicated tool for news and current events
- No query expansion (time-sensitive, focused queries)
- Source attribution and breaking news indicators
- Time filters: pd (day), pw (week), pm (month), py (year)
- Country-specific news available
- Breaking news marked with 🔴 indicator

For Multi-Step Research:
- Call web_search multiple times with different queries
- Main LLM analyzes previous results and decides next search
- Iterations are controlled by the orchestrating LLM
- Each search call = 1 iteration

Brave Search (current backend):
- Free tier: 2,000 queries/month, 1 req/sec
- Rate limiting enforced automatically
- Up to 5 snippets per result for context
- Dedicated news endpoint for current events
"""
    return info.strip()


if __name__ == "__main__":
    logger.info("Starting Web Search MCP server (Streamable HTTP)...")

    # Verify backend can be initialized
    try:
        backend = get_backend()
        logger.info(f"Backend initialized successfully: {backend.name}")
    except Exception as e:
        logger.error(f"Failed to initialize backend: {e}")
        logger.error("Server will start but searches will fail until backend is configured")

    mcp.run(transport="streamable-http")
