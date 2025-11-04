"""ABOUTME: Shared utilities for web search - filtering, formatting, and result orchestration.

Provides filtering functions used by both quick and deep modes,
formatting utilities, and result condensing for token budgets.
"""

import json
import logging
import os
from typing import Optional
from urllib.parse import urlparse

from openai import AsyncOpenAI

from .backend import SearchBackend, SearchResult, SearchResponse

# Configure logging
logger = logging.getLogger(__name__)


# ===========================
# Filtering Functions
# ===========================

def filter_low_quality(
    results: list[SearchResult],
    min_snippet_length: int = 50
) -> list[SearchResult]:
    """Remove low-quality search results.

    Filters out results with:
    - Snippet shorter than min_snippet_length
    - Missing title or URL
    - Empty or whitespace-only content

    Args:
        results: List of search results
        min_snippet_length: Minimum snippet length in characters (default 50)

    Returns:
        Filtered list of search results
    """
    filtered = []

    for result in results:
        # Must have title and URL
        if not result.title or not result.url:
            logger.debug(f"Filtered: missing title or URL - {result.url}")
            continue

        # Must have minimum snippet length
        if len(result.snippet) < min_snippet_length:
            logger.debug(f"Filtered: snippet too short ({len(result.snippet)} chars) - {result.title}")
            continue

        # Must have non-whitespace content
        if not result.snippet.strip():
            logger.debug(f"Filtered: empty snippet - {result.title}")
            continue

        filtered.append(result)

    logger.debug(f"Quality filtering: {len(results)} → {len(filtered)} results")
    return filtered


def deduplicate_domains(
    results: list[SearchResult],
    max_per_domain: int = 3
) -> list[SearchResult]:
    """Limit number of results per domain to reduce redundancy.

    Args:
        results: List of search results
        max_per_domain: Maximum results allowed per domain (default 3)

    Returns:
        Deduplicated list of search results
    """
    seen_domains = {}
    deduplicated = []

    for result in results:
        domain = extract_domain(result.url)

        # Count results from this domain
        count = seen_domains.get(domain, 0)

        if count >= max_per_domain:
            logger.debug(f"Filtered: domain limit reached ({domain}) - {result.title}")
            continue

        seen_domains[domain] = count + 1
        deduplicated.append(result)

    logger.debug(f"Domain deduplication: {len(results)} → {len(deduplicated)} results")
    return deduplicated


def extract_domain(url: str) -> str:
    """Extract domain from URL.

    Args:
        url: Full URL

    Returns:
        Domain name (e.g., "example.com")
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc

        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]

        return domain
    except Exception:
        return url


def apply_quality_filters(
    results: list[SearchResult],
    min_snippet_length: int = 50,
    max_per_domain: int = 3
) -> list[SearchResult]:
    """Apply all quality filters to search results.

    Combines filtering and deduplication in one pass.

    Args:
        results: List of search results
        min_snippet_length: Minimum snippet length in characters
        max_per_domain: Maximum results per domain

    Returns:
        Filtered and deduplicated list of search results
    """
    # First filter low quality
    filtered = filter_low_quality(results, min_snippet_length)

    # Then deduplicate by domain
    deduplicated = deduplicate_domains(filtered, max_per_domain)

    logger.info(f"Quality filtering: {len(results)} → {len(deduplicated)} results")
    return deduplicated


# ===========================
# Formatting Functions
# ===========================

def format_as_markdown(
    results: list[SearchResult],
    query: str,
    include_metadata: bool = True
) -> str:
    """Format search results as markdown.

    Args:
        results: List of search results
        query: Original search query
        include_metadata: Include result count and metadata (default True)

    Returns:
        Markdown-formatted string
    """
    if not results:
        return f"No results found for query: {query}"

    output = []

    if include_metadata:
        output.append(f"# Search Results for: \"{query}\"\n")
        output.append(f"Found {len(results)} results\n")
        output.append("---\n")

    for idx, result in enumerate(results, 1):
        # Title and URL
        output.append(f"## {idx}. {result.title}\n")
        output.append(f"**URL**: {result.url}\n")

        # Date if available
        if result.date:
            output.append(f"**Date**: {result.date}\n")

        # Snippet (combine main + extra snippets)
        full_text = result.get_all_text()
        output.append(f"{full_text}\n")

        output.append("---\n")

    return "\n".join(output)


def estimate_markdown_tokens(
    results: list[SearchResult],
    query: str
) -> int:
    """Estimate token count for markdown-formatted results.

    Args:
        results: List of search results
        query: Original search query

    Returns:
        Estimated token count
    """
    # Header tokens
    header_tokens = len(query.split()) + 20  # "Search Results for: {query}\nFound X results"

    # Result tokens
    result_tokens = sum(r.estimate_tokens() for r in results)

    # Formatting overhead (markdown headers, separators, etc.)
    formatting_overhead = len(results) * 15

    total = header_tokens + result_tokens + formatting_overhead
    return total


def deduplicate_by_url(results: list[SearchResult]) -> list[SearchResult]:
    """Remove results with duplicate URLs, keeping first occurrence.

    Args:
        results: List of search results

    Returns:
        Deduplicated list of search results
    """
    seen_urls = set()
    unique = []

    for result in results:
        if result.url not in seen_urls:
            seen_urls.add(result.url)
            unique.append(result)

    logger.debug(f"URL deduplication: {len(results)} → {len(unique)} results")
    return unique


def condense_results(
    results: list[SearchResult],
    max_tokens: int = 2000
) -> list[SearchResult]:
    """Condense results to fit within token budget.

    Strategy:
    1. Estimate current token count
    2. If over budget, trim snippets to fit
    3. Prioritize keeping more results over longer snippets

    Args:
        results: Search results to condense
        max_tokens: Maximum token budget

    Returns:
        Condensed results that fit within budget
    """
    if not results:
        return results

    # Estimate current size (rough: 4 chars = 1 token)
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    def result_tokens(result: SearchResult) -> int:
        # Title + URL + snippet + formatting overhead
        return estimate_tokens(result.title + result.url + result.get_all_text()) + 20

    current_tokens = sum(result_tokens(r) for r in results)

    if current_tokens <= max_tokens:
        logger.debug(f"Results fit in budget: {current_tokens} <= {max_tokens} tokens")
        return results

    logger.debug(f"Condensing results: {current_tokens} → ~{max_tokens} tokens")

    # Strategy: Keep all results but trim snippets proportionally
    ratio = max_tokens / current_tokens
    condensed = []

    for result in results:
        # Calculate target snippet length
        current_snippet = result.get_all_text()
        target_length = int(len(current_snippet) * ratio * 0.9)  # 0.9 for safety margin

        if target_length < 100:
            target_length = 100  # Minimum useful snippet

        # Trim snippet
        if len(current_snippet) > target_length:
            trimmed_snippet = current_snippet[:target_length] + "..."
            # Create new result with trimmed snippet
            condensed_result = SearchResult(
                title=result.title,
                url=result.url,
                snippet=trimmed_snippet,
                date=result.date,
                extra_snippets=[]  # Drop extra snippets to save space
            )
            condensed.append(condensed_result)
        else:
            condensed.append(result)

    final_tokens = sum(result_tokens(r) for r in condensed)
    logger.debug(f"Condensed to {final_tokens} tokens ({len(condensed)} results)")

    return condensed
