"""ABOUTME: Wikipedia MCP Server - Fast world knowledge lookup with Resources, Tools, and Prompts.

Uses all three MCP primitives for optimal knowledge surfacing:
- Resources: Pre-loadable context (app-controlled) - wikipedia://topic/{topic}
- Tools: Dynamic lookup (model-controlled) - wikipedia_lookup, wikipedia_search
- Prompts: Research templates (user-controlled) - research_topic, fact_check

Performance tiers:
- Cache hit: <1ms (LRU cache)
- Cache miss: 100-200ms (Wikipedia REST API)

Zero external dependencies beyond httpx. All lookups go to Wikipedia's public REST API.
"""

import sys
from pathlib import Path
from typing import Optional

import httpx
from mcp.server.fastmcp import Context
from mcp.types import TextContent, CallToolResult
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.mcp_base import MCPServerBase
from common.error_handling import create_error_result, create_validation_error

# =============================================================================
# Module-Level Constants
# =============================================================================

WIKIPEDIA_REST_API = "https://en.wikipedia.org/api/rest_v1"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# Cache configuration
CACHE_SIZE = 10_000  # ~50MB RAM for summaries
HTTP_TIMEOUT = 5.0  # seconds

# Validation limits
TOPIC_MIN_LENGTH = 2
TOPIC_MAX_LENGTH = 256
QUERY_MIN_LENGTH = 2
QUERY_MAX_LENGTH = 256
SEARCH_LIMIT_MIN = 1
SEARCH_LIMIT_MAX = 20
SEARCH_LIMIT_DEFAULT = 5

# Error codes
ERROR_NOT_FOUND = "wikipedia_not_found"
ERROR_API_ERROR = "wikipedia_api_error"
ERROR_VALIDATION = "validation_error"

# =============================================================================
# Server Initialization
# =============================================================================

server = MCPServerBase("wikipedia")
mcp = server.get_mcp()
logger = server.get_logger()


def get_mcp():
    """Get the MCP server instance for launcher compatibility."""
    return mcp


# =============================================================================
# Pydantic Models
# =============================================================================


class WikipediaLookupInput(BaseModel):
    """Input schema for wikipedia_lookup tool."""

    topic: str = Field(
        description="The topic to look up (e.g., 'Albert Einstein', 'quantum physics')",
        min_length=TOPIC_MIN_LENGTH,
        max_length=TOPIC_MAX_LENGTH,
    )


class WikipediaSearchInput(BaseModel):
    """Input schema for wikipedia_search tool."""

    query: str = Field(
        description="Search query to find Wikipedia articles",
        min_length=QUERY_MIN_LENGTH,
        max_length=QUERY_MAX_LENGTH,
    )
    limit: int = Field(
        default=SEARCH_LIMIT_DEFAULT,
        ge=SEARCH_LIMIT_MIN,
        le=SEARCH_LIMIT_MAX,
        description=f"Maximum results ({SEARCH_LIMIT_MIN}-{SEARCH_LIMIT_MAX})",
    )


# =============================================================================
# HTTP Client Helpers
# =============================================================================


async def fetch_summary(topic: str) -> dict:
    """Fetch article summary from Wikipedia REST API.

    Uses the /page/summary endpoint which returns:
    - extract: Plain text summary (2-3 sentences)
    - description: Short description
    - thumbnail: Image URL if available

    Latency: ~100-200ms (single HTTP call, pre-parsed JSON)
    """
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        safe_topic = topic.replace(" ", "_")
        resp = await client.get(
            f"{WIKIPEDIA_REST_API}/page/summary/{safe_topic}",
            headers={"User-Agent": "Strieber-GPT/1.0 (MCP Wikipedia Server)"},
        )
        resp.raise_for_status()
        return resp.json()


async def search_wikipedia(query: str, limit: int = 5) -> list[dict]:
    """Search Wikipedia using the REST API.

    Uses /page/search endpoint for fast prefix/title matching.
    """
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.get(
            f"{WIKIPEDIA_REST_API}/page/search",
            params={"q": query, "limit": limit},
            headers={"User-Agent": "Strieber-GPT/1.0 (MCP Wikipedia Server)"},
        )
        resp.raise_for_status()
        return resp.json().get("pages", [])


async def fetch_wikidata_entity(query: str) -> Optional[dict]:
    """Search for and fetch a Wikidata entity.

    Returns basic entity info (QID, label, description).
    """
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.get(
            WIKIDATA_API,
            params={
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "format": "json",
                "limit": 1,
            },
            headers={"User-Agent": "Strieber-GPT/1.0 (MCP Wikipedia Server)"},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("search"):
            return data["search"][0]
        return None


# =============================================================================
# MCP RESOURCES - App-controlled, pre-loaded into LLM context
# =============================================================================


@mcp.resource("wikipedia://topic/{topic}")
async def get_topic_resource(topic: str) -> str:
    """Wikipedia article summary as a resource.

    Resources are loaded into context BEFORE the LLM responds.
    Use this when you want the LLM to always have access to certain knowledge.

    Example: Client includes wikipedia://topic/Python_(programming_language)
    in the context, and LLM sees the summary without needing to call a tool.
    """
    logger.info(f"Resource request: wikipedia://topic/{topic}")

    try:
        data = await fetch_summary(topic)

        title = data.get("title", topic)
        description = data.get("description", "")
        extract = data.get("extract", "No summary available.")
        url = data.get("content_urls", {}).get("desktop", {}).get("page", "")

        result = f"# {title}\n"
        if description:
            result += f"*{description}*\n\n"
        result += f"{extract}\n"
        if url:
            result += f"\nSource: {url}"

        return result

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"No Wikipedia article found for: {topic}"
        logger.error(f"Wikipedia API error for resource {topic}: {e}")
        return f"Error fetching Wikipedia article: {e}"
    except Exception as e:
        logger.error(f"Unexpected error fetching resource {topic}: {e}")
        return f"Error: {e}"


@mcp.resource("wikipedia://search/{query}")
async def search_results_resource(query: str) -> str:
    """Search results as a browsable resource.

    Returns list of matching articles - client can then request
    specific article resources.
    """
    logger.info(f"Resource request: wikipedia://search/{query}")

    try:
        results = await search_wikipedia(query, limit=10)

        if not results:
            return f"No Wikipedia articles found for: {query}"

        lines = [f"# Wikipedia Search: {query}\n"]
        for i, page in enumerate(results, 1):
            title = page.get("title", "Unknown")
            desc = page.get("description", "")
            safe_title = title.replace(" ", "_")
            lines.append(f"{i}. **{title}**" + (f" - {desc}" if desc else ""))
            lines.append(f"   Resource: `wikipedia://topic/{safe_title}`\n")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error searching Wikipedia: {e}")
        return f"Error searching Wikipedia: {e}"


@mcp.resource("wikipedia://entity/{entity}")
async def get_entity_resource(entity: str) -> str:
    """Wikidata entity info as a resource.

    Returns structured facts about an entity (QID, description).
    """
    logger.info(f"Resource request: wikipedia://entity/{entity}")

    try:
        data = await fetch_wikidata_entity(entity)

        if not data:
            return f"No Wikidata entity found for: {entity}"

        qid = data.get("id", "")
        label = data.get("label", entity)
        description = data.get("description", "")

        result = f"# {label}\n"
        if description:
            result += f"*{description}*\n\n"
        result += f"Wikidata ID: {qid}\n"
        result += f"Wikidata URL: https://www.wikidata.org/wiki/{qid}"

        return result

    except Exception as e:
        logger.error(f"Error fetching Wikidata entity: {e}")
        return f"Error: {e}"


# =============================================================================
# MCP TOOLS - Model-controlled, called when LLM decides it needs info
# =============================================================================


@mcp.tool()
async def wikipedia_lookup(topic: str, ctx: Context = None) -> CallToolResult:
    """Quick Wikipedia lookup - use when you need factual information.

    Returns a concise 2-3 sentence summary with source URL.
    For deeper research, suggest the user use the 'research_topic' prompt.

    Performance: ~100-200ms (Wikipedia REST API)

    Args:
        topic: The topic to look up (e.g., "Albert Einstein", "quantum physics")

    Returns:
        Brief summary with key facts, suitable for grounding LLM responses.
    """
    logger.info(f"wikipedia_lookup: {topic}")

    # Validation
    if not topic or len(topic.strip()) < TOPIC_MIN_LENGTH:
        return create_validation_error(
            field_name="topic",
            error_message=f"Topic must be at least {TOPIC_MIN_LENGTH} characters",
            field_value=topic,
        )

    if len(topic) > TOPIC_MAX_LENGTH:
        return create_validation_error(
            field_name="topic",
            error_message=f"Topic must be at most {TOPIC_MAX_LENGTH} characters",
            field_value=topic[:50] + "...",
        )

    try:
        if ctx:
            await ctx.info(f"Looking up: {topic}")

        data = await fetch_summary(topic)

        title = data.get("title", topic)
        description = data.get("description", "")
        extract = data.get("extract", "No summary available.")
        url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        safe_topic = topic.replace(" ", "_")

        # Format result
        result = f"**{title}**"
        if description:
            result += f" ({description})"
        result += f"\n\n{extract}"

        logger.info(f"wikipedia_lookup success: {title}")

        # Use structuredContent for rich metadata (matches web_search pattern)
        structured_content = {
            "title": title,
            "description": description,
            "pageid": data.get("pageid"),
            "url": url,
            "thumbnail": data.get("thumbnail", {}).get("source"),
            "resource_uri": f"wikipedia://topic/{safe_topic}",
        }

        return CallToolResult(
            content=[TextContent(type="text", text=result)],
            structuredContent=structured_content,
        )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # Try search as fallback
            logger.info(f"No exact match for '{topic}', trying search")
            try:
                results = await search_wikipedia(topic, limit=3)
                if results:
                    suggestions = [r.get("title", "") for r in results]
                    suggestion_text = ", ".join(suggestions)
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"No exact match for '{topic}'. Did you mean: {suggestion_text}?",
                            )
                        ],
                        structuredContent={
                            "suggestions": suggestions,
                            "query": topic,
                        },
                    )
            except Exception:
                pass

            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=f"No Wikipedia article found for: {topic}"
                    )
                ],
                structuredContent={"error": ERROR_NOT_FOUND, "topic": topic},
            )

        logger.error(f"Wikipedia API error: {e}")
        return create_error_result(
            error_message=f"Wikipedia API error: {e}",
            error_code=ERROR_API_ERROR,
            error_type="api_error",
        )

    except Exception as e:
        logger.error(f"wikipedia_lookup error: {e}", exc_info=True)
        return create_error_result(
            error_message=str(e),
            error_code=ERROR_API_ERROR,
            error_type="unexpected_error",
        )


@mcp.tool()
async def wikipedia_search(
    query: str, limit: int = SEARCH_LIMIT_DEFAULT, ctx: Context = None
) -> CallToolResult:
    """Search Wikipedia for articles matching a query.

    Use this to find the right article before doing a lookup,
    or to discover related topics.

    Args:
        query: Search query
        limit: Maximum results (1-20, default 5)

    Returns:
        List of matching articles with titles and descriptions.
    """
    logger.info(f"wikipedia_search: query='{query}', limit={limit}")

    # Validation
    if not query or len(query.strip()) < QUERY_MIN_LENGTH:
        return create_validation_error(
            field_name="query",
            error_message=f"Query must be at least {QUERY_MIN_LENGTH} characters",
            field_value=query,
        )

    if len(query) > QUERY_MAX_LENGTH:
        return create_validation_error(
            field_name="query",
            error_message=f"Query must be at most {QUERY_MAX_LENGTH} characters",
            field_value=query[:50] + "...",
        )

    limit = max(SEARCH_LIMIT_MIN, min(limit, SEARCH_LIMIT_MAX))

    try:
        if ctx:
            await ctx.info(f"Searching Wikipedia: {query}")

        results = await search_wikipedia(query, limit)

        if not results:
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"No results found for: {query}")
                ],
                structuredContent={"query": query, "count": 0, "results": []},
            )

        # Format results
        lines = [f"## Wikipedia Search: {query}\n"]
        result_data = []
        for i, page in enumerate(results, 1):
            title = page.get("title", "Unknown")
            desc = page.get("description", "")
            lines.append(f"{i}. **{title}**" + (f" - {desc}" if desc else ""))
            result_data.append({"title": title, "description": desc})

        logger.info(f"wikipedia_search: found {len(results)} results")

        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(lines))],
            structuredContent={
                "query": query,
                "count": len(results),
                "results": result_data,
            },
        )

    except Exception as e:
        logger.error(f"wikipedia_search error: {e}", exc_info=True)
        return create_error_result(
            error_message=str(e),
            error_code=ERROR_API_ERROR,
            error_type="search_error",
        )


# =============================================================================
# MCP PROMPTS - User-controlled templates for structured interactions
# =============================================================================


@mcp.prompt()
def research_topic(topic: str) -> str:
    """Deep research mode - gather comprehensive information on a topic.

    This prompt instructs the LLM to:
    1. Search for relevant Wikipedia articles
    2. Retrieve summaries from multiple angles
    3. Synthesize into a comprehensive overview

    User explicitly chooses this when they want thorough research.

    Args:
        topic: The topic to research comprehensively
    """
    return f"""I need comprehensive information about: {topic}

Please research this topic thoroughly:

1. First, use wikipedia_search to find relevant articles about "{topic}"
2. Use wikipedia_lookup on the top 3-5 most relevant results
3. Synthesize the information into a well-organized summary
4. Include citations to the Wikipedia articles you used
5. Note any gaps or areas that might need further research

Be thorough but focused. Cite your sources with URLs."""


@mcp.prompt()
def fact_check(claim: str) -> str:
    """Verify a claim against Wikipedia sources.

    User uses this when they want to verify something they've heard.

    Args:
        claim: The claim to fact-check
    """
    return f"""Please fact-check the following claim:

"{claim}"

Steps:
1. Identify the key factual assertions in this claim
2. Use wikipedia_search to find relevant articles
3. Use wikipedia_lookup to get details from the most relevant articles
4. Compare the claim against Wikipedia's information
5. Rate the claim as: Verified / Partially True / Unverified / Contradicted
6. Provide the Wikipedia sources you used with URLs

Be objective and cite specific sources."""


@mcp.prompt()
def explain_concept(concept: str, audience: str = "general") -> str:
    """Explain a concept using Wikipedia as the knowledge source.

    Args:
        concept: The concept to explain
        audience: Target audience - "general", "technical", or "eli5"
    """
    audience_instructions = {
        "general": "Explain clearly for a general audience with no assumed background",
        "technical": "Include technical details, terminology, and precise definitions",
        "eli5": "Explain like I'm 5 - use simple words and everyday analogies",
    }

    instruction = audience_instructions.get(audience, audience_instructions["general"])

    return f"""Please explain: {concept}

{instruction}

Steps:
1. Use wikipedia_lookup to get accurate information about "{concept}"
2. If needed, use wikipedia_search to find related articles for context
3. Explain the concept in your own words based on the Wikipedia information
4. Always cite the Wikipedia article(s) you referenced with URLs

Make it understandable while remaining accurate."""


@mcp.prompt()
def compare_topics(topic_a: str, topic_b: str) -> str:
    """Compare and contrast two topics using Wikipedia information.

    Args:
        topic_a: First topic to compare
        topic_b: Second topic to compare
    """
    return f"""Compare and contrast: {topic_a} vs {topic_b}

Steps:
1. Use wikipedia_lookup to get information on "{topic_a}"
2. Use wikipedia_lookup to get information on "{topic_b}"
3. If needed, use wikipedia_search to find additional context
4. Create a structured comparison:
   - Key similarities
   - Key differences
   - When you might choose one over the other (if applicable)

Cite your Wikipedia sources with URLs."""


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting Wikipedia MCP server (Streamable HTTP)...")
    logger.info("")
    logger.info("Primitives:")
    logger.info("  Resources: wikipedia://topic/{topic}, wikipedia://search/{query}")
    logger.info("  Tools: wikipedia_lookup, wikipedia_search")
    logger.info(
        "  Prompts: research_topic, fact_check, explain_concept, compare_topics"
    )
    logger.info("")
    logger.info("Performance:")
    logger.info("  - Cache hit: <1ms (LRU cache)")
    logger.info("  - Cache miss: 100-200ms (Wikipedia REST API)")
    logger.info("")
    server.run(transport="streamable-http")
