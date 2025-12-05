"""ABOUTME: HTML preprocessing, metadata extraction, and markdown conversion utilities.

Uses Trafilatura (F1: 0.958) for content extraction - better than ReadabiliPy (F1: 0.92).
Used by HuggingFace, IBM, Microsoft Research.

Features:
- Fast path: Trafilatura extraction + markdownify (rule-based, ~100ms)
- Metadata extraction: title, author, date, language, sitename
- Links extraction: internal/external URLs
- BM25 query filtering: extract only content relevant to a query
- Multiple output formats: markdown, html, text
"""

import logging
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

try:
    import trafilatura
    from trafilatura.settings import use_config
    from trafilatura.metadata import extract_metadata as traf_extract_metadata
    TRAFILATURA_AVAILABLE = True
except ImportError:
    trafilatura = None
    TRAFILATURA_AVAILABLE = False

try:
    from markdownify import markdownify as md
    MARKDOWNIFY_AVAILABLE = True
except ImportError:
    md = None
    MARKDOWNIFY_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None
    BM25_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PageMetadata:
    """Metadata extracted from a web page."""
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    sitename: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    categories: List[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.categories is None:
            self.categories = []
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtractedLink:
    """A link extracted from the page."""
    url: str
    text: str
    is_internal: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Constants
# =============================================================================

# Minimum content length to consider extraction successful
MIN_CONTENT_LENGTH = 50

# =============================================================================
# Utility Functions
# =============================================================================

def strip_scripts_and_styles(html: str) -> str:
    """Remove script and style tags from HTML."""
    html = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', html, flags=re.IGNORECASE)
    html = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '', html, flags=re.IGNORECASE)
    html = re.sub(r'<!--[\s\S]*?-->', '', html)
    return html


def tokenize_simple(text: str) -> List[str]:
    """Simple tokenizer for BM25 - lowercase and split on non-alphanumeric."""
    return re.findall(r'\b\w+\b', text.lower())


def is_valid_content(content: Optional[str], min_length: int = MIN_CONTENT_LENGTH) -> bool:
    """Check if extracted content is valid (not empty, not just whitespace, meets minimum length).

    Args:
        content: The extracted content to validate
        min_length: Minimum character length for valid content

    Returns:
        True if content is valid, False otherwise
    """
    if not content:
        return False
    stripped = content.strip()
    if len(stripped) < min_length:
        return False
    # Check if it's just whitespace/newlines
    if not re.search(r'[a-zA-Z0-9]', stripped):
        return False
    return True


def extract_title_fallback(html: str) -> Optional[str]:
    """Extract title from HTML using regex as fallback.

    Args:
        html: Raw HTML content

    Returns:
        Title string or None if not found
    """
    # Try <title> tag
    match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
    if match:
        title = match.group(1).strip()
        if title:
            return title

    # Try og:title meta tag
    match = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try twitter:title meta tag
    match = re.search(r'<meta[^>]+name=["\']twitter:title["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


# =============================================================================
# Metadata Extraction
# =============================================================================

def extract_metadata(html: str) -> PageMetadata:
    """Extract metadata from HTML using Trafilatura with regex fallbacks.

    Extracts: title, author, date, sitename, description, language, categories, tags.
    Uses regex fallback for title if Trafilatura fails.

    Args:
        html: Raw HTML content

    Returns:
        PageMetadata object with extracted fields
    """
    result = PageMetadata()

    if not TRAFILATURA_AVAILABLE:
        logger.warning("Trafilatura not available for metadata extraction, using fallback")
        # Use fallback for title at minimum
        result.title = extract_title_fallback(html)
        return result

    try:
        # Use Trafilatura's metadata extraction
        metadata = traf_extract_metadata(html)

        if metadata is not None:
            result = PageMetadata(
                title=metadata.title if hasattr(metadata, 'title') else None,
                author=metadata.author if hasattr(metadata, 'author') else None,
                date=metadata.date if hasattr(metadata, 'date') else None,
                sitename=metadata.sitename if hasattr(metadata, 'sitename') else None,
                description=metadata.description if hasattr(metadata, 'description') else None,
                language=metadata.language if hasattr(metadata, 'language') else None,
                categories=list(metadata.categories) if hasattr(metadata, 'categories') and metadata.categories else [],
                tags=list(metadata.tags) if hasattr(metadata, 'tags') and metadata.tags else []
            )

        # Fallback: if title is still None, try regex extraction
        if result.title is None:
            result.title = extract_title_fallback(html)
            if result.title:
                logger.debug(f"Used regex fallback for title: {result.title[:50]}...")

        return result

    except Exception as e:
        logger.warning(f"Metadata extraction failed: {e}")
        # Still try to get title via fallback
        result.title = extract_title_fallback(html)
        return result


# =============================================================================
# Links Extraction
# =============================================================================

def extract_links(html: str, base_url: Optional[str] = None) -> List[ExtractedLink]:
    """Extract all links from HTML.

    Args:
        html: Raw HTML content
        base_url: Base URL to determine internal vs external links

    Returns:
        List of ExtractedLink objects
    """
    links = []

    # Extract href and link text using regex
    link_pattern = re.compile(
        r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL
    )

    base_domain = None
    if base_url:
        # Extract domain from base URL
        domain_match = re.match(r'https?://([^/]+)', base_url)
        if domain_match:
            base_domain = domain_match.group(1).lower()

    for match in link_pattern.finditer(html):
        url = match.group(1).strip()
        text = re.sub(r'<[^>]+>', '', match.group(2)).strip()  # Remove nested HTML

        # Skip empty, anchor-only, or javascript links
        if not url or url.startswith('#') or url.startswith('javascript:'):
            continue

        # Determine if internal
        is_internal = False
        if base_domain:
            if url.startswith('/') or url.startswith('./'):
                is_internal = True
            elif base_domain in url.lower():
                is_internal = True

        links.append(ExtractedLink(url=url, text=text, is_internal=is_internal))

    return links


# =============================================================================
# BM25 Query Filtering
# =============================================================================

def filter_content_by_query(
    content: str,
    query: str,
    top_k: int = 10,
    min_score: float = 0.0
) -> tuple[str, dict]:
    """Filter content to only include paragraphs relevant to the query using BM25.

    This is useful for RAG applications where you only want content relevant
    to a specific question or topic.

    Args:
        content: Text content (markdown or plain text)
        query: Query string to filter by
        top_k: Maximum number of paragraphs to return
        min_score: Minimum BM25 score to include (0.0 = include all top_k)

    Returns:
        Tuple of (filtered_content, metadata)
    """
    metadata = {
        "query": query,
        "bm25_available": BM25_AVAILABLE,
        "original_paragraphs": 0,
        "filtered_paragraphs": 0,
        "top_scores": []
    }

    if not BM25_AVAILABLE:
        logger.warning("rank-bm25 not available, returning full content")
        return content, metadata

    if not query or not query.strip():
        return content, metadata

    # Split content into paragraphs (double newline or single newline with content)
    paragraphs = re.split(r'\n\s*\n', content)
    paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]

    metadata["original_paragraphs"] = len(paragraphs)

    if len(paragraphs) == 0:
        return content, metadata

    if len(paragraphs) <= top_k:
        # Not enough paragraphs to filter
        metadata["filtered_paragraphs"] = len(paragraphs)
        return content, metadata

    try:
        # Tokenize paragraphs and query
        tokenized_paragraphs = [tokenize_simple(p) for p in paragraphs]
        tokenized_query = tokenize_simple(query)

        # Create BM25 index
        bm25 = BM25Okapi(tokenized_paragraphs)

        # Score paragraphs
        scores = bm25.get_scores(tokenized_query)

        # Get top-k paragraphs by score
        scored_paragraphs = list(zip(paragraphs, scores, range(len(paragraphs))))
        scored_paragraphs.sort(key=lambda x: x[1], reverse=True)

        # Filter by min_score and take top_k
        filtered = [
            (p, score, idx) for p, score, idx in scored_paragraphs[:top_k]
            if score >= min_score
        ]

        # Sort by original order to maintain document flow
        filtered.sort(key=lambda x: x[2])

        metadata["filtered_paragraphs"] = len(filtered)
        metadata["top_scores"] = [round(score, 3) for _, score, _ in filtered[:5]]

        # Reconstruct content
        filtered_content = "\n\n".join(p for p, _, _ in filtered)

        logger.info(
            f"BM25 filtered {len(paragraphs)} paragraphs to {len(filtered)} "
            f"for query: '{query[:50]}...'" if len(query) > 50 else f"for query: '{query}'"
        )

        return filtered_content, metadata

    except Exception as e:
        logger.error(f"BM25 filtering failed: {e}")
        metadata["error"] = str(e)
        return content, metadata


# =============================================================================
# Content Extraction
# =============================================================================

def extract_with_trafilatura(
    html: str,
    include_links: bool = True,
    include_images: bool = True,
    include_tables: bool = True,
    output_format: str = "html"
) -> tuple[Optional[str], bool, dict]:
    """Extract main content from HTML using Trafilatura with fallback chain.

    Trafilatura achieves F1: 0.958 on benchmarks (vs ReadabiliPy ~0.92).
    Uses a 2-step fallback: first tries precision mode, then favor_recall=True.

    Args:
        html: Raw HTML content
        include_links: Whether to preserve links (default True)
        include_images: Whether to preserve image references (default True)
        include_tables: Whether to preserve tables (default True)
        output_format: Output format - "html", "markdown", or "txt"

    Returns:
        Tuple of (extracted_content, success, metadata)
    """
    metadata = {
        "extractor": "trafilatura",
        "extraction_success": False,
        "output_format": output_format,
        "fallback_used": False
    }

    if not TRAFILATURA_AVAILABLE:
        logger.warning("Trafilatura not available, falling back to script stripping")
        cleaned = strip_scripts_and_styles(html)
        metadata["extractor"] = "fallback_strip"
        return cleaned, False, metadata

    try:
        # Configure trafilatura for optimal extraction
        config = use_config()
        config.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")

        # Map output_format to trafilatura's expected values
        traf_format = output_format
        if output_format == "text":
            traf_format = "txt"

        # First attempt: standard extraction (precision mode)
        extracted = trafilatura.extract(
            html,
            include_links=include_links,
            include_images=include_images,
            include_tables=include_tables,
            output_format=traf_format,
            config=config
        )

        # Check if extraction is valid
        if not is_valid_content(extracted):
            # Fallback: try with favor_recall=True (captures more content)
            logger.debug("First extraction attempt yielded insufficient content, trying favor_recall=True")
            extracted = trafilatura.extract(
                html,
                include_links=include_links,
                include_images=include_images,
                include_tables=include_tables,
                output_format=traf_format,
                favor_recall=True,
                config=config
            )
            metadata["fallback_used"] = True

        if is_valid_content(extracted):
            original_size = len(html)
            extracted_size = len(extracted)
            compression = (original_size - extracted_size) / original_size * 100

            logger.info(
                f"Trafilatura extracted content: {original_size} → {extracted_size} bytes "
                f"({compression:.1f}% reduction){' [favor_recall]' if metadata['fallback_used'] else ''}"
            )

            metadata["extraction_success"] = True
            metadata["original_size"] = original_size
            metadata["extracted_size"] = extracted_size
            metadata["compression_percent"] = round(compression, 1)

            return extracted, True, metadata
        else:
            logger.debug("Trafilatura extraction returned insufficient content, falling back to strip")
            cleaned = strip_scripts_and_styles(html)
            metadata["extractor"] = "fallback_strip"
            return cleaned, False, metadata

    except Exception as e:
        logger.warning(f"Trafilatura extraction failed: {e}, falling back to strip approach")
        cleaned = strip_scripts_and_styles(html)
        metadata["extractor"] = "fallback_strip"
        metadata["error"] = str(e)
        return cleaned, False, metadata


def html_to_markdown_fast(html: str) -> tuple[str, bool, dict]:
    """Convert HTML to Markdown using rule-based markdownify.

    This is the fast path (~50-100ms) that doesn't require LLM inference.
    Good for 90%+ of pages. Use LLM path for complex/broken HTML.

    Args:
        html: HTML content to convert

    Returns:
        Tuple of (markdown_content, success, metadata)
    """
    metadata = {
        "converter": "markdownify",
        "conversion_success": False
    }

    if not MARKDOWNIFY_AVAILABLE:
        logger.warning("markdownify not available, returning raw HTML")
        metadata["converter"] = "none"
        return html, False, metadata

    try:
        # Convert HTML to Markdown with sensible defaults
        markdown = md(
            html,
            heading_style="ATX",
            bullets="-",
            code_language="",
            strip=['script', 'style'],
            convert=['a', 'b', 'blockquote', 'br', 'code', 'em', 'h1', 'h2',
                     'h3', 'h4', 'h5', 'h6', 'hr', 'i', 'img', 'li', 'ol',
                     'p', 'pre', 'strong', 'table', 'tbody', 'td', 'th',
                     'thead', 'tr', 'ul']
        )

        if markdown:
            # Clean up excessive whitespace
            markdown = re.sub(r'\n{3,}', '\n\n', markdown)
            markdown = markdown.strip()

            metadata["conversion_success"] = True
            metadata["output_size"] = len(markdown)

            logger.info(f"markdownify converted HTML to {len(markdown)} chars")
            return markdown, True, metadata
        else:
            logger.warning("markdownify returned empty result")
            return html, False, metadata

    except Exception as e:
        logger.error(f"markdownify conversion failed: {e}")
        metadata["error"] = str(e)
        return html, False, metadata


# =============================================================================
# Main Pipeline Functions
# =============================================================================

def preprocess_html(
    html: str,
    use_trafilatura: bool = True,
    output_format: str = "html"
) -> tuple[str, dict]:
    """Preprocess HTML for conversion (extraction step only).

    This extracts main content but doesn't convert to markdown.
    Use html_to_markdown_fast() or ReaderLM for the conversion step.

    Args:
        html: Raw HTML content
        use_trafilatura: Whether to use Trafilatura extraction (default True)
        output_format: Trafilatura output format - "html", "markdown", or "text"

    Returns:
        Tuple of (processed_content, metadata)
    """
    metadata = {
        'original_size_bytes': len(html.encode('utf-8')),
        'trafilatura_used': False,
        'scripts_stripped': False
    }

    processed = html

    if use_trafilatura:
        extracted, success, extract_meta = extract_with_trafilatura(
            processed,
            output_format=output_format
        )
        processed = extracted
        metadata['trafilatura_used'] = success
        metadata['extraction_metadata'] = extract_meta
    else:
        processed = strip_scripts_and_styles(processed)
        metadata['scripts_stripped'] = True

    metadata['final_size_bytes'] = len(processed.encode('utf-8') if isinstance(processed, str) else processed)

    if metadata['original_size_bytes'] > 0:
        compression = (
            (metadata['original_size_bytes'] - metadata['final_size_bytes']) /
            metadata['original_size_bytes'] * 100
        )
        metadata['compression_percent'] = round(compression, 1)
    else:
        metadata['compression_percent'] = 0.0

    if metadata['compression_percent'] > 10:
        logger.info(
            f"HTML preprocessed: {metadata['original_size_bytes']} → "
            f"{metadata['final_size_bytes']} bytes ({metadata['compression_percent']:.1f}% reduction)"
        )

    return processed, metadata


def extract_and_convert_to_markdown(
    html: str,
    use_trafilatura: bool = True,
    include_links: bool = True,
    query: Optional[str] = None,
    query_top_k: int = 10
) -> tuple[str, bool, dict]:
    """Full fast-path: Extract content with Trafilatura + convert to Markdown with markdownify.

    This is the recommended default path - no LLM needed, ~100-200ms total.
    Achieves ~95% accuracy on most pages.

    Args:
        html: Raw HTML content
        use_trafilatura: Whether to use Trafilatura for extraction
        include_links: Whether to preserve links in output
        query: Optional query for BM25 filtering (returns only relevant content)
        query_top_k: Max paragraphs to return when filtering by query

    Returns:
        Tuple of (markdown_content, success, metadata)
    """
    metadata = {
        "pipeline": "fast_path",
        "extraction": {},
        "conversion": {},
        "bm25_filter": None,
        "content_valid": False
    }

    # Step 1: Extract main content with Trafilatura (output as HTML for markdownify)
    extracted_html, extract_success, extract_meta = extract_with_trafilatura(
        html,
        include_links=include_links,
        output_format="html"
    )
    metadata["extraction"] = extract_meta

    if not extracted_html:
        return "", False, metadata

    # Step 2: Convert to Markdown with markdownify
    markdown, convert_success, convert_meta = html_to_markdown_fast(extracted_html)
    metadata["conversion"] = convert_meta

    # Step 3: Optional BM25 query filtering
    if query and convert_success:
        markdown, bm25_meta = filter_content_by_query(
            markdown,
            query,
            top_k=query_top_k
        )
        metadata["bm25_filter"] = bm25_meta

    # Step 4: Validate final content meets minimum threshold
    content_valid = is_valid_content(markdown)
    metadata["content_valid"] = content_valid

    # Overall success requires extraction, conversion, AND valid content
    overall_success = extract_success and convert_success and content_valid

    if not content_valid and convert_success:
        logger.warning(
            f"Content validation failed: extracted {len(markdown) if markdown else 0} chars "
            f"(minimum: {MIN_CONTENT_LENGTH})"
        )

    if overall_success:
        logger.info(
            f"Fast path complete: {len(html)} bytes HTML → {len(markdown)} chars Markdown"
        )

    return markdown, overall_success, metadata


def process_html_full(
    html: str,
    output_format: str = "markdown",
    include_metadata: bool = False,
    include_links: bool = False,
    query: Optional[str] = None,
    query_top_k: int = 10,
    base_url: Optional[str] = None
) -> dict:
    """Full processing pipeline with all features.

    Args:
        html: Raw HTML content
        output_format: "markdown", "html", or "text"
        include_metadata: Extract and return page metadata
        include_links: Extract and return all links
        query: Optional BM25 query to filter content by relevance
        query_top_k: Max paragraphs when filtering by query
        base_url: Base URL for determining internal/external links

    Returns:
        Dict with keys: content, success, metadata, page_metadata, links
    """
    result = {
        "content": "",
        "success": False,
        "metadata": {},
        "page_metadata": None,
        "links": None
    }

    # Extract metadata if requested
    if include_metadata:
        page_meta = extract_metadata(html)
        result["page_metadata"] = page_meta.to_dict()

    # Extract links if requested
    if include_links:
        links = extract_links(html, base_url)
        result["links"] = [link.to_dict() for link in links]

    # Process content based on output format
    if output_format == "text":
        # Use Trafilatura's text output directly
        content, success, meta = extract_with_trafilatura(
            html,
            output_format="txt",
            include_links=False,
            include_images=False
        )
        result["content"] = content or ""
        result["success"] = success
        result["metadata"] = meta

    elif output_format == "html":
        # Return cleaned HTML
        content, success, meta = extract_with_trafilatura(
            html,
            output_format="html"
        )
        result["content"] = content or ""
        result["success"] = success
        result["metadata"] = meta

    else:  # markdown (default)
        content, success, meta = extract_and_convert_to_markdown(
            html,
            query=query,
            query_top_k=query_top_k
        )
        result["content"] = content
        result["success"] = success
        result["metadata"] = meta

    # Apply BM25 filter for non-markdown formats if query provided
    if query and output_format != "markdown" and result["content"]:
        filtered, bm25_meta = filter_content_by_query(
            result["content"],
            query,
            top_k=query_top_k
        )
        result["content"] = filtered
        result["metadata"]["bm25_filter"] = bm25_meta

    return result
