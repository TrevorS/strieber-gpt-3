"""ABOUTME: HTML preprocessing, metadata extraction, and markdown conversion utilities.

Uses Trafilatura (F1: 0.958) for content extraction with readability-lxml ensemble fallback.
Ensemble approaches achieve F1: 0.981 per research benchmarks.

Features:
- Fast path: Trafilatura extraction + markdownify (rule-based, ~100ms)
- Ensemble fallback: readability-lxml when Trafilatura fails
- Metadata extraction: title, author, date, language, sitename
- Links extraction: internal/external URLs
- BM25 query filtering: extract only content relevant to a query
- Multiple output formats: markdown, html, text
- Language filtering: discard non-matching content
- Deduplication: remove duplicate segments
- XPath pruning: remove boilerplate elements
"""

import logging
import re
from dataclasses import asdict, dataclass
from typing import Optional

import trafilatura
from lxml import etree
from markdownify import markdownify as md
from rank_bm25 import BM25Okapi
from readability import Document as ReadabilityDocument
from trafilatura.metadata import extract_metadata as traf_extract_metadata
from trafilatura.settings import use_config

logger = logging.getLogger(__name__)

# Minimum content length to consider extraction successful
MIN_CONTENT_LENGTH = 50

# Default XPath patterns to prune (common boilerplate elements)
DEFAULT_PRUNE_XPATH = [
    '//div[contains(@class, "cookie")]',
    '//div[contains(@class, "Cookie")]',
    '//div[contains(@id, "cookie")]',
    '//div[contains(@class, "consent")]',
    '//div[contains(@class, "newsletter")]',
    '//div[contains(@class, "subscribe")]',
    '//div[contains(@class, "popup")]',
    '//div[contains(@class, "modal")]',
    '//div[contains(@class, "advertisement")]',
    '//div[contains(@class, "ad-")]',
    '//div[contains(@class, "ads-")]',
    '//aside[contains(@class, "sidebar")]',
    '//nav[contains(@class, "breadcrumb")]',
    '//div[contains(@class, "share")]',
    '//div[contains(@class, "social")]',
    '//footer',
    '//header[contains(@class, "site-header")]',
]


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
    categories: list[str] = None
    tags: list[str] = None

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


@dataclass
class ExtractionOptions:
    """Options for content extraction."""

    favor_precision: bool = False
    favor_recall: bool = False
    deduplicate: bool = False
    target_language: Optional[str] = None  # ISO 639-1 code (e.g., "en", "de")
    prune_xpath: Optional[list[str]] = None
    fast: bool = False
    include_links: bool = True
    include_images: bool = True
    include_tables: bool = True


# =============================================================================
# XPath Pruning
# =============================================================================


def prune_html_xpath(html: str, xpath_patterns: list[str]) -> str:
    """Remove elements matching XPath patterns from HTML.

    Args:
        html: HTML content
        xpath_patterns: List of XPath expressions to remove

    Returns:
        Cleaned HTML with matched elements removed
    """
    if not xpath_patterns:
        return html

    try:
        # Parse HTML
        tree = etree.HTML(html)
        if tree is None:
            return html

        removed_count = 0
        for pattern in xpath_patterns:
            try:
                elements = tree.xpath(pattern)
                for elem in elements:
                    parent = elem.getparent()
                    if parent is not None:
                        parent.remove(elem)
                        removed_count += 1
            except etree.XPathError as e:
                logger.debug(f"Invalid XPath pattern '{pattern}': {e}")
                continue

        if removed_count > 0:
            logger.debug(f"Pruned {removed_count} elements via XPath")
            return etree.tostring(tree, encoding="unicode", method="html")

        return html

    except Exception as e:
        logger.warning(f"XPath pruning failed: {e}")
        return html


# =============================================================================
# Utility Functions
# =============================================================================


def strip_scripts_and_styles(html: str) -> str:
    """Remove script and style tags from HTML."""
    html = re.sub(
        r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>", "", html, flags=re.IGNORECASE
    )
    html = re.sub(
        r"<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>", "", html, flags=re.IGNORECASE
    )
    html = re.sub(r"<!--[\s\S]*?-->", "", html)
    return html


def tokenize_simple(text: str) -> list[str]:
    """Simple tokenizer for BM25 - lowercase and split on non-alphanumeric."""
    return re.findall(r"\b\w+\b", text.lower())


def is_valid_content(content: Optional[str], min_length: int = MIN_CONTENT_LENGTH) -> bool:
    """Check if extracted content is valid (not empty, not just whitespace, meets minimum length)."""
    if not content:
        return False
    stripped = content.strip()
    if len(stripped) < min_length:
        return False
    if not re.search(r"[a-zA-Z0-9]", stripped):
        return False
    return True


def extract_title_fallback(html: str) -> Optional[str]:
    """Extract title from HTML using regex as fallback."""
    # Try <title> tag
    match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    if match:
        title = match.group(1).strip()
        if title:
            return title

    # Try og:title meta tag
    match = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Try twitter:title meta tag
    match = re.search(
        r'<meta[^>]+name=["\']twitter:title["\'][^>]+content=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    return None


# =============================================================================
# Metadata Extraction
# =============================================================================


def extract_metadata(html: str) -> PageMetadata:
    """Extract metadata from HTML using Trafilatura with regex fallbacks."""
    result = PageMetadata()

    try:
        metadata = traf_extract_metadata(html)

        if metadata is not None:
            result = PageMetadata(
                title=metadata.title if hasattr(metadata, "title") else None,
                author=metadata.author if hasattr(metadata, "author") else None,
                date=metadata.date if hasattr(metadata, "date") else None,
                sitename=metadata.sitename if hasattr(metadata, "sitename") else None,
                description=metadata.description if hasattr(metadata, "description") else None,
                language=metadata.language if hasattr(metadata, "language") else None,
                categories=list(metadata.categories)
                if hasattr(metadata, "categories") and metadata.categories
                else [],
                tags=list(metadata.tags) if hasattr(metadata, "tags") and metadata.tags else [],
            )

        # Fallback: if title is still None, try regex extraction
        if result.title is None:
            result.title = extract_title_fallback(html)
            if result.title:
                logger.debug(f"Used regex fallback for title: {result.title[:50]}...")

        return result

    except Exception as e:
        logger.warning(f"Metadata extraction failed: {e}")
        result.title = extract_title_fallback(html)
        return result


# =============================================================================
# Links Extraction
# =============================================================================


def extract_links(html: str, base_url: Optional[str] = None) -> list[ExtractedLink]:
    """Extract all links from HTML."""
    links = []

    link_pattern = re.compile(
        r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL
    )

    base_domain = None
    if base_url:
        domain_match = re.match(r"https?://([^/]+)", base_url)
        if domain_match:
            base_domain = domain_match.group(1).lower()

    for match in link_pattern.finditer(html):
        url = match.group(1).strip()
        text = re.sub(r"<[^>]+>", "", match.group(2)).strip()

        if not url or url.startswith("#") or url.startswith("javascript:"):
            continue

        is_internal = False
        if base_domain:
            if url.startswith("/") or url.startswith("./"):
                is_internal = True
            elif base_domain in url.lower():
                is_internal = True

        links.append(ExtractedLink(url=url, text=text, is_internal=is_internal))

    return links


# =============================================================================
# BM25 Query Filtering
# =============================================================================


def filter_content_by_query(
    content: str, query: str, top_k: int = 10, min_score: float = 0.0
) -> tuple[str, dict]:
    """Filter content to only include paragraphs relevant to the query using BM25."""
    metadata = {
        "query": query,
        "original_paragraphs": 0,
        "filtered_paragraphs": 0,
        "top_scores": [],
    }

    if not query or not query.strip():
        return content, metadata

    paragraphs = re.split(r"\n\s*\n", content)
    paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]

    metadata["original_paragraphs"] = len(paragraphs)

    if len(paragraphs) == 0:
        return content, metadata

    if len(paragraphs) <= top_k:
        metadata["filtered_paragraphs"] = len(paragraphs)
        return content, metadata

    try:
        tokenized_paragraphs = [tokenize_simple(p) for p in paragraphs]
        tokenized_query = tokenize_simple(query)

        bm25 = BM25Okapi(tokenized_paragraphs)
        scores = bm25.get_scores(tokenized_query)

        scored_paragraphs = list(zip(paragraphs, scores, range(len(paragraphs))))
        scored_paragraphs.sort(key=lambda x: x[1], reverse=True)

        filtered = [
            (p, score, idx) for p, score, idx in scored_paragraphs[:top_k] if score >= min_score
        ]
        filtered.sort(key=lambda x: x[2])

        metadata["filtered_paragraphs"] = len(filtered)
        metadata["top_scores"] = [round(score, 3) for _, score, _ in filtered[:5]]

        filtered_content = "\n\n".join(p for p, _, _ in filtered)

        logger.info(
            f"BM25 filtered {len(paragraphs)} paragraphs to {len(filtered)} "
            f"for query: '{query[:50]}...'"
            if len(query) > 50
            else f"for query: '{query}'"
        )

        return filtered_content, metadata

    except Exception as e:
        logger.error(f"BM25 filtering failed: {e}")
        metadata["error"] = str(e)
        return content, metadata


# =============================================================================
# Readability Fallback (Ensemble)
# =============================================================================


def extract_with_readability(html: str) -> tuple[Optional[str], bool, dict]:
    """Extract content using readability-lxml as fallback.

    Readability achieves median F1 of 0.970 on benchmarks.
    """
    metadata = {
        "extractor": "readability",
        "extraction_success": False,
    }

    try:
        doc = ReadabilityDocument(html)
        content = doc.summary()
        title = doc.title()

        if content and is_valid_content(content):
            metadata["extraction_success"] = True
            metadata["title"] = title
            metadata["content_size"] = len(content)
            logger.info(f"Readability extracted {len(content)} chars")
            return content, True, metadata

        return None, False, metadata

    except Exception as e:
        logger.warning(f"Readability extraction failed: {e}")
        metadata["error"] = str(e)
        return None, False, metadata


# =============================================================================
# Content Extraction
# =============================================================================


def extract_with_trafilatura(
    html: str,
    include_links: bool = True,
    include_images: bool = True,
    include_tables: bool = True,
    output_format: str = "html",
    options: Optional[ExtractionOptions] = None,
) -> tuple[Optional[str], bool, dict]:
    """Extract main content from HTML using Trafilatura with ensemble fallback.

    Uses a multi-step fallback chain:
    1. Standard Trafilatura extraction
    2. Trafilatura with favor_recall=True
    3. Readability-lxml (ensemble member)
    """
    if options is None:
        options = ExtractionOptions()

    metadata = {
        "extractor": "trafilatura",
        "extraction_success": False,
        "output_format": output_format,
        "fallback_used": None,
        "options": {
            "favor_precision": options.favor_precision,
            "favor_recall": options.favor_recall,
            "deduplicate": options.deduplicate,
            "target_language": options.target_language,
            "fast": options.fast,
        },
    }

    # Apply XPath pruning if configured
    prune_patterns = options.prune_xpath if options.prune_xpath else DEFAULT_PRUNE_XPATH
    html = prune_html_xpath(html, prune_patterns)

    try:
        config = use_config()
        config.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")

        traf_format = "txt" if output_format == "text" else output_format

        # Build extraction kwargs
        extract_kwargs = {
            "include_links": include_links,
            "include_images": include_images,
            "include_tables": include_tables,
            "output_format": traf_format,
            "config": config,
            "deduplicate": options.deduplicate,
        }

        # Add precision/recall preference
        if options.favor_precision:
            extract_kwargs["favor_precision"] = True
        elif options.favor_recall:
            extract_kwargs["favor_recall"] = True

        # Add language filter
        if options.target_language:
            extract_kwargs["target_language"] = options.target_language

        # Add fast mode
        if options.fast:
            extract_kwargs["fast"] = True

        # First attempt: configured extraction
        extracted = trafilatura.extract(html, **extract_kwargs)

        # Check if extraction is valid
        if not is_valid_content(extracted):
            # Second attempt: try favor_recall if not already using it
            if not options.favor_recall and not options.favor_precision:
                logger.debug("First extraction insufficient, trying favor_recall=True")
                extract_kwargs["favor_recall"] = True
                extracted = trafilatura.extract(html, **extract_kwargs)
                metadata["fallback_used"] = "favor_recall"

        # Third attempt: readability-lxml ensemble fallback
        if not is_valid_content(extracted):
            logger.debug("Trafilatura extraction insufficient, trying readability-lxml")
            readability_content, readability_success, readability_meta = extract_with_readability(
                html
            )
            if readability_success and readability_content:
                metadata["fallback_used"] = "readability"
                metadata["readability_meta"] = readability_meta
                extracted = readability_content

        if is_valid_content(extracted):
            original_size = len(html)
            extracted_size = len(extracted)
            compression = (original_size - extracted_size) / original_size * 100

            fallback_info = f" [{metadata['fallback_used']}]" if metadata["fallback_used"] else ""
            logger.info(
                f"Extracted content: {original_size} → {extracted_size} bytes "
                f"({compression:.1f}% reduction){fallback_info}"
            )

            metadata["extraction_success"] = True
            metadata["original_size"] = original_size
            metadata["extracted_size"] = extracted_size
            metadata["compression_percent"] = round(compression, 1)

            return extracted, True, metadata
        else:
            logger.debug("All extraction methods returned insufficient content")
            cleaned = strip_scripts_and_styles(html)
            metadata["extractor"] = "fallback_strip"
            return cleaned, False, metadata

    except Exception as e:
        logger.warning(f"Extraction failed: {e}, falling back to strip approach")
        cleaned = strip_scripts_and_styles(html)
        metadata["extractor"] = "fallback_strip"
        metadata["error"] = str(e)
        return cleaned, False, metadata


def html_to_markdown_fast(html: str) -> tuple[str, bool, dict]:
    """Convert HTML to Markdown using rule-based markdownify.

    This is the fast path (~50-100ms) that doesn't require LLM inference.
    """
    metadata = {"converter": "markdownify", "conversion_success": False}

    try:
        # Note: markdownify doesn't allow both strip and convert, so we use convert only
        # Scripts and styles should already be stripped by trafilatura/preprocessing
        markdown = md(
            html,
            heading_style="ATX",
            bullets="-",
            code_language="",
            convert=[
                "a",
                "b",
                "blockquote",
                "br",
                "code",
                "em",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "hr",
                "i",
                "img",
                "li",
                "ol",
                "p",
                "pre",
                "strong",
                "table",
                "tbody",
                "td",
                "th",
                "thead",
                "tr",
                "ul",
            ],
        )

        if markdown:
            markdown = re.sub(r"\n{3,}", "\n\n", markdown)
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
    output_format: str = "html",
    options: Optional[ExtractionOptions] = None,
) -> tuple[str, dict]:
    """Preprocess HTML for conversion (extraction step only)."""
    metadata = {
        "original_size_bytes": len(html.encode("utf-8")),
        "trafilatura_used": False,
        "scripts_stripped": False,
    }

    processed = html

    if use_trafilatura:
        extracted, success, extract_meta = extract_with_trafilatura(
            processed, output_format=output_format, options=options
        )
        processed = extracted
        metadata["trafilatura_used"] = success
        metadata["extraction_metadata"] = extract_meta
    else:
        processed = strip_scripts_and_styles(processed)
        metadata["scripts_stripped"] = True

    metadata["final_size_bytes"] = len(
        processed.encode("utf-8") if isinstance(processed, str) else processed
    )

    if metadata["original_size_bytes"] > 0:
        compression = (
            (metadata["original_size_bytes"] - metadata["final_size_bytes"])
            / metadata["original_size_bytes"]
            * 100
        )
        metadata["compression_percent"] = round(compression, 1)
    else:
        metadata["compression_percent"] = 0.0

    if metadata["compression_percent"] > 10:
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
    query_top_k: int = 10,
    options: Optional[ExtractionOptions] = None,
) -> tuple[str, bool, dict]:
    """Full fast-path: Extract content with Trafilatura + convert to Markdown with markdownify.

    This is the recommended default path - no LLM needed, ~100-200ms total.
    """
    metadata = {
        "pipeline": "fast_path",
        "extraction": {},
        "conversion": {},
        "bm25_filter": None,
        "content_valid": False,
    }

    # Step 1: Extract main content with Trafilatura (with ensemble fallback)
    extracted_html, extract_success, extract_meta = extract_with_trafilatura(
        html, include_links=include_links, output_format="html", options=options
    )
    metadata["extraction"] = extract_meta

    if not extracted_html:
        return "", False, metadata

    # Step 2: Convert to Markdown
    markdown, convert_success, convert_meta = html_to_markdown_fast(extracted_html)
    metadata["conversion"] = convert_meta

    # Step 3: Optional BM25 query filtering
    if query and convert_success:
        markdown, bm25_meta = filter_content_by_query(markdown, query, top_k=query_top_k)
        metadata["bm25_filter"] = bm25_meta

    # Step 4: Validate final content
    content_valid = is_valid_content(markdown)
    metadata["content_valid"] = content_valid

    overall_success = extract_success and convert_success and content_valid

    if not content_valid and convert_success:
        logger.warning(
            f"Content validation failed: extracted {len(markdown) if markdown else 0} chars "
            f"(minimum: {MIN_CONTENT_LENGTH})"
        )

    if overall_success:
        logger.info(f"Fast path complete: {len(html)} bytes HTML → {len(markdown)} chars Markdown")

    return markdown, overall_success, metadata


def process_html_full(
    html: str,
    output_format: str = "markdown",
    include_metadata: bool = False,
    include_links: bool = False,
    query: Optional[str] = None,
    query_top_k: int = 10,
    base_url: Optional[str] = None,
    options: Optional[ExtractionOptions] = None,
) -> dict:
    """Full processing pipeline with all features."""
    result = {
        "content": "",
        "success": False,
        "metadata": {},
        "page_metadata": None,
        "links": None,
    }

    if include_metadata:
        page_meta = extract_metadata(html)
        result["page_metadata"] = page_meta.to_dict()

    if include_links:
        links = extract_links(html, base_url)
        result["links"] = [link.to_dict() for link in links]

    if output_format == "text":
        content, success, meta = extract_with_trafilatura(
            html,
            output_format="txt",
            include_links=False,
            include_images=False,
            options=options,
        )
        result["content"] = content or ""
        result["success"] = success
        result["metadata"] = meta

    elif output_format == "html":
        content, success, meta = extract_with_trafilatura(
            html, output_format="html", options=options
        )
        result["content"] = content or ""
        result["success"] = success
        result["metadata"] = meta

    else:  # markdown (default)
        content, success, meta = extract_and_convert_to_markdown(
            html, query=query, query_top_k=query_top_k, options=options
        )
        result["content"] = content
        result["success"] = success
        result["metadata"] = meta

    # Apply BM25 filter for non-markdown formats if query provided
    if query and output_format != "markdown" and result["content"]:
        filtered, bm25_meta = filter_content_by_query(result["content"], query, top_k=query_top_k)
        result["content"] = filtered
        result["metadata"]["bm25_filter"] = bm25_meta

    return result
