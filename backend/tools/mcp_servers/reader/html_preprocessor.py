"""ABOUTME: HTML preprocessing utilities for content extraction and cleanup.

Uses ReadabiliPy (pure Python mode) to extract main article content,
and provides utilities to strip unnecessary HTML elements like scripts and styles.
"""

import logging
import re
from typing import Optional

try:
    from readabilipy import simple_json_from_html_string
except ImportError:
    simple_json_from_html_string = None

logger = logging.getLogger(__name__)


def strip_scripts_and_styles(html: str) -> str:
    """Remove script and style tags from HTML.

    Args:
        html: HTML content to clean

    Returns:
        HTML with script and style tags removed
    """
    # Remove script tags and content
    html = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', html, flags=re.IGNORECASE)

    # Remove style tags and content
    html = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '', html, flags=re.IGNORECASE)

    # Remove HTML comments
    html = re.sub(r'<!--[\s\S]*?-->', '', html)

    return html


def extract_with_readability(html: str, use_readability: bool = True) -> tuple[str, bool]:
    """Extract main article content from HTML using ReadabiliPy.

    Args:
        html: Raw HTML content
        use_readability: Whether to use Readability extraction (default True)

    Returns:
        Tuple of (processed_html, was_extracted) where was_extracted indicates
        if Readability successfully extracted content
    """
    if not use_readability or simple_json_from_html_string is None:
        # Just strip scripts/styles and return original
        cleaned = strip_scripts_and_styles(html)
        return cleaned, False

    try:
        # Use ReadabiliPy to extract main content
        article_json = simple_json_from_html_string(html, use_readability=True)

        if article_json and article_json.get('content'):
            extracted_html = article_json['content']
            logger.info(
                f"ReadabiliPy extracted content: {len(html)} → {len(extracted_html)} bytes"
            )
            return extracted_html, True
        else:
            # Extraction failed, fall back to stripping scripts/styles
            logger.debug("ReadabiliPy extraction returned empty content, falling back")
            cleaned = strip_scripts_and_styles(html)
            return cleaned, False

    except Exception as e:
        logger.warning(f"ReadabiliPy extraction failed: {e}, falling back to strip approach")
        # If ReadabiliPy fails for any reason, just strip scripts/styles
        cleaned = strip_scripts_and_styles(html)
        return cleaned, False


def preprocess_html(
    html: str,
    use_readability: bool = True,
    strip_scripts: bool = True
) -> tuple[str, dict]:
    """Preprocess HTML for ReaderLM inference.

    Applies a pipeline of cleaning steps:
    1. Extract main content with ReadabiliPy (if enabled)
    2. Strip script/style tags (if enabled)

    Args:
        html: Raw HTML content
        use_readability: Whether to use Readability content extraction (default True)
        strip_scripts: Whether to strip script/style tags (default True)

    Returns:
        Tuple of (processed_html, metadata) where metadata contains info about preprocessing
    """
    metadata = {
        'original_size_bytes': len(html.encode('utf-8')),
        'readability_used': False,
        'scripts_stripped': False
    }

    processed_html = html

    # Step 1: Extract main content with Readability
    if use_readability:
        processed_html, extracted = extract_with_readability(processed_html, use_readability=True)
        metadata['readability_used'] = extracted

    # Step 2: Strip scripts and styles
    if strip_scripts and not metadata['readability_used']:
        # Only strip scripts if Readability wasn't used
        # (Readability already removes most unnecessary elements)
        processed_html = strip_scripts_and_styles(processed_html)
        metadata['scripts_stripped'] = True

    metadata['final_size_bytes'] = len(processed_html.encode('utf-8'))
    compression = (
        (metadata['original_size_bytes'] - metadata['final_size_bytes']) /
        metadata['original_size_bytes'] * 100
    )
    metadata['compression_percent'] = round(compression, 1)

    if compression > 10:
        logger.info(
            f"HTML preprocessed: {metadata['original_size_bytes']} → "
            f"{metadata['final_size_bytes']} bytes ({compression:.1f}% reduction)"
        )

    return processed_html, metadata
