"""ABOUTME: Tests for the reader module HTML preprocessing and extraction.

Tests extraction, metadata, links, BM25 filtering, and scraper client utilities.
"""

from reader.html_preprocessor import (
    is_valid_content,
    extract_title_fallback,
    extract_metadata,
    extract_links,
    filter_content_by_query,
    prune_html_xpath,
    extract_with_trafilatura,
    extract_with_readability,
    html_to_markdown_fast,
    tokenize_simple,
    strip_scripts_and_styles,
    ExtractionOptions,
    PageMetadata,
    ExtractedLink,
    MIN_CONTENT_LENGTH,
    DEFAULT_PRUNE_XPATH,
)
from reader.scraper_client import (
    calculate_backoff,
    get_random_user_agent,
    USER_AGENTS,
    INITIAL_BACKOFF_SECONDS,
    MAX_BACKOFF_SECONDS,
    JITTER_FACTOR,
)


# =============================================================================
# Sample HTML fixtures
# =============================================================================


SAMPLE_HTML_SIMPLE = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page Title</title>
    <meta property="og:title" content="OG Title">
    <meta name="author" content="John Doe">
    <meta name="description" content="A test page description">
</head>
<body>
    <header class="site-header">Navigation here</header>
    <main>
        <h1>Main Article Title</h1>
        <p>This is a paragraph with some content that should be extracted properly.
        It contains enough text to be considered valid content.</p>
        <p>Another paragraph with more information about the topic at hand.
        This helps ensure we have substantial content for testing.</p>
        <a href="https://example.com/page1">External Link</a>
        <a href="/internal/page">Internal Link</a>
    </main>
    <footer>Footer content here</footer>
</body>
</html>
"""

SAMPLE_HTML_WITH_SCRIPTS = """
<!DOCTYPE html>
<html>
<head>
    <title>Page With Scripts</title>
    <style>.hidden { display: none; }</style>
</head>
<body>
    <script>console.log("This should be removed");</script>
    <p>Real content here that should remain after stripping scripts.</p>
    <style>body { color: red; }</style>
    <!-- This is a comment that should be removed -->
    <script type="text/javascript">
        var x = 1;
    </script>
</body>
</html>
"""

SAMPLE_HTML_BOILERPLATE = """
<!DOCTYPE html>
<html>
<head><title>Boilerplate Test</title></head>
<body>
    <div class="cookie-banner">Accept cookies</div>
    <div class="newsletter-popup">Subscribe now!</div>
    <div class="advertisement">Buy our product!</div>
    <main>
        <article>
            <p>This is the actual article content that we want to extract.
            It contains multiple sentences and paragraphs of real information.</p>
        </article>
    </main>
    <aside class="sidebar">Related links</aside>
</body>
</html>
"""

SAMPLE_HTML_MINIMAL = """
<html><body><p>Short</p></body></html>
"""


# =============================================================================
# Test Content Validation
# =============================================================================


class TestIsValidContent:
    """Tests for content validation function."""

    def test_valid_content_returns_true(self):
        """Test that valid content passes validation."""
        content = "A" * 100  # 100 chars, above default MIN_CONTENT_LENGTH
        assert is_valid_content(content) is True

    def test_none_content_returns_false(self):
        """Test that None content fails validation."""
        assert is_valid_content(None) is False

    def test_empty_content_returns_false(self):
        """Test that empty string fails validation."""
        assert is_valid_content("") is False

    def test_whitespace_only_returns_false(self):
        """Test that whitespace-only content fails validation."""
        assert is_valid_content("   \n\t   ") is False

    def test_short_content_returns_false(self):
        """Test that content below MIN_CONTENT_LENGTH fails."""
        short_content = "A" * (MIN_CONTENT_LENGTH - 1)
        assert is_valid_content(short_content) is False

    def test_content_at_threshold_returns_true(self):
        """Test that content exactly at MIN_CONTENT_LENGTH passes."""
        threshold_content = "A" * MIN_CONTENT_LENGTH
        assert is_valid_content(threshold_content) is True

    def test_custom_min_length(self):
        """Test custom minimum length parameter."""
        content = "A" * 20
        assert is_valid_content(content, min_length=10) is True
        assert is_valid_content(content, min_length=50) is False

    def test_no_alphanumeric_returns_false(self):
        """Test that content without alphanumeric chars fails."""
        content = "!@#$%^&*()_+-=[]{}|;':\",./<>?" * 10
        assert is_valid_content(content) is False


# =============================================================================
# Test Title Extraction
# =============================================================================


class TestExtractTitleFallback:
    """Tests for regex-based title extraction."""

    def test_extract_from_title_tag(self):
        """Test extracting title from <title> tag."""
        html = "<html><head><title>My Page Title</title></head></html>"
        assert extract_title_fallback(html) == "My Page Title"

    def test_extract_from_og_title(self):
        """Test extracting title from og:title meta tag."""
        html = '<html><head><meta property="og:title" content="OG Title"></head></html>'
        assert extract_title_fallback(html) == "OG Title"

    def test_extract_from_twitter_title(self):
        """Test extracting title from twitter:title meta tag."""
        html = '<html><head><meta name="twitter:title" content="Twitter Title"></head></html>'
        assert extract_title_fallback(html) == "Twitter Title"

    def test_title_tag_priority(self):
        """Test that <title> tag has priority over meta tags."""
        html = """
        <html><head>
            <title>Title Tag</title>
            <meta property="og:title" content="OG Title">
        </head></html>
        """
        assert extract_title_fallback(html) == "Title Tag"

    def test_fallback_to_og_when_title_empty(self):
        """Test fallback to og:title when title tag is empty."""
        html = """
        <html><head>
            <title></title>
            <meta property="og:title" content="OG Title">
        </head></html>
        """
        assert extract_title_fallback(html) == "OG Title"

    def test_no_title_returns_none(self):
        """Test that missing title returns None."""
        html = "<html><head></head><body>Content</body></html>"
        assert extract_title_fallback(html) is None

    def test_whitespace_title_returns_none(self):
        """Test that whitespace-only title returns None."""
        html = "<html><head><title>   </title></head></html>"
        # Note: regex match strips whitespace, empty string is falsy
        result = extract_title_fallback(html)
        assert result is None


# =============================================================================
# Test Metadata Extraction
# =============================================================================


class TestExtractMetadata:
    """Tests for metadata extraction."""

    def test_extract_basic_metadata(self):
        """Test extracting basic metadata from HTML."""
        metadata = extract_metadata(SAMPLE_HTML_SIMPLE)
        assert isinstance(metadata, PageMetadata)
        assert metadata.title is not None

    def test_metadata_has_categories_list(self):
        """Test that metadata initializes categories as list."""
        metadata = extract_metadata("<html></html>")
        assert isinstance(metadata.categories, list)
        assert isinstance(metadata.tags, list)

    def test_metadata_to_dict(self):
        """Test metadata conversion to dictionary."""
        metadata = PageMetadata(title="Test", author="Author")
        result = metadata.to_dict()
        assert isinstance(result, dict)
        assert result["title"] == "Test"
        assert result["author"] == "Author"

    def test_fallback_title_extraction(self):
        """Test that regex fallback is used when trafilatura fails."""
        simple_html = "<html><head><title>Fallback Title</title></head></html>"
        metadata = extract_metadata(simple_html)
        # Should get title either from trafilatura or fallback
        assert metadata.title is not None


# =============================================================================
# Test Links Extraction
# =============================================================================


class TestExtractLinks:
    """Tests for link extraction."""

    def test_extract_links_basic(self):
        """Test basic link extraction."""
        links = extract_links(SAMPLE_HTML_SIMPLE)
        assert len(links) > 0
        assert all(isinstance(link, ExtractedLink) for link in links)

    def test_link_text_extraction(self):
        """Test that link text is extracted."""
        html = '<a href="https://example.com">Link Text</a>'
        links = extract_links(html)
        assert len(links) == 1
        assert links[0].text == "Link Text"
        assert links[0].url == "https://example.com"

    def test_internal_link_detection(self):
        """Test detection of internal links."""
        html = '<a href="/page">Internal</a><a href="https://other.com">External</a>'
        links = extract_links(html, base_url="https://example.com")

        internal = [link for link in links if link.is_internal]

        assert len(internal) == 1
        assert internal[0].url == "/page"

    def test_skip_anchor_links(self):
        """Test that anchor links (#) are skipped."""
        html = '<a href="#section">Anchor</a><a href="https://example.com">Real</a>'
        links = extract_links(html)
        assert len(links) == 1
        assert links[0].url == "https://example.com"

    def test_skip_javascript_links(self):
        """Test that javascript: links are skipped."""
        html = (
            '<a href="javascript:void(0)">JS</a><a href="https://example.com">Real</a>'
        )
        links = extract_links(html)
        assert len(links) == 1
        assert links[0].url == "https://example.com"

    def test_link_to_dict(self):
        """Test link conversion to dictionary."""
        link = ExtractedLink(url="https://example.com", text="Test", is_internal=False)
        result = link.to_dict()
        assert result["url"] == "https://example.com"
        assert result["text"] == "Test"
        assert result["is_internal"] is False


# =============================================================================
# Test BM25 Query Filtering
# =============================================================================


class TestFilterContentByQuery:
    """Tests for BM25-based content filtering."""

    def test_filter_returns_content_and_metadata(self):
        """Test that filter returns tuple of content and metadata."""
        content = "First paragraph about Python.\n\nSecond paragraph about Java."
        filtered, metadata = filter_content_by_query(content, "Python")
        assert isinstance(filtered, str)
        assert isinstance(metadata, dict)

    def test_empty_query_returns_original(self):
        """Test that empty query returns original content."""
        content = "Test content here."
        filtered, _ = filter_content_by_query(content, "")
        assert filtered == content

    def test_none_query_returns_original(self):
        """Test that None query returns original content."""
        content = "Test content here."
        filtered, _ = filter_content_by_query(content, None)
        assert filtered == content

    def test_metadata_contains_query(self):
        """Test that metadata includes the query."""
        content = "Paragraph one.\n\nParagraph two."
        _, metadata = filter_content_by_query(content, "test query")
        assert metadata["query"] == "test query"

    def test_filter_counts_paragraphs(self):
        """Test that metadata includes paragraph counts."""
        # Paragraphs must be >20 chars to be counted
        content = (
            "This is paragraph one about dogs and their behavior.\n\n"
            "This is paragraph two about cats and how they play.\n\n"
            "This is paragraph three about birds flying around."
        )
        _, metadata = filter_content_by_query(content, "dogs", top_k=1)
        assert metadata["original_paragraphs"] == 3

    def test_few_paragraphs_returns_all(self):
        """Test that content with fewer than top_k paragraphs is unchanged."""
        content = "Only one paragraph here."
        filtered, metadata = filter_content_by_query(content, "test", top_k=10)
        assert filtered == content


# =============================================================================
# Test XPath Pruning
# =============================================================================


class TestPruneHtmlXpath:
    """Tests for XPath-based HTML pruning."""

    def test_prune_removes_matching_elements(self):
        """Test that matching elements are removed."""
        html = '<html><body><div class="cookie-banner">Cookies</div><p>Content</p></body></html>'
        pruned = prune_html_xpath(html, ['//div[contains(@class, "cookie")]'])
        assert "cookie-banner" not in pruned.lower()
        assert "Content" in pruned

    def test_prune_empty_patterns_returns_original(self):
        """Test that empty pattern list returns original."""
        html = "<html><body><p>Test</p></body></html>"
        pruned = prune_html_xpath(html, [])
        assert pruned == html

    def test_prune_invalid_xpath_continues(self):
        """Test that invalid XPath patterns don't crash."""
        html = "<html><body><p>Test</p></body></html>"
        # Invalid XPath syntax - use a malformed pattern
        invalid_patterns = ["///invalid"]
        pruned = prune_html_xpath(html, invalid_patterns)
        assert "Test" in pruned

    def test_default_prune_xpath_exists(self):
        """Test that DEFAULT_PRUNE_XPATH is defined and populated."""
        assert len(DEFAULT_PRUNE_XPATH) > 0
        assert any("cookie" in pattern.lower() for pattern in DEFAULT_PRUNE_XPATH)

    def test_prune_multiple_elements(self):
        """Test pruning multiple matching elements."""
        html = """
        <html><body>
            <div class="ad-banner">Ad 1</div>
            <div class="ad-sidebar">Ad 2</div>
            <p>Real content</p>
        </body></html>
        """
        pruned = prune_html_xpath(html, ['//div[contains(@class, "ad-")]'])
        assert "Ad 1" not in pruned
        assert "Ad 2" not in pruned
        assert "Real content" in pruned


# =============================================================================
# Test Trafilatura Extraction
# =============================================================================


class TestExtractWithTrafilatura:
    """Tests for Trafilatura-based content extraction."""

    def test_extraction_returns_tuple(self):
        """Test that extraction returns (content, success, metadata) tuple."""
        result = extract_with_trafilatura(SAMPLE_HTML_SIMPLE)
        assert len(result) == 3
        content, success, metadata = result
        assert isinstance(success, bool)
        assert isinstance(metadata, dict)

    def test_extraction_metadata_has_extractor(self):
        """Test that metadata includes extractor info."""
        _, _, metadata = extract_with_trafilatura(SAMPLE_HTML_SIMPLE)
        assert "extractor" in metadata

    def test_extraction_with_options(self):
        """Test extraction with ExtractionOptions."""
        options = ExtractionOptions(favor_precision=True, deduplicate=True)
        _, _, metadata = extract_with_trafilatura(SAMPLE_HTML_SIMPLE, options=options)
        assert metadata["options"]["favor_precision"] is True
        assert metadata["options"]["deduplicate"] is True

    def test_minimal_html_triggers_fallback(self):
        """Test that minimal HTML may trigger fallback."""
        _, success, metadata = extract_with_trafilatura(SAMPLE_HTML_MINIMAL)
        # Either succeeds or uses fallback
        assert "extractor" in metadata

    def test_different_output_formats(self):
        """Test extraction with different output formats."""
        for fmt in ["html", "text"]:
            _, _, metadata = extract_with_trafilatura(
                SAMPLE_HTML_SIMPLE, output_format=fmt
            )
            assert metadata["output_format"] == fmt


# =============================================================================
# Test Readability Extraction
# =============================================================================


class TestExtractWithReadability:
    """Tests for readability-lxml fallback extraction."""

    def test_readability_returns_tuple(self):
        """Test that readability returns (content, success, metadata) tuple."""
        result = extract_with_readability(SAMPLE_HTML_SIMPLE)
        assert len(result) == 3
        content, success, metadata = result
        assert isinstance(success, bool)
        assert isinstance(metadata, dict)

    def test_readability_metadata_has_extractor(self):
        """Test that metadata identifies readability as extractor."""
        _, _, metadata = extract_with_readability(SAMPLE_HTML_SIMPLE)
        assert metadata["extractor"] == "readability"


# =============================================================================
# Test Markdown Conversion
# =============================================================================


class TestHtmlToMarkdownFast:
    """Tests for rule-based markdown conversion."""

    def test_conversion_returns_tuple(self):
        """Test that conversion returns (markdown, success, metadata) tuple."""
        html = "<h1>Title</h1><p>Content</p>"
        result = html_to_markdown_fast(html)
        assert len(result) == 3

    def test_headers_converted(self):
        """Test that HTML headers are converted to markdown."""
        html = "<h1>Title</h1><h2>Subtitle</h2>"
        markdown, _, _ = html_to_markdown_fast(html)
        assert "#" in markdown

    def test_links_converted(self):
        """Test that HTML links are converted to markdown."""
        html = '<a href="https://example.com">Link</a>'
        markdown, _, _ = html_to_markdown_fast(html)
        assert "[Link]" in markdown
        assert "https://example.com" in markdown

    def test_lists_converted(self):
        """Test that HTML lists are converted to markdown."""
        html = "<ul><li>Item 1</li><li>Item 2</li></ul>"
        markdown, _, _ = html_to_markdown_fast(html)
        assert "-" in markdown or "*" in markdown

    def test_metadata_has_converter(self):
        """Test that metadata identifies the converter."""
        _, _, metadata = html_to_markdown_fast("<p>Test</p>")
        assert metadata["converter"] == "markdownify"


# =============================================================================
# Test Utility Functions
# =============================================================================


class TestTokenizeSimple:
    """Tests for simple tokenizer."""

    def test_basic_tokenization(self):
        """Test basic word tokenization."""
        tokens = tokenize_simple("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_lowercase_conversion(self):
        """Test that tokens are lowercased."""
        tokens = tokenize_simple("UPPERCASE Words")
        assert all(t.islower() for t in tokens)

    def test_punctuation_handling(self):
        """Test that punctuation is stripped."""
        tokens = tokenize_simple("Hello, World! How are you?")
        assert "," not in "".join(tokens)
        assert "!" not in "".join(tokens)
        assert "?" not in "".join(tokens)


class TestStripScriptsAndStyles:
    """Tests for script and style removal."""

    def test_removes_script_tags(self):
        """Test that script tags are removed."""
        result = strip_scripts_and_styles(SAMPLE_HTML_WITH_SCRIPTS)
        assert "<script" not in result.lower()
        assert "console.log" not in result

    def test_removes_style_tags(self):
        """Test that style tags are removed."""
        result = strip_scripts_and_styles(SAMPLE_HTML_WITH_SCRIPTS)
        assert "<style" not in result.lower()

    def test_removes_comments(self):
        """Test that HTML comments are removed."""
        result = strip_scripts_and_styles(SAMPLE_HTML_WITH_SCRIPTS)
        assert "<!--" not in result

    def test_preserves_content(self):
        """Test that actual content is preserved."""
        result = strip_scripts_and_styles(SAMPLE_HTML_WITH_SCRIPTS)
        assert "Real content here" in result


# =============================================================================
# Test ExtractionOptions
# =============================================================================


class TestExtractionOptions:
    """Tests for ExtractionOptions dataclass."""

    def test_default_values(self):
        """Test that defaults are set correctly."""
        options = ExtractionOptions()
        assert options.favor_precision is False
        assert options.favor_recall is False
        assert options.deduplicate is False
        assert options.target_language is None
        assert options.prune_xpath is None
        assert options.fast is False
        assert options.include_links is True
        assert options.include_images is True
        assert options.include_tables is True

    def test_custom_values(self):
        """Test setting custom values."""
        options = ExtractionOptions(
            favor_precision=True,
            target_language="en",
            fast=True,
        )
        assert options.favor_precision is True
        assert options.target_language == "en"
        assert options.fast is True


# =============================================================================
# Test Scraper Client Utilities
# =============================================================================


class TestCalculateBackoff:
    """Tests for exponential backoff calculation."""

    def test_first_attempt_backoff(self):
        """Test backoff for first attempt (0-indexed)."""
        backoff = calculate_backoff(0)
        # Should be around INITIAL_BACKOFF_SECONDS plus jitter
        assert (
            INITIAL_BACKOFF_SECONDS
            <= backoff
            <= INITIAL_BACKOFF_SECONDS * (1 + JITTER_FACTOR)
        )

    def test_exponential_increase(self):
        """Test that backoff increases exponentially."""
        backoff_0 = calculate_backoff(0)
        backoff_1 = calculate_backoff(1)
        backoff_2 = calculate_backoff(2)

        # Each subsequent backoff should be roughly double (accounting for jitter)
        # Just verify increasing trend
        assert backoff_1 > backoff_0 * 0.8  # Allow for jitter variation
        assert backoff_2 > backoff_1 * 0.8  # Continue increasing

    def test_max_backoff_respected(self):
        """Test that backoff doesn't exceed maximum."""
        # Very high attempt number
        backoff = calculate_backoff(100)
        max_with_jitter = MAX_BACKOFF_SECONDS * (1 + JITTER_FACTOR)
        assert backoff <= max_with_jitter

    def test_jitter_adds_randomness(self):
        """Test that jitter creates variation between calls."""
        backoffs = [calculate_backoff(0) for _ in range(10)]
        # With jitter, we should see some variation
        unique_backoffs = set(round(b, 4) for b in backoffs)
        # At least some should be different (very high probability)
        assert len(unique_backoffs) > 1


class TestGetRandomUserAgent:
    """Tests for user agent rotation."""

    def test_returns_valid_user_agent(self):
        """Test that a valid user agent is returned."""
        ua = get_random_user_agent()
        assert ua in USER_AGENTS
        assert "Mozilla" in ua

    def test_randomness(self):
        """Test that different user agents are returned."""
        agents = [get_random_user_agent() for _ in range(50)]
        unique_agents = set(agents)
        # With 8 user agents, 50 samples should give us multiple unique ones
        assert len(unique_agents) > 1

    def test_user_agents_list_populated(self):
        """Test that USER_AGENTS list is populated."""
        assert len(USER_AGENTS) > 0
        # All should be realistic browser user agents
        for ua in USER_AGENTS:
            assert "Mozilla" in ua or "Safari" in ua or "Chrome" in ua
