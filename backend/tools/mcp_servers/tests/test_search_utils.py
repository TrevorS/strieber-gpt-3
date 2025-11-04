"""ABOUTME: Tests for search utility functions.

Tests filtering, formatting, and result condensing utilities
for the web search tool.
"""

import pytest
from common.search import SearchResult
from common.search.utils import (
    filter_low_quality,
    deduplicate_domains,
    extract_domain,
    apply_quality_filters,
    format_as_markdown,
    deduplicate_by_url,
    condense_results
)


class TestDomainExtraction:
    """Tests for domain extraction utility."""

    def test_extract_domain_simple(self):
        """Test extracting domain from simple URL."""
        domain = extract_domain("https://example.com/path")
        assert domain == "example.com"

    def test_extract_domain_with_www(self):
        """Test domain extraction removes www prefix."""
        domain = extract_domain("https://www.example.com/path")
        assert domain == "example.com"

    def test_extract_domain_with_subdomain(self):
        """Test domain extraction with subdomains."""
        domain = extract_domain("https://blog.example.com/path")
        assert "example.com" in domain

    def test_extract_domain_invalid(self):
        """Test domain extraction with invalid URL."""
        domain = extract_domain("invalid://url")
        assert domain is not None


class TestLowQualityFilter:
    """Tests for low quality result filtering."""

    def test_filter_removes_short_snippets(self):
        """Test that short snippets are filtered."""
        results = [
            SearchResult(
                title="Short",
                url="https://example.com/1",
                snippet="too short"  # Less than 50 chars
            ),
            SearchResult(
                title="Good",
                url="https://example.com/2",
                snippet="A" * 100  # Good length
            )
        ]

        filtered = filter_low_quality(results, min_snippet_length=50)
        assert len(filtered) == 1
        assert filtered[0].title == "Good"

    def test_filter_removes_missing_fields(self):
        """Test that results with missing title/URL are filtered."""
        results = [
            SearchResult(
                title="",  # Missing title
                url="https://example.com",
                snippet="A" * 100
            ),
            SearchResult(
                title="Good",
                url="https://example.com",
                snippet="A" * 100
            )
        ]

        filtered = filter_low_quality(results)
        assert len(filtered) == 1

    def test_filter_removes_empty_snippets(self):
        """Test that empty snippets are filtered."""
        results = [
            SearchResult(
                title="Title",
                url="https://example.com",
                snippet="   "  # Whitespace only
            ),
            SearchResult(
                title="Good",
                url="https://example.com",
                snippet="A" * 100
            )
        ]

        filtered = filter_low_quality(results)
        assert len(filtered) == 1


class TestDomainDeduplication:
    """Tests for domain-based deduplication."""

    def test_deduplicate_domains(self):
        """Test limiting results per domain."""
        results = [
            SearchResult(title="1", url="https://example.com/a", snippet="A" * 100),
            SearchResult(title="2", url="https://example.com/b", snippet="A" * 100),
            SearchResult(title="3", url="https://example.com/c", snippet="A" * 100),
            SearchResult(title="4", url="https://example.com/d", snippet="A" * 100),
            SearchResult(title="5", url="https://other.com/a", snippet="A" * 100),
        ]

        deduped = deduplicate_domains(results, max_per_domain=3)
        # Should keep 3 from example.com and 1 from other.com
        assert len(deduped) == 4

    def test_deduplicate_preserves_order(self):
        """Test that deduplication preserves result order."""
        results = [
            SearchResult(title="1", url="https://example.com/a", snippet="A" * 100),
            SearchResult(title="2", url="https://other.com/a", snippet="A" * 100),
            SearchResult(title="3", url="https://example.com/b", snippet="A" * 100),
        ]

        deduped = deduplicate_domains(results, max_per_domain=2)
        assert deduped[0].title == "1"
        assert deduped[1].title == "2"
        assert deduped[2].title == "3"


class TestURLDeduplication:
    """Tests for URL-based deduplication."""

    def test_deduplicate_by_url(self):
        """Test removing duplicate URLs."""
        results = [
            SearchResult(title="1", url="https://example.com/a", snippet="A" * 100),
            SearchResult(title="2", url="https://example.com/a", snippet="B" * 100),  # Duplicate
            SearchResult(title="3", url="https://example.com/b", snippet="C" * 100),
        ]

        deduped = deduplicate_by_url(results)
        assert len(deduped) == 2
        assert deduped[0].title == "1"
        assert deduped[1].title == "3"


class TestMarkdownFormatting:
    """Tests for markdown formatting."""

    def test_format_as_markdown_simple(self):
        """Test basic markdown formatting."""
        results = [
            SearchResult(
                title="Test Result",
                url="https://example.com",
                snippet="This is a test snippet."
            )
        ]

        md = format_as_markdown(results, "test query")
        assert "Test Result" in md
        assert "https://example.com" in md
        assert "This is a test snippet" in md

    def test_format_as_markdown_no_metadata(self):
        """Test formatting without metadata."""
        results = [
            SearchResult(
                title="Result",
                url="https://example.com",
                snippet="Snippet"
            )
        ]

        md = format_as_markdown(results, "query", include_metadata=False)
        assert "Search Results" not in md

    def test_format_as_markdown_empty_results(self):
        """Test formatting with no results."""
        md = format_as_markdown([], "no results")
        assert "No results found" in md


class TestResultCondensing:
    """Tests for token budget aware result condensing."""

    def test_condense_results_within_budget(self):
        """Test that results within budget are not modified."""
        results = [
            SearchResult(
                title="Short",
                url="https://example.com",
                snippet="A" * 100
            )
        ]

        condensed = condense_results(results, max_tokens=1000)
        assert len(condensed) == 1
        assert condensed[0].snippet == "A" * 100

    def test_condense_results_over_budget(self):
        """Test that long results are trimmed."""
        results = [
            SearchResult(
                title="Title",
                url="https://example.com",
                snippet="A" * 5000  # Very long snippet
            )
        ]

        condensed = condense_results(results, max_tokens=500)
        assert len(condensed) == 1
        assert len(condensed[0].snippet) < len(results[0].snippet)
        assert "..." in condensed[0].snippet


class TestQualityFilters:
    """Tests for combined quality filtering."""

    def test_apply_quality_filters_combined(self):
        """Test combined filtering and deduplication."""
        results = [
            SearchResult(title="", url="https://example.com", snippet="A" * 100),  # Missing title
            SearchResult(title="1", url="https://example.com/a", snippet="A" * 100),
            SearchResult(title="2", url="https://example.com/b", snippet="A" * 100),
            SearchResult(title="3", url="https://example.com/c", snippet="A" * 100),
            SearchResult(title="4", url="https://example.com/d", snippet="A" * 100),  # Over domain limit
            SearchResult(title="5", url="https://other.com/a", snippet="A" * 100),
            SearchResult(title="6", url="https://another.com/a", snippet="short"),  # Too short
        ]

        filtered = apply_quality_filters(results, min_snippet_length=50, max_per_domain=3)

        # Should remove: empty title, extra from example.com, short snippet
        assert len(filtered) < len(results)
