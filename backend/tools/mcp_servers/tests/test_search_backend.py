"""ABOUTME: Tests for search backend interface and factory.

Tests the SearchBackend interface, data classes, and factory pattern
for backend instantiation.
"""

import pytest
from common.search import SearchBackend, SearchResult, SearchResponse, get_search_backend
from common.search.backend import SearchBackend as DirectBackend


class TestSearchResult:
    """Tests for SearchResult data class."""

    def test_search_result_creation(self, mock_search_result):
        """Test creating a SearchResult instance."""
        result = SearchResult(**mock_search_result)
        assert result.title == "Test Result"
        assert result.url == "https://example.com"
        assert result.snippet == "This is a test snippet with enough content to be meaningful."

    def test_get_all_text_single_snippet(self, mock_search_result):
        """Test get_all_text with only main snippet."""
        result = SearchResult(**mock_search_result)
        text = result.get_all_text()
        assert result.snippet in text

    def test_get_all_text_with_extra_snippets(self):
        """Test get_all_text combines main and extra snippets."""
        result = SearchResult(
            title="Test",
            url="https://example.com",
            snippet="Main snippet",
            extra_snippets=["Extra 1", "Extra 2"]
        )
        text = result.get_all_text()
        assert "Main snippet" in text
        assert "Extra 1" in text
        assert "Extra 2" in text

    def test_estimate_tokens(self):
        """Test token estimation for a result."""
        result = SearchResult(
            title="Test Title",
            url="https://example.com/very/long/path",
            snippet="A" * 400  # ~100 tokens
        )
        tokens = result.estimate_tokens()
        assert tokens > 0
        assert tokens < 200  # Rough estimate


class TestSearchResponse:
    """Tests for SearchResponse data class."""

    def test_search_response_creation(self, mock_search_result):
        """Test creating a SearchResponse."""
        result = SearchResult(**mock_search_result)
        response = SearchResponse(
            query="test query",
            results=[result],
            total_results=1
        )
        assert response.query == "test query"
        assert len(response.results) == 1
        assert response.total_results == 1

    def test_search_response_empty_results(self):
        """Test SearchResponse with no results."""
        response = SearchResponse(
            query="no results",
            results=[],
            total_results=0
        )
        assert response.results == []
        assert response.total_results == 0


class TestSearchBackendFactory:
    """Tests for search backend factory pattern."""

    def test_get_search_backend_brave(self, monkeypatch):
        """Test factory returns Brave backend when requested."""
        # Mock environment to avoid requiring real API key
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")

        backend = get_search_backend("brave")
        assert backend is not None
        assert backend.name == "brave"

    def test_get_search_backend_default(self, monkeypatch):
        """Test factory uses brave as default backend."""
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")

        backend = get_search_backend()
        assert backend.name == "brave"

    def test_get_search_backend_from_env(self, monkeypatch):
        """Test factory reads backend type from environment."""
        monkeypatch.setenv("SEARCH_BACKEND", "brave")
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")

        backend = get_search_backend(None)
        assert backend.name == "brave"

    def test_get_search_backend_unknown_type(self):
        """Test factory raises ValueError for unknown backend."""
        with pytest.raises(ValueError, match="Unknown search backend"):
            get_search_backend("unknown_backend")

    def test_get_search_backend_requires_api_key(self, monkeypatch):
        """Test Brave backend requires API key."""
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)

        with pytest.raises(ValueError, match="BRAVE_API_KEY"):
            get_search_backend("brave")
