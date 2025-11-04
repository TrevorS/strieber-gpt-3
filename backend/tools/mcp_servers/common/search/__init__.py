"""ABOUTME: Search backend infrastructure - interfaces, implementations, and utilities."""

from .backend import SearchBackend, SearchResult, SearchResponse
from .factory import get_search_backend

__all__ = ["SearchBackend", "SearchResult", "SearchResponse", "get_search_backend"]
