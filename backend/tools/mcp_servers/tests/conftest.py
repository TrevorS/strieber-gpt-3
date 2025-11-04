"""ABOUTME: Pytest configuration and shared fixtures for MCP server tests.

Provides mock objects, fixtures, and utilities for testing MCP servers
with mocked external API calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
import httpx


@pytest.fixture
def mock_httpx_client():
    """Fixture providing a mocked httpx.AsyncClient.

    Returns:
        AsyncMock configured as httpx.AsyncClient
    """
    return AsyncMock(spec=httpx.AsyncClient)


@pytest.fixture
def mock_search_result():
    """Fixture providing sample search result data.

    Returns:
        Dictionary with search result fields
    """
    return {
        "title": "Test Result",
        "url": "https://example.com",
        "snippet": "This is a test snippet with enough content to be meaningful.",
        "date": "2024-01-01",
        "extra_snippets": []
    }


@pytest.fixture
def mock_weather_data():
    """Fixture providing sample weather API response.

    Returns:
        Dictionary with Open-Meteo format weather data
    """
    return {
        "current": {
            "time": "2024-01-01T12:00",
            "temperature_2m": 72.5,
            "apparent_temperature": 70.0,
            "relative_humidity_2m": 65,
            "weather_code": 1,
            "wind_speed_10m": 10.5
        }
    }


@pytest.fixture
def mock_openai_client():
    """Fixture providing a mocked OpenAI async client.

    Returns:
        AsyncMock configured with chat completion mock
    """
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content='["variant 1", "variant 2"]')
                )
            ]
        )
    )
    return mock_client


@pytest.fixture
def mock_docker_client():
    """Fixture providing a mocked Docker client.

    Returns:
        MagicMock configured as docker.Client
    """
    mock_client = MagicMock()
    mock_client.containers.run = MagicMock(return_value="output")
    return mock_client
