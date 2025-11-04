"""ABOUTME: Backend factory for instantiating search backends.

Provides a clean factory interface for creating search backend instances,
allowing runtime selection of backend implementations.
"""

import logging
import os
from typing import Optional

from .backend import SearchBackend
from .brave import BraveSearchBackend

# Configure logging
logger = logging.getLogger(__name__)


def get_search_backend(backend_type: Optional[str] = None, **kwargs) -> SearchBackend:
    """Create and return a search backend instance.

    Args:
        backend_type: Type of backend to create. If None, reads from SEARCH_BACKEND
                     environment variable. Default: "brave"
        **kwargs: Additional arguments to pass to backend constructor

    Returns:
        SearchBackend instance

    Raises:
        ValueError: If backend_type is unknown or required API key is missing
    """
    # Get backend type from parameter or environment variable
    if backend_type is None:
        backend_type = os.getenv("SEARCH_BACKEND", "brave").lower()

    logger.info(f"Creating search backend: {backend_type}")

    if backend_type == "brave":
        return BraveSearchBackend(**kwargs)

    # Add support for additional backends here in the future
    # elif backend_type == "google":
    #     return GoogleSearchBackend(**kwargs)
    # elif backend_type == "duckduckgo":
    #     return DuckDuckGoSearchBackend(**kwargs)

    raise ValueError(f"Unknown search backend: {backend_type}. Supported: brave")
