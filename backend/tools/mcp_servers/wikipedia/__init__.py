"""ABOUTME: Wikipedia MCP Server package for fast world knowledge lookup.

Provides Resources, Tools, and Prompts for Wikipedia access:
- Resources: Pre-loadable context (app-controlled)
- Tools: Dynamic lookup (model-controlled)
- Prompts: Research templates (user-controlled)
"""

from . import server

__all__ = ["server"]
