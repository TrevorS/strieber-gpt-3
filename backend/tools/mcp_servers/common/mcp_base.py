"""ABOUTME: Base class for MCP servers with common initialization, logging, and health check patterns.

Uses the official MCP SDK (modelcontextprotocol/python-sdk) instead of community FastMCP.
"""

import logging
import os
from mcp.server.fastmcp import FastMCP


def setup_logging(logger_name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure logging for an MCP server.

    Args:
        logger_name: Name of the logger (typically __name__)
        level: Logging level (default: logging.INFO)

    Returns:
        Configured logger instance
    """
    logging.basicConfig(level=level)
    return logging.getLogger(logger_name)


class MCPServerBase:
    """Base class for MCP servers with common patterns.

    Provides:
    - Standard MCP server initialization
    - Consistent logging setup
    - Async request handling

    Note: Health check endpoint is provided by launcher.py
    """

    def __init__(self, server_name: str):
        """Initialize MCP server base.

        Args:
            server_name: Name of the MCP server (e.g., "weather", "web_search")
        """
        self.server_name = server_name
        self.mcp = FastMCP(server_name)
        self.logger = setup_logging(__name__)

    def get_logger(self) -> logging.Logger:
        """Get the logger instance.

        Returns:
            Configured logger for this server
        """
        return self.logger

    def get_mcp(self) -> FastMCP:
        """Get the FastMCP server instance.

        Returns:
            FastMCP server for tool registration
        """
        return self.mcp

    def run(self, transport: str = "streamable-http") -> None:
        """Run the MCP server.

        Note: For Docker deployment, use launcher.py instead which properly
        binds to 0.0.0.0 using Starlette + Uvicorn.

        Args:
            transport: Transport protocol ("streamable-http", "stdio", "sse")
        """
        self.mcp.run(transport=transport)

    def get_streamable_http_app(self):
        """Get the Starlette ASGI app for streamable-http transport.

        This is used by launcher.py for proper Docker networking.

        Returns:
            Starlette ASGI application instance
        """
        return self.mcp.streamable_http_app()
