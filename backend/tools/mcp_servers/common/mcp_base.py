"""ABOUTME: Base class for MCP servers with common initialization, logging, and health check patterns.

Uses the official MCP SDK (modelcontextprotocol/python-sdk) instead of community FastMCP.
"""

import logging
import os
from typing import Any, Dict, Optional
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent


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

    def create_success_result(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CallToolResult:
        """Create standardized success result.

        Args:
            content: The response text to return
            metadata: Optional metadata dictionary

        Returns:
            CallToolResult with standardized success format

        Examples:
            >>> result = server.create_success_result("Task completed")
            >>> result = server.create_success_result("Found 5 items", {"count": 5})
        """
        text_content = TextContent(type="text", text=content)
        if metadata:
            return CallToolResult(content=[text_content], metadata=metadata)
        return CallToolResult(content=[text_content])

    def log_tool_start(self, tool_name: str, **params) -> None:
        """Log tool invocation with parameters.

        Args:
            tool_name: Name of the tool being invoked
            **params: Keyword arguments to log (will be formatted)

        Examples:
            >>> server.log_tool_start("search", query="python", limit=10)
            >>> server.log_tool_start("fetch", url="https://example.com")
        """
        if params:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            self.logger.info(f"{tool_name} started: {param_str}")
        else:
            self.logger.info(f"{tool_name} started")

    def log_tool_complete(self, tool_name: str, **metrics) -> None:
        """Log tool completion with execution metrics.

        Args:
            tool_name: Name of the tool that completed
            **metrics: Performance/execution metrics to log

        Examples:
            >>> server.log_tool_complete("search", results=42, duration_ms=150)
            >>> server.log_tool_complete("fetch", bytes_received=1024)
        """
        if metrics:
            metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
            self.logger.info(f"{tool_name} completed: {metric_str}")
        else:
            self.logger.info(f"{tool_name} completed")

    def log_tool_error(
        self,
        tool_name: str,
        error_code: str,
        error_message: str,
        **context
    ) -> None:
        """Log tool error with context.

        Args:
            tool_name: Name of the tool that failed
            error_code: Machine-readable error code
            error_message: Human-readable error message
            **context: Additional error context

        Examples:
            >>> server.log_tool_error("search", "TIMEOUT", "Request timed out", timeout=30)
            >>> server.log_tool_error("fetch", "HTTP_404", "Page not found", url="https://example.com")
        """
        context_str = ", ".join(f"{k}={v}" for k, v in context.items()) if context else ""
        if context_str:
            self.logger.error(f"{tool_name} error [{error_code}]: {error_message} ({context_str})")
        else:
            self.logger.error(f"{tool_name} error [{error_code}]: {error_message}")
