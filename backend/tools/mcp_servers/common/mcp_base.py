"""ABOUTME: Base class for MCP servers with common initialization, logging, and health check patterns."""

import logging
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse


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
    - Health check endpoint at /health
    - Consistent logging setup
    - Async request handling
    """

    def __init__(self, server_name: str):
        """Initialize MCP server base.

        Args:
            server_name: Name of the MCP server (e.g., "weather", "web_search")
        """
        self.server_name = server_name
        self.mcp = FastMCP(server_name)
        self.logger = setup_logging(__name__)

        # Register health check endpoint
        self._register_health_check()

    def _register_health_check(self) -> None:
        """Register the standard /health endpoint."""
        @self.mcp.custom_route("/health", methods=["GET"])
        async def health_check(request: Request) -> JSONResponse:
            """Health check endpoint for Docker container orchestration.

            Returns:
                JSON response with status="ok" and HTTP 200
            """
            self.logger.info(f"{self.server_name} health check passed")
            return JSONResponse({"status": "ok"})

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
