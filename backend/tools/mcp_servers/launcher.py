"""ABOUTME: MCP server launcher that properly binds to 0.0.0.0 for Docker networking.

Uses monkeypatching of uvicorn.Config to ensure servers bind to 0.0.0.0
instead of the default 127.0.0.1, enabling inter-container communication.
"""

import logging
import sys
import os

# Setup logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_server(server_module: str, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run an MCP server with proper host binding for Docker networking.

    Args:
        server_module: Name of the MCP server to run (e.g., "weather", "web_search")
        host: Host to bind to (default: 0.0.0.0 for Docker)
        port: Port to bind to (default: 8000)
    """
    logger.info(f"Loading MCP server: {server_module}")

    # Dynamically import the server module and get the FastMCP instance
    try:
        if server_module == "weather":
            from weather import server as weather_server
            mcp_instance = weather_server.get_mcp()
        elif server_module == "web_search":
            from web_search import server as web_search_server
            mcp_instance = web_search_server.get_mcp()
        elif server_module == "jina_reader":
            from jina_reader import server as jina_reader_server
            mcp_instance = jina_reader_server.get_mcp()
        elif server_module == "code_interpreter":
            from code_interpreter import server as code_interpreter_server
            mcp_instance = code_interpreter_server.get_mcp()
        elif server_module == "reader":
            from reader import server as reader_server
            mcp_instance = reader_server.get_mcp()
        else:
            raise ValueError(f"Unknown MCP server module: {server_module}")
    except ImportError as e:
        logger.error(f"Failed to import server module '{server_module}': {e}")
        sys.exit(1)

    logger.info(f"Successfully loaded MCP server: {server_module}")

    # Patch uvicorn.Config to bind to 0.0.0.0 for Docker inter-container networking
    # This must be done BEFORE any uvicorn instances are created
    import uvicorn
    original_init = uvicorn.Config.__init__

    def patched_init(self, *args, **kwargs):
        # Force host to 0.0.0.0 to ensure Docker inter-container networking works
        kwargs["host"] = host
        kwargs.setdefault("port", port)
        logger.info(f"Patched uvicorn.Config - binding to {host}:{kwargs.get('port', port)}")
        return original_init(self, *args, **kwargs)

    uvicorn.Config.__init__ = patched_init

    try:
        logger.info(f"Starting {server_module} MCP server on {host}:{port} (transport: streamable-http)")
        # Call mcp.run() which properly initializes the streamable HTTP transport
        # The patched uvicorn.Config ensures binding to 0.0.0.0
        mcp_instance.run(transport="streamable-http")
    finally:
        # Restore original
        uvicorn.Config.__init__ = original_init


if __name__ == "__main__":
    # Get server module from environment variable
    server_module = os.getenv("SERVER_MODULE", "weather")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    run_server(server_module, host, port)
