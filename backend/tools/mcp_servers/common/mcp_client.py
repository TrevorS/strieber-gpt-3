# ABOUTME: MCP client for testing MCP servers via streamable-http transport.
# ABOUTME: Uses the official MCP SDK for proper session-based communication.

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


class MCPError(Exception):
    """Error from MCP server."""

    pass


@dataclass
class MCPTool:
    """An MCP tool definition."""

    name: str
    description: str = ""
    inputSchema: dict = field(default_factory=dict)


@dataclass
class MCPResult:
    """Result from calling an MCP tool."""

    content: list[dict]
    is_error: bool = False

    def text(self) -> str:
        """Extract text content from result."""
        texts = []
        for item in self.content:
            if hasattr(item, "text"):
                texts.append(item.text)
            elif isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts)

    def __repr__(self) -> str:
        if self.is_error:
            return f"MCPResult(error={self.text()[:100]})"
        return f"MCPResult({len(self.content)} items)"


class MCPClient:
    """Client for making MCP calls via streamable-http transport."""

    # Known server aliases (use docker compose service names for container-to-container)
    SERVERS = {
        "weather": "http://mcp-weather:8000/mcp",
        "web_search": "http://mcp-web-search:8000/mcp",
        "code_interpreter": "http://mcp-code-interpreter:8000/mcp",
        "reader": "http://mcp-reader:8000/mcp",
        "comfy_zimage": "http://mcp-comfy-zimage:8000/mcp",
        "lora_trainer": "http://mcp-lora-trainer:8000/mcp",
    }

    def __init__(self, url_or_alias: str, timeout: float = 60.0):
        """Initialize client with URL or server alias.

        Args:
            url_or_alias: Full URL (http://...) or alias (weather, web_search, etc.)
            timeout: Request timeout in seconds
        """
        if url_or_alias in self.SERVERS:
            self.url = self.SERVERS[url_or_alias]
        elif url_or_alias.startswith("http"):
            self.url = url_or_alias
        else:
            raise ValueError(
                f"Unknown server: {url_or_alias}. "
                f"Use full URL or one of: {', '.join(self.SERVERS.keys())}"
            )
        self.timeout = timeout

    @asynccontextmanager
    async def _session(self):
        """Create an MCP session context."""
        async with streamablehttp_client(
            self.url, timeout=self.timeout, sse_read_timeout=self.timeout
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session

    async def list_tools(self) -> list[MCPTool]:
        """List all tools available on the server."""
        async with self._session() as session:
            result = await session.list_tools()
            tools = []
            for t in result.tools:
                tools.append(
                    MCPTool(
                        name=t.name,
                        description=t.description or "",
                        inputSchema=t.inputSchema if hasattr(t, "inputSchema") else {},
                    )
                )
            return tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> MCPResult:
        """Call a tool on the server.

        Args:
            name: Tool name
            arguments: Tool arguments as a dictionary

        Returns:
            MCPResult with content and error status
        """
        async with self._session() as session:
            result = await session.call_tool(name, arguments or {})
            # Convert content items to dicts for consistent handling
            content = []
            for item in result.content:
                if hasattr(item, "model_dump"):
                    content.append(item.model_dump())
                elif hasattr(item, "dict"):
                    content.append(item.dict())
                else:
                    content.append({"type": "text", "text": str(item)})
            return MCPResult(
                content=content,
                is_error=result.isError if hasattr(result, "isError") else False,
            )

    async def ping(self) -> bool:
        """Check if server is reachable by listing tools."""
        try:
            await self.list_tools()
            return True
        except Exception:
            return False


# Convenience functions for quick testing
async def list_tools(server: str) -> list[MCPTool]:
    """List tools on a server."""
    client = MCPClient(server)
    return await client.list_tools()


async def call_tool(server: str, tool: str, args: dict | None = None) -> MCPResult:
    """Call a tool on a server."""
    client = MCPClient(server)
    return await client.call_tool(tool, args)
