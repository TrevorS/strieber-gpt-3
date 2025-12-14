---
name: MCP Tester
description: Test MCP servers interactively. Use when debugging MCP tools, verifying server connectivity, or testing tool inputs/outputs during development.
allowed-tools: Bash, Read
---

# MCP Tester

Test MCP servers via CLI or Python.

## Quick Examples

```bash
# List tools on a server
python backend/tools/mcp_servers/mcp_test.py list weather

# Call a tool with JSON args
python backend/tools/mcp_servers/mcp_test.py call weather get_weather '{"location": "NYC"}'

# Call with key=value args
python backend/tools/mcp_servers/mcp_test.py call weather get_weather location="San Francisco"

# Check if server is up
python backend/tools/mcp_servers/mcp_test.py ping weather
```

## Available MCP Servers

| Alias | Port | Tools |
|-------|------|-------|
| weather | 9100 | get_weather |
| web_search | 9110 | brave_search |
| code_interpreter | 9120 | execute_python |
| reader | 9130 | read_url |
| comfy_zimage | 9141 | zimage_turbo, zimage_controlnet, lora_list_available |
| lora_trainer | 9145 | lora_create_dataset, lora_start_training, etc. |

## CLI Commands

```bash
# List known server aliases
python backend/tools/mcp_servers/mcp_test.py servers

# List tools (verbose for parameter schemas)
python backend/tools/mcp_servers/mcp_test.py list weather -v

# Call with JSON arguments
python backend/tools/mcp_servers/mcp_test.py call weather get_weather '{"location": "NYC"}'

# Call with key=value arguments
python backend/tools/mcp_servers/mcp_test.py call comfy_zimage zimage_turbo prompt="a cat" steps=12

# Extended timeout for slow tools
python backend/tools/mcp_servers/mcp_test.py --timeout 120 call comfy_zimage zimage_turbo '{"prompt": "sunset"}'
```

## Python API

```python
import asyncio
from backend.tools.mcp_servers.common.mcp_client import MCPClient

async def main():
    client = MCPClient("weather")  # or full URL

    # List tools
    tools = await client.list_tools()
    for t in tools:
        print(f"{t.name}: {t.description}")

    # Call a tool
    result = await client.call_tool("get_weather", {"location": "NYC"})
    print(result.text())

asyncio.run(main())
```

## Troubleshooting

1. **Connection refused**: Check container is running with `docker ps | grep mcp-`
2. **Tool not found**: Run `list -v` to see available tools and parameters
3. **Timeout**: Use `--timeout 120` for slow operations (image generation)
4. **Invalid args**: Check tool schema with `list -v`
