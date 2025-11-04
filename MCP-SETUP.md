# MCP (Model Context Protocol) Server Setup for Strieber-GPT-3

This document describes the Model Context Protocol (MCP) servers integrated into strieber-gpt-3 and how to configure them with Open WebUI.

## Overview

Strieber-GPT-3 includes four MCP servers that provide extended capabilities to the LLM:

1. **Weather Service** (port 8100) - Current weather and forecasts via Open-Meteo
2. **Web Search** (port 8102) - Web search and news via Brave Search API
3. **Jina Reader** (port 8101) - Web page content extraction and web scraping
4. **Code Interpreter** (port 8103) - Sandboxed Python code execution with visualization support

All servers use **Streamable HTTP** transport for web-friendly, non-blocking communication.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Open WebUI (port 3000)                     │
│                                                                   │
│  Admin Settings → External Tools → Add MCP Server (HTTP)         │
└──────────────┬──────────────────────────────────────────────────┘
               │
       Docker Bridge Network (strieber-net)
               │
┌──────────────┴──────────────────────────────────────────────────┐
│                                                                   │
│  MCP Servers (Streamable HTTP)                                   │
│  ├── mcp-weather (localhost:8100)                                │
│  ├── mcp-web-search (localhost:8102)                             │
│  ├── mcp-jina-reader (localhost:8101)                            │
│  └── mcp-code-interpreter (localhost:8103)                       │
│      └── code-executor (sandboxed Python)                        │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

## Service Status

All MCP services are containerized and automatically started via Docker Compose:

```bash
# Check status
docker compose ps | grep mcp

# Check health
docker exec strieber-mcp-weather curl http://localhost:8000/health
docker exec strieber-mcp-jina-reader curl http://localhost:8000/health
docker exec strieber-mcp-web-search curl http://localhost:8000/health
docker exec strieber-mcp-code-interpreter curl http://localhost:8000/health
```

Expected response: `{"status":"ok"}`

## Configuring Open WebUI

### Step 1: Access Admin Settings

1. Open Open WebUI at `http://localhost:3000`
2. Log in with admin credentials
3. Click the **⚙️ Settings** icon (top right)
4. Select **Admin Settings** (requires admin role)

### Step 2: Add MCP Servers

In Admin Settings, navigate to **External Tools** and add each MCP server:

#### Weather Service
- **Type**: MCP (Streamable HTTP)
- **Server URL**: `http://llama-server:8000/v1`  *(Note: Weather tools will be available through this)*
- Or configure via: `http://mcp-weather:8000` (within Docker network)

#### Web Search
- **Type**: MCP (Streamable HTTP)
- **Server URL**: `http://mcp-web-search:8000`
- **Required Environment**: `BRAVE_API_KEY` (set in `.env`)

#### Jina Reader
- **Type**: MCP (Streamable HTTP)
- **Server URL**: `http://mcp-jina-reader:8000`
- **Optional Environment**: `JINA_API_KEY` (set in `.env` for higher rate limits)

#### Code Interpreter
- **Type**: MCP (Streamable HTTP)
- **Server URL**: `http://mcp-code-interpreter:8000`

### Step 3: Environment Variables

Set the following in `.env` for API-based services:

```bash
# Required
BRAVE_API_KEY=your_brave_search_api_key

# Optional (improves Jina reader rate limits)
JINA_API_KEY=your_jina_api_key

# MCP server ports (if running on different host)
LLAMA_HOST=0.0.0.0
OPENWEBUI_PORT=3000
```

## Available Tools

### Weather (`get_weather`)

Get current weather, daily (24h), or weekly (7-day) forecasts.

**Parameters:**
- `location` (required): Location name (e.g., "Paris", "New York", "Tokyo")
- `forecast_type` (optional): "current", "daily", or "weekly" (default: "current")
- `units` (optional): "celsius" or "fahrenheit" (default: "fahrenheit")

**Example:**
```
Get current weather for Paris in Celsius
get_weather(location="Paris", units="celsius")

Get weekly forecast for New York
get_weather(location="New York", forecast_type="weekly")
```

### Web Search (`search_web`, `search_news`)

Search the web and get news results with query expansion.

**Parameters:**
- `query` (required): Search query
- `count` (optional): Number of results (default: 5)

**Example:**
```
Search for recent AI developments
search_web(query="AI developments 2024", count=10)

Get latest tech news
search_news(query="technology news")
```

### Web Content Extraction (`read_url`)

Extract clean, structured content from web pages.

**Parameters:**
- `url` (required): URL to extract (supports HTTP/HTTPS and PDFs)

**Example:**
```
Extract content from article
read_url(url="https://example.com/article")

Extract PDF content
read_url(url="https://example.com/document.pdf")
```

### Code Interpreter (`execute_code`)

Safely execute Python code in a sandboxed container environment with:
- Standard libraries: numpy, pandas, matplotlib, scipy, sympy
- Isolated execution with resource limits
- Automatic figure capture from matplotlib

**Parameters:**
- `code` (required): Python code to execute
- `return_figures` (optional): Return matplotlib figures as base64 (default: true)

**Example:**
```python
# Plot data and return figure
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Sine Wave")
plt.show()
```

## Docker Compose Services

### mcp-weather
- **Image**: `strieber-mcp-weather:latest`
- **Container**: `strieber-mcp-weather`
- **Port**: `8100:8000`
- **Health Check**: `curl -f http://localhost:8000/health`
- **Dependencies**: None
- **Network**: `strieber-net`

### mcp-web-search
- **Image**: `strieber-mcp-web-search:latest`
- **Container**: `strieber-mcp-web-search`
- **Port**: `8102:8000`
- **Health Check**: `curl -f http://localhost:8000/health`
- **Dependencies**: `llama-server` (for query expansion)
- **Environment**: `BRAVE_API_KEY` (required)
- **Network**: `strieber-net`

### mcp-jina-reader
- **Image**: `strieber-mcp-jina-reader:latest`
- **Container**: `strieber-mcp-jina-reader`
- **Port**: `8101:8000`
- **Health Check**: `curl -f http://localhost:8000/health`
- **Dependencies**: None
- **Environment**: `JINA_API_KEY` (optional)
- **Network**: `strieber-net`

### mcp-code-interpreter
- **Image**: `strieber-mcp-code-interpreter:latest`
- **Container**: `strieber-mcp-code-interpreter`
- **Port**: `8103:8000`
- **Health Check**: `curl -f http://localhost:8000/health`
- **Dependencies**: `code-executor`
- **Volumes**: Docker socket for sandboxed execution
- **Network**: `strieber-net`

### code-executor
- **Image**: `code-executor:latest`
- **Container**: `code-executor`
- **Purpose**: Sandboxed Python execution environment
- **Network**: `strieber-net`

## File Structure

```
strieber-gpt-3/
├── backend/
│   └── tools/
│       ├── mcp_servers/
│       │   ├── weather.py              # Weather forecast tool
│       │   ├── web_search.py           # Web search tool
│       │   ├── jina_reader.py          # Web content extraction
│       │   ├── code_interpreter.py     # Sandboxed code execution
│       │   ├── brave_backend.py        # Brave Search API backend
│       │   ├── search_backend.py       # Abstract search interface
│       │   ├── search_utils.py         # Utility functions
│       │   ├── requirements.txt        # MCP server dependencies
│       │   └── Dockerfile.mcp-server   # Generic MCP server Dockerfile
│       └── code-executor/
│           ├── Dockerfile              # Code executor container
│           └── requirements.txt        # Python dependencies
├── compose.yml                         # Docker Compose configuration
└── MCP-SETUP.md                        # This file
```

## Troubleshooting

### MCP Server Not Responding

**Check service health:**
```bash
docker compose ps | grep mcp
```

All services should show `Healthy` status. If showing `Unhealthy` or `Restarting`:

```bash
# Check logs
docker logs strieber-mcp-weather
docker logs strieber-mcp-web-search
docker logs strieber-mcp-jina-reader
docker logs strieber-mcp-code-interpreter
```

### API Key Issues

**Web Search Not Working:**
- Verify `BRAVE_API_KEY` is set in `.env`
- Restart services: `docker compose restart mcp-web-search`

**Jina Reader Rate Limited:**
- Optionally set `JINA_API_KEY` in `.env` for higher limits
- Restart: `docker compose restart mcp-jina-reader`

### Docker Network Issues

**Services can't communicate:**
```bash
# Verify network exists
docker network ls | grep strieber-net

# Inspect network
docker network inspect strieber-gpt-3_strieber-net
```

### Container Name Conflicts

If you see "container name already in use" errors:
```bash
# Force remove old containers
docker ps -a | grep strieber-mcp | awk '{print $1}' | xargs -r docker rm -f
docker ps -a | grep code-executor | awk '{print $1}' | xargs -r docker rm -f

# Restart
docker compose up -d
```

## Testing Tools Directly

### Test Weather Tool

```bash
docker exec strieber-mcp-weather curl -s http://localhost:8000/health
# Expected: {"status":"ok"}
```

### Test Web Search

```bash
# Requires BRAVE_API_KEY in environment
docker exec strieber-mcp-web-search curl -s http://localhost:8000/health
```

### Test Jina Reader

```bash
docker exec strieber-mcp-jina-reader curl -s http://localhost:8000/health
```

### Test Code Interpreter

```bash
docker exec strieber-mcp-code-interpreter curl -s http://localhost:8000/health
```

## Integration with Open WebUI

Once configured, tools will be available in the LLM context:

1. In chat, the LLM can automatically use tools based on conversation context
2. Tool calls are visible in the chat history
3. Results are integrated seamlessly into responses
4. Code execution results (including plots) are displayed inline

## Advanced Configuration

### Custom Tool Parameters

You can configure tool behavior through environment variables in `compose.yml`:

```yaml
mcp-web-search:
  environment:
    - PORT=8000
    - BRAVE_API_KEY=${BRAVE_API_KEY}
    - LLAMA_BASE_URL=http://llama-server:8000
    - MODEL_NAME=${MODEL_FILE:-gpt-oss-20b}
```

### Resource Limits

Code executor runs with isolated containers:

```yaml
code-executor:
  environment:
    - # Standard Python 3.11 with numpy, pandas, matplotlib, scipy, sympy
```

Memory and CPU limits can be set in Docker compose if needed.

## Related Documentation

- [Open WebUI MCP Integration Docs](https://docs.openwebui.com/features/mcp/)
- [FastMCP Framework](https://github.com/janus-llm/fastmcp)
- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/)
- [Brave Search API](https://api.search.brave.com/)
- [Jina Reader API](https://jina.ai/reader/)
- [Open-Meteo Weather API](https://open-meteo.com/)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review container logs: `docker logs <container_name>`
3. Verify environment variables: `docker exec <container> env | grep -E "(KEY|URL|PORT)"`
4. Test connectivity: `docker exec <container> curl http://localhost:8000/health`
