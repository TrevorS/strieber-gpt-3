# MCP (Model Context Protocol) Server Setup for Strieber-GPT-3

This document describes the Model Context Protocol (MCP) servers integrated into strieber-gpt-3 and how to configure them with Open WebUI.

## Overview

Strieber-GPT-3 includes five MCP servers that provide extended capabilities to the LLM:

1. **Weather Service** (port 8100) - Current weather and forecasts via Open-Meteo
2. **Web Search** (port 8102) - Web search and news via Brave Search API
3. **Web Reader** (port 8104) - Privacy-first web content extraction via Playwright + ReaderLM-v2
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
│  ├── mcp-reader (localhost:8104) - Privacy-first web reader     │
│  │   ├── playwright-scraper (localhost, internal)               │
│  │   └── llama-server-readerlm (for ReaderLM-v2 extraction)    │
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
docker exec strieber-mcp-reader curl http://localhost:8000/health
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

#### Web Reader (Privacy-First)
- **Type**: MCP (Streamable HTTP)
- **Server URL**: `http://mcp-reader:8000`
- **Features**: Full privacy - no external API calls, complete local processing
- **Backend**: Playwright (web scraping) + ReaderLM-v2 (content extraction)

#### Code Interpreter
- **Type**: MCP (Streamable HTTP)
- **Server URL**: `http://mcp-code-interpreter:8000`

### Step 3: Environment Variables

Set the following in `.env` for API-based services:

```bash
# Required for Web Search
BRAVE_API_KEY=your_brave_search_api_key

# MCP server ports (if running on different host)
LLAMA_HOST=0.0.0.0
OPENWEBUI_PORT=3000
```

Note: The Web Reader service is fully local - no API keys or external services required.

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

### Web Content Extraction (`fetch_page`)

Extract clean, structured content from web pages with full privacy (local processing only).

**Parameters:**
- `url` (required): URL to fetch (HTTP/HTTPS)
- `prompt` (optional): Extraction instruction for targeted content extraction
- `timeout` (optional): Maximum fetch time in seconds (default: 30)
- `force_js_rendering` (optional): Force JavaScript rendering for SPAs (default: false)

**Example:**
```
Extract content from article
fetch_page(url="https://example.com/article")

Extract specific content with instruction
fetch_page(url="https://example.com/article", prompt="Extract main article text and author")

Handle JavaScript-heavy sites
fetch_page(url="https://twitter.com/user/status/123", force_js_rendering=true)
```

**Privacy Note:** All processing happens locally. URLs and content never leave your infrastructure.

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

### mcp-reader (Privacy-First Web Reader)
- **Image**: `strieber-mcp-reader:latest`
- **Container**: `strieber-mcp-reader`
- **Port**: `8104:8000`
- **Health Check**: `curl -f http://localhost:8000/health`
- **Dependencies**: `playwright-scraper`, `llama-server-readerlm`
- **Environment**: None required (fully local operation)
- **Network**: `strieber-net`
- **Features**: No external API calls, no API keys needed, unlimited usage

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

The MCP server codebase is organized for maintainability and code reuse:

```
strieber-gpt-3/
├── backend/
│   └── tools/
│       ├── mcp_servers/
│       │   ├── common/                      # Shared infrastructure
│       │   │   ├── __init__.py
│       │   │   ├── mcp_base.py              # MCPServerBase class (health check, logging)
│       │   │   └── search/                  # Search-specific utilities
│       │   │       ├── __init__.py
│       │   │       ├── backend.py           # SearchBackend interface
│       │   │       ├── brave.py             # BraveSearchBackend implementation
│       │   │       ├── utils.py             # Search filtering & formatting
│       │   │       └── factory.py           # Backend factory pattern
│       │   ├── tests/                       # Test suite with pytest
│       │   │   ├── conftest.py              # Pytest fixtures & mocks
│       │   │   ├── test_search_backend.py   # Backend factory & data class tests
│       │   │   ├── test_search_utils.py     # Utility function tests
│       │   │   └── README.md                # Testing documentation
│       │   ├── weather.py                   # Weather forecast MCP server
│       │   ├── web_search.py                # Web search MCP server (uses factory)
│       │   ├── code_interpreter.py          # Sandboxed code execution server
│       │   ├── reader/
│       │   │   └── server.py                # Privacy-first web reader (local extraction)
│       │   ├── requirements.txt             # Dependencies (includes pytest for dev)
│       │   └── Dockerfile.mcp-server        # Parameterized Dockerfile (builds all servers)
│       └── code-executor/
│           ├── Dockerfile                   # Code executor container
│           └── requirements.txt             # Python dependencies (numpy, pandas, etc)
├── compose.yml                              # Docker Compose configuration
└── MCP-SETUP.md                             # This file
```

### Architecture Highlights

- **common/mcp_base.py**: Eliminates boilerplate by providing MCPServerBase class with:
  - Standard FastMCP server initialization
  - Health check endpoint (`/health`)
  - Consistent logging setup

- **common/search/**: Pluggable search backend infrastructure:
  - `backend.py`: Abstract interface defining what all backends must implement
  - `brave.py`: Brave Search API implementation
  - `factory.py`: Factory pattern for runtime backend selection
  - `utils.py`: Shared filtering, formatting, deduplication utilities

- **tests/**: Comprehensive test suite:
  - Fixtures for mocking external APIs (no real API keys needed)
  - Tests for data validation, filtering, formatting
  - Backend factory and interface tests
  - Documentation for adding new tests

- **MCP Servers**: All 4 servers inherit patterns from common:
  - Use MCPServerBase for consistent initialization
  - Import shared utilities from common/
  - No duplicated boilerplate code

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

**Web Reader Not Working:**
- Check playwright-scraper health: `docker logs strieber-playwright-scraper`
- Check llama-server-readerlm health: `docker logs strieber-llama-server-readerlm`
- Ensure both dependencies are healthy before using reader

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

### Test Web Reader

```bash
docker exec strieber-mcp-reader curl -s http://localhost:8000/health
# Expected: {"status":"ok"}
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
- [Open-Meteo Weather API](https://open-meteo.com/)
- [ReaderLM Model](https://huggingface.co/jina-ai/ReaderLM-v2)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review container logs: `docker logs <container_name>`
3. Verify environment variables: `docker exec <container> env | grep -E "(KEY|URL|PORT)"`
4. Test connectivity: `docker exec <container> curl http://localhost:8000/health`
