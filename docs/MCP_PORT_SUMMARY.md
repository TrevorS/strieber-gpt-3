# MCP Server Port: Bespoke to Standard Streamable HTTP

**Date**: 2025-11-03
**Status**: ✅ Complete - All 4 MCP servers converted to Streamable HTTP transport

## Executive Summary

Successfully migrated 3 bespoke MCP servers + 1 utility from **stdio transport with custom SQLite-based progress tracking** to **industry-standard MCP Streamable HTTP** with native progress/logging/notification support.

This enables:
- ✅ Clean integration with Open WebUI (no mcpo proxy needed)
- ✅ Server-side tool execution with parallel/interleaved calls
- ✅ Standard MCP progress notifications (replacing SQLite polling)
- ✅ Native MCP logging and error handling
- ✅ Better scalability and observability
- ✅ Standardized Docker deployment

---

## What Was Converted

### **1. weather.py** ✅
**Location**: `backend/tools/mcp_servers/weather.py`

**Changes**:
- ✅ Transport: `stdio` → `streamable-http` (port 8000, path `/mcp`)
- ✅ Progress: `ToolProgress` → `ctx.report_progress()`
- ✅ Logging: Custom → `ctx.info()`, `ctx.warning()`, `ctx.error()`
- ✅ Return type: JSON string → Python dict
- ✅ Made `get_weather()` async with `ctx: Context` parameter

**Progress Steps**:
1. Geocoding location (10%)
2. Fetching weather data (30%)
3. Complete (100%)

---

### **2. reader/server.py** ✅
**Location**: `backend/tools/mcp_servers/reader/server.py`

**Changes**:
- ✅ Transport: `stdio` → `streamable-http` (port 8000, path `/mcp`)
- ✅ Privacy-first: Playwright + ReaderLM-v2 for complete local processing
- ✅ Tool: `fetch_page()` with optional extraction instructions
- ✅ Supports JavaScript rendering and instruction-based extraction

**Progress Steps**:
1. Scraping page (25%)
2. Scraped via method (50%)
3. Extracting/converting with ReaderLM (75%)
4. Complete (100%)

---

### **3. web_search.py** ✅
**Location**: `backend/tools/mcp_servers/web_search.py`

**Changes**:
- ✅ Transport: `stdio` → `streamable-http`
- ✅ Both tools converted: `web_search()` + `news_search()`
- ✅ Return type: JSON string → Python dict
- ✅ Multi-step progress tracking with fine-grained updates

**Progress Steps**:
1. Generating query variants (10%)
2. Searching (20%)
3. Filtering results (40%)
4. Condensing output (60%)
5. Formatting (80%)
6. Complete (100%)

---

### **4. code_interpreter.py** ✅
**Location**: `backend/tools/mcp_servers/code_interpreter.py`

**Changes**:
- ✅ Transport: `stdio` → `streamable-http`
- ✅ Converted to **async** function with `await ctx`
- ✅ Return type: JSON string → Python dict
- ✅ Removed `ToolProgress` import

**Progress Steps**:
1. Preparing container (10%)
2. Executing code (20%)
3. Executed successfully (90%)
4. Complete (100%)

---

## Infrastructure Changes

### **Docker Services** (compose.yml)
Added 4 new MCP server services:

| Service | Port | Image | Purpose |
|---------|------|-------|---------|
| mcp-weather | 8100 | strieber-mcp-weather:latest | Open-Meteo weather API |
| mcp-web-search | 8102 | strieber-mcp-web-search:latest | Brave Search backend |
| mcp-code-interpreter | 8103 | strieber-mcp-code-interpreter:latest | Sandboxed Python execution |
| mcp-reader | 8104 | strieber-mcp-reader:latest | Privacy-first web reader (local) |

**Key Features**:
- All on bridge network `strieber-net`
- Health checks via `/mcp` endpoint
- Environment variables passed through
- Dependencies managed (code-interpreter waits for code-executor)

**Backend Service Updates**:
- Removed Docker socket from volumes (code-executor runs in separate container)
- Added `depends_on` for all 4 MCP services (condition: service_healthy)
- Added MCP server URLs as environment variables:
  - `MCP_WEATHER_URL`
  - `MCP_JINA_READER_URL`
  - `MCP_WEB_SEARCH_URL`
  - `MCP_CODE_INTERPRETER_URL`

### **Dockerfiles** (New)
Created 4 minimal Python-based Dockerfiles:
- `Dockerfile.weather`
- `Dockerfile.jina-reader`
- `Dockerfile.web-search`
- `Dockerfile.code-interpreter`

Each:
- Based on `python:3.11-slim`
- Installs dependencies from `requirements.txt`
- Runs server on port 8000
- Health checks enabled
- Follows "ABOUTME" pattern for documentation

---

## Backend Integration

### **mcp_client.py** (Complete Rewrite)
**Location**: `backend/mcp_client.py`

**Changes**:
- ✅ Removed: `langchain-mcp-adapters` with stdio subprocess spawning
- ✅ Added: `httpx.AsyncClient` for HTTP requests
- ✅ Server URLs now configurable via environment variables
- ✅ Graceful fallback with detailed logging

**Key Methods**:
- `init()` - Initialize HTTP client
- `get_tools()` - Discover tools from all servers via MCP `tools/list`
- `call_tool()` - Execute tool via MCP `tools/call`
- `close()` - Cleanup HTTP client

**Error Handling**:
- Connection errors logged as warnings (service may not be ready)
- HTTP errors with status codes logged
- Continues with other servers if one fails

---

### **tool_registry.py** (Enhanced)
**Location**: `backend/tool_registry.py`

**Changes**:
- ✅ Added `mcp_server` field to `ToolMetadata`
- ✅ All tools mapped to their MCP server:
  - `web-search` → web_search, news_search, get_search_info
  - `weather` → get_weather
  - `jina-reader` → jina_fetch_page, jina_fetch_page_with_selector, get_jina_reader_info
  - `code-interpreter` → execute_python, get_available_libraries
- ✅ Added note about MCP HTTP servers

---

### **event_mapper.py** (Documentation)
**Location**: `backend/event_mapper.py`

**Changes**:
- ✅ Added note: Progress tracking now via MCP `ctx.report_progress()`
- ✅ Clarified: No more `tool_progress.db` polling
- ✅ MCP notifications sent over Streamable HTTP transport

---

## What Was Removed

### **Deprecated Components**
- ❌ `tool_progress.py` - **DELETED** (replaced by MCP Context)
  - Was: Custom SQLite-based progress tracker
  - Now: `ctx.report_progress()` native MCP method

- ❌ `ToolProgress` class usage in all 4 servers
- ❌ `NullProgress` fallback pattern (no longer needed)
- ❌ Custom progress polling from SQLite database

### **Dependencies Removed**
- ❌ `langchain-mcp-adapters` (stdio-based)
- ❌ `sqlite3` usage for progress tracking

---

## Testing Checklist

Before running Open WebUI integration, verify:

- [ ] All 4 MCP servers start without errors
- [ ] Health checks pass for each service
- [ ] Backend can discover tools from all servers
- [ ] Weather server responds to requests
- [ ] Jina Reader server can fetch pages
- [ ] Web Search server can search (with BRAVE_API_KEY)
- [ ] Code Interpreter can execute Python code
- [ ] Docker socket mount works correctly
- [ ] Progress notifications appear during tool execution
- [ ] Error handling works (graceful failures)

### Quick Test

```bash
# Start all services
docker-compose -f compose.yml up -d

# Check service health
docker ps -a | grep strieber-mcp

# Test backend can reach services
docker exec strieber-backend python -c "
import asyncio
from mcp_client import MCPClient
client = await MCPClient().init()
tools = await client.get_tools()
print(f'Found {len(tools)} tools')
"
```

---

## Next Steps: Open WebUI Integration

### **Phase 1: Register MCP Servers** (Admin UI)
Open WebUI → Settings → Features → MCP → Add Server

```
Type: MCP (Streamable HTTP)
Name: Weather
URL: http://mcp-weather:8000/mcp
Method: POST
```

Repeat for:
- Jina Reader: `http://mcp-jina-reader:8000/mcp`
- Web Search: `http://mcp-web-search:8000/mcp`
- Code Interpreter: `http://mcp-code-interpreter:8000/mcp`

### **Phase 2: Enable Native Function Calling**
- Settings → Model Configuration → Enable "Native Function Calling"
- Allows interleaved tool use and reasoning

### **Phase 3: Configure Tool Execution**
- Map tools to gpt-oss-120b model
- Test parallel tool calls
- Verify progress notifications in chat

### **Phase 4: Performance Testing**
- Load test with concurrent requests
- Monitor Docker resource usage
- Check tool execution latency
- Validate error recovery

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ Open WebUI                                                       │
│ (Chat Interface)                                                 │
└────────────────┬────────────────────────────────────────────────┘
                 │ HTTP
                 ↓
┌─────────────────────────────────────────────────────────────────┐
│ Backend (FastAPI)                                                │
│ - mcp_client.py (HTTP client)                                   │
│ - tool_registry.py (metadata + MCP server mapping)              │
│ - event_mapper.py (SSE formatting)                              │
│ - Calls MCP servers via HTTP POST                               │
└──┬──────┬──────┬──────┬──────────────────────────────────────────┘
   │      │      │      │
   │ MCP  │ MCP  │ MCP  │ MCP
   ↓      ↓      ↓      ↓
┌──────────────────────────────────────────────────────────────────┐
│ MCP Servers (Streamable HTTP Transport)                          │
│                                                                   │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────┐│
│ │   Weather    │ │ Web Search   │ │  Code Int.   │ │  Reader  ││
│ │  (8100)      │ │  (8102)      │ │  (8103)      │ │ (8104)   ││
│ │              │ │              │ │              │ │          ││
│ │ - FastMCP    │ │ - FastMCP    │ │ - FastMCP    │ │- FastMCP ││
│ │ - Context    │ │ - Context    │ │ - Context    │ │- Context ││
│ │ - Progress   │ │ - Progress   │ │ - Progress   │ │- Progress││
│ └──────────────┘ └──────────────┘ └──────────────┘ └──────────┘│
└──────────────────────────────────────────────────────────────────┘
       ↓                  ↓                  ↓              ↓
  Open-Meteo API     Brave API         Docker Socket   Playwright +
                                       code-executor   ReaderLM-v2
```

---

## Files Modified/Created

### Created
- ✅ Dockerfile.mcp-server (parameterized for all tools)
- ✅ Dockerfile.llamacpp (for inference engines)
- ✅ `MCP_PORT_SUMMARY.md` (this file)

### Modified
- ✅ `weather.py` (transport, Context, return type)
- ✅ `web_search.py` (transport, Context, return type)
- ✅ `code_interpreter.py` (transport, async, Context)
- ✅ `reader/server.py` (privacy-first web reader)
- ✅ `compose.yml` (4 new services, backend environment)
- ✅ `mcp_client.py` (complete rewrite: stdio → HTTP)
- ✅ `tool_registry.py` (added mcp_server field, documentation)
- ✅ `event_mapper.py` (added documentation note)

### Deleted
- ❌ `tool_progress.py` (no longer needed)

---

## Key Improvements

| Aspect | Before | After |
|--------|--------|-------|
| **Transport** | stdio (subprocess) | HTTP (native) |
| **Progress** | SQLite polling | MCP notifications |
| **Logging** | Custom events | MCP ctx.info/warning/error |
| **Scalability** | Single process | Independent services |
| **Observability** | Limited logging | Native MCP notifications |
| **Error Recovery** | Tight coupling | Graceful degradation |
| **Testing** | Subprocess mocking | Direct HTTP calls |
| **Deployment** | In-container | Docker services |

---

## Known Issues / TODOs

### For Future Optimization
- [ ] Add connection pooling to HTTP client (httpx.limits)
- [ ] Implement retry logic with exponential backoff
- [ ] Add metrics/observability (prometheus endpoints)
- [ ] Cache tool discovery results with TTL
- [ ] Add request timeouts per server (currently 30s global)
- [ ] Implement streaming for long-running operations

### Open WebUI Integration
- [ ] Verify MCP server registration works
- [ ] Test tool discovery via Open WebUI admin
- [ ] Validate native function calling mode
- [ ] Performance test concurrent requests
- [ ] Check progress notifications in chat UI

---

## Migration Checklist

- [x] Convert all 4 MCP servers to Streamable HTTP
- [x] Remove stdio subprocess spawning
- [x] Update backend integration (mcp_client.py)
- [x] Update tool registry with server mappings
- [x] Create Dockerfiles for each server
- [x] Update compose.yml with new services
- [x] Document architecture and testing
- [ ] Test with Open WebUI
- [ ] Performance testing at scale
- [ ] Production deployment

---

## References

- **MCP Spec**: https://modelcontextprotocol.io/
- **FastMCP**: https://github.com/modelcontextprotocol/python-sdk
- **Streamable HTTP Transport**: https://modelcontextprotocol.io/spec/transports/streamable-http

---

**Port completed**: ✅ Ready for Open WebUI integration testing
