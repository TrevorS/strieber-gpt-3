# Local ReaderLM-v2 Setup Guide

This guide covers setting up the local ReaderLM-v2 model for HTML-to-Markdown conversion, replacing the cloud-based Jina Reader API with a self-hosted solution.

## Overview

The local reader setup consists of three components:

```
┌─────────────────────────────────────────────────────────┐
│  mcp-jina-reader (Enhanced MCP Server)                  │
│                                                          │
│  ┌──────────────────┐         ┌─────────────────────┐  │
│  │ Playwright       │────────▶│ ReaderLM-v2 Model   │  │
│  │ HTML Fetcher     │         │ (llama-server:8004) │  │
│  │ (port 8005)      │         │  Q4_K_M (1.12GB)    │  │
│  │                  │         │                     │  │
│  │ - JS rendering   │         └─────────────────────┘  │
│  │ - Resource block │                  │               │
│  │ - Headless       │                  │               │
│  └──────────────────┘                  │               │
│          │                             │               │
│          ▼                             ▼               │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Smart Router:                                   │  │
│  │  - HTML pages? → Local (Playwright + ReaderLM)  │  │
│  │  - PDF files?  → Jina API (fallback)            │  │
│  │  - Local fail? → Jina API (fallback)            │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. ReaderLM-v2 Model (llama-server-reader)

- **Model**: ReaderLM-v2 by Jina AI
- **Size**: 1.5B parameters
- **Quantization**: Q4_K_M (1.12 GB)
- **Task**: HTML → Markdown conversion
- **Context**: 64K tokens (supports up to 512K)
- **Languages**: 29 languages supported

### 2. Playwright HTML Fetcher

- **Browser**: Headless Chromium
- **Purpose**: Fetch JavaScript-rendered HTML
- **Features**:
  - Resource blocking (images, CSS) for speed
  - Custom user agents
  - Configurable wait strategies
  - Timeout control

### 3. Enhanced jina_reader.py MCP Server

- **Primary**: Local processing (Playwright + ReaderLM-v2)
- **Fallback**: Jina Reader API (PDFs, complex pages)
- **Auto-routing**: PDFs → API, HTML → Local

## Setup Instructions

### Step 1: Download the ReaderLM-v2 Model

Download the Q4_K_M quantization (1.12 GB):

```bash
# Option 1: Using wget
cd /home/trevor/models
wget https://huggingface.co/mradermacher/ReaderLM-v2-GGUF/resolve/main/ReaderLM-v2.Q4_K_M.gguf

# Option 2: Using huggingface-cli (if installed)
huggingface-cli download mradermacher/ReaderLM-v2-GGUF \
  --include "ReaderLM-v2.Q4_K_M.gguf" \
  --local-dir /home/trevor/models/ \
  --local-dir-use-symlinks False

# Option 3: Using curl
curl -L -o /home/trevor/models/ReaderLM-v2.Q4_K_M.gguf \
  https://huggingface.co/mradermacher/ReaderLM-v2-GGUF/resolve/main/ReaderLM-v2.Q4_K_M.gguf
```

**Alternative Quantizations** (optional):

- **Q8_0** (1.89 GB): Higher quality, slower - `ReaderLM-v2.Q8_0.gguf`
- **Q5_K_M** (1.29 GB): Balanced - `ReaderLM-v2.Q5_K_M.gguf`

Repository: https://huggingface.co/mradermacher/ReaderLM-v2-GGUF

### Step 2: Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Enable local reader processing
USE_LOCAL_READER=true

# Optional: Jina API key for PDF fallback
# Get key at: https://jina.ai/
# JINA_API_KEY=your-api-key-here
```

### Step 3: Build and Start Services

```bash
# Build all services (including new ones)
docker compose build

# Start all services
docker compose up -d

# Or start only the reader stack
docker compose up -d llama-server-reader playwright-fetcher mcp-jina-reader
```

### Step 4: Verify Services

```bash
# Check service health
docker compose ps

# Check logs
docker logs strieber-llama-server-reader
docker logs strieber-playwright-fetcher
docker logs strieber-mcp-jina-reader

# Test Playwright fetcher
curl "http://localhost:8005/fetch?url=https://example.com"

# Test ReaderLM-v2 model
curl -X POST http://localhost:8004/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "Convert the HTML to Markdown."},
      {"role": "user", "content": "<html><body><h1>Test</h1></body></html>"}
    ],
    "temperature": 0.1
  }'

# Test MCP server
curl http://localhost:8101/health
```

## Usage

### From Open WebUI

The MCP tools are automatically available in Open WebUI under "External Tools":

1. Navigate to **Admin Settings → External Tools**
2. Add/configure the Jina Reader MCP server:
   - **URL**: `http://mcp-jina-reader:8000/mcp`
   - **Name**: `jina-reader`
3. Use in chat: "Fetch the content from https://example.com"

### Direct API Usage

```bash
# Fetch page (will use local processing)
curl -X POST http://localhost:8101/tools/jina_fetch_page \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://news.ycombinator.com",
    "timeout": 10
  }'

# Get reader info
curl -X POST http://localhost:8101/tools/get_jina_reader_info
```

### Python Client Example

```python
import httpx

async def fetch_page(url: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8101/tools/jina_fetch_page",
            json={"url": url, "timeout": 10}
        )
        return response.json()

# Fetch a page
result = await fetch_page("https://example.com")
print(result)
```

## Performance

### Expected Performance on DGX Spark Blackwell

| Component | Metric | Value |
|-----------|--------|-------|
| **ReaderLM-v2** | Model Size | 1.12 GB (Q4_K_M) |
| | VRAM Usage | ~2 GB |
| | Tokens/sec | 80-120 (estimated) |
| | Avg Processing | 1-2 seconds |
| **Playwright** | Page Load | 1-3 seconds (avg) |
| | Resource Blocking | 2-3x speedup |
| **Total** | End-to-End | 2-5 seconds/page |
| | Rate Limit | Unlimited (local) |

### Comparison: Local vs API

| Method | Speed | Cost | Rate Limit | PDF Support |
|--------|-------|------|------------|-------------|
| **Local** | 2-5 sec | Free | Unlimited | No |
| **Jina API** | 1-2 sec | $0.05/1M tokens | 20-500 RPM | Yes |

## Architecture Details

### Port Allocation

- **8004**: llama-server-reader (ReaderLM-v2 model)
- **8005**: playwright-fetcher (HTML fetching)
- **8101**: mcp-jina-reader (MCP server)

### Processing Flow

1. **User requests page**: `jina_fetch_page("https://example.com")`
2. **Router decides**:
   - Is PDF? → Use Jina API
   - Is HTML + local enabled? → Try local
   - Local failed? → Fallback to API
3. **Local processing**:
   - Playwright fetches rendered HTML (1-3s)
   - ReaderLM-v2 converts HTML → Markdown (1-2s)
   - Return clean markdown
4. **API fallback** (if needed):
   - Call Jina Reader API
   - Return markdown

### Resource Usage

```yaml
llama-server-reader:
  GPU: 1x Blackwell (overkill for 1.5B model)
  VRAM: ~2-3 GB
  RAM: 4 GB (shm_size)
  Context: 64K tokens

playwright-fetcher:
  CPU: 1-2 cores
  RAM: 2 GB (shm_size)
  Browser: Chromium headless
```

## Troubleshooting

### Model Not Loading

```bash
# Check model file exists
ls -lh /home/trevor/models/ReaderLM-v2.Q4_K_M.gguf

# Check logs
docker logs strieber-llama-server-reader

# Verify GPU access
docker exec strieber-llama-server-reader nvidia-smi
```

### Playwright Fetch Failing

```bash
# Check Playwright service
docker logs strieber-playwright-fetcher

# Test directly
curl "http://localhost:8005/health"

# Check browser installation
docker exec strieber-playwright-fetcher playwright --version
```

### Performance Issues

```bash
# Check GPU utilization
nvidia-smi

# Check model inference speed
docker logs strieber-llama-server-reader | grep "tokens/s"

# Monitor resource usage
docker stats strieber-llama-server-reader strieber-playwright-fetcher
```

### Disable Local Processing

If you want to use only the Jina API:

```bash
# In .env file
USE_LOCAL_READER=false

# Restart services
docker compose restart mcp-jina-reader
```

## Benefits of Local Processing

1. **No Rate Limits**: Unlimited requests, no throttling
2. **Cost**: Completely free (no API fees)
3. **Privacy**: All processing happens locally
4. **Speed**: 2-5 seconds average (comparable to API)
5. **Control**: Full control over processing pipeline
6. **Reliability**: No external dependencies for HTML pages

## Limitations

1. **PDF Support**: Still requires Jina API (ReaderLM doesn't process PDFs)
2. **Complex Auth**: Pages requiring authentication may need API
3. **Cloudflare**: Some Cloudflare-protected sites may need API
4. **CAPTCHA**: Cannot solve CAPTCHAs (would need API)

## Future Enhancements

- [ ] Add Mozilla Readability.js preprocessing (like official Jina Reader)
- [ ] Support for authentication (cookies, headers)
- [ ] Cloudflare bypass with stealth plugins
- [ ] PDF support via local PDF.js + ReaderLM
- [ ] Caching layer for frequently accessed pages
- [ ] Metrics/monitoring dashboard

## References

- **ReaderLM-v2 Model**: https://huggingface.co/jinaai/ReaderLM-v2
- **ReaderLM-v2 Paper**: https://arxiv.org/abs/2503.01151
- **Jina Reader API**: https://jina.ai/reader/
- **Playwright**: https://playwright.dev/python/
- **llama.cpp**: https://github.com/ggerganov/llama.cpp

## Support

For issues or questions:
- Check logs: `docker compose logs -f`
- Verify health: `docker compose ps`
- Review this guide: `READERLM-SETUP.md`
- Original Jina Reader: https://github.com/jina-ai/reader
