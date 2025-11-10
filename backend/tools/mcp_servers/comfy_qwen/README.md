# ComfyUI Qwen MCP Server

Production-ready MCP server for **Qwen Image** generation workflows via ComfyUI, with full Open WebUI integration.

## Features

- **Two MCP Tools:**
  - `qwen_image`: Text-to-image generation with Qwen model
  - `qwen_image_edit`: Image-to-image editing and inpainting

- **Non-Vision Model Compatible:**
  - Returns ResourceLink blocks (OWUI-hosted URLs) by default
  - Optional inline thumbnails only when explicitly requested
  - Works with models that don't support vision/multimodal input

- **Progress Streaming:**
  - Real-time progress via WebSocket (with polling fallback)
  - Updates sent via MCP `send_progress_notification`

- **Open WebUI Integration:**
  - Automatic upload of generated images to OWUI Files API
  - Pre-model filter for converting inline images to file references
  - SSRF protection for security

## Architecture

```
comfy_qwen/
├── server.py                  # MCP server with qwen_image & qwen_image_edit tools
├── comfy_client.py            # ComfyUI API client (queue, progress, outputs)
├── owui_client.py             # Open WebUI Files API client
├── filter/
│   └── image_to_file_router.py  # Pre-model filter for non-vision models
├── workflows/
│   ├── qwen_image_api.json    # Text-to-image workflow template
│   └── qwen_edit_api.json     # Image editing workflow template
└── README.md
```

## Prerequisites

- **ComfyUI** running with Qwen model loaded (default: `http://127.0.0.1:8188`)
- **Open WebUI** with Files API enabled
- Python 3.10+
- Dependencies: `mcp`, `httpx`, `websockets`, `Pillow`, `pydantic`

## Installation

### 1. Install Dependencies

Add to your `requirements.txt`:

```txt
# Already included in parent requirements.txt:
# mcp>=0.1.0
# httpx>=0.25.0
# uvicorn>=0.24.0
# starlette>=0.35.0

# Additional for ComfyQwen:
websockets>=12.0
Pillow>=10.0.0
```

Install:

```bash
pip install websockets Pillow
```

### 2. Configure Environment Variables

Create or update `.env`:

```bash
# ComfyUI Configuration
COMFY_URL=http://strieber-comfyui:8188  # Or http://127.0.0.1:9040 for host access

# Open WebUI Configuration
OWUI_BASE_URL=http://strieber-open-webui:8080  # Or http://localhost:9200 for host
OWUI_API_TOKEN=your_api_token_here

# Server Configuration (optional)
HOST=0.0.0.0
PORT=8000
```

**Getting your OWUI API Token:**

1. Log into Open WebUI
2. Go to Settings → Account → API Keys
3. Generate a new API key
4. Copy and set as `OWUI_API_TOKEN`

### 3. Register with Launcher

Edit `backend/tools/mcp_servers/launcher.py` to add the new server:

```python
elif server_module == "comfy_qwen":
    from comfy_qwen import server as comfy_qwen_server
    mcp_instance = comfy_qwen_server.get_mcp()
```

## Running the Server

### Option A: Using Launcher (Recommended for Production)

```bash
cd backend/tools/mcp_servers
export SERVER_MODULE=comfy_qwen
export COMFY_URL=http://127.0.0.1:9040
export OWUI_BASE_URL=http://localhost:9200
export OWUI_API_TOKEN=your_token
python launcher.py
```

The server will bind to `0.0.0.0:8000` by default.

### Option B: Direct Execution (Development)

```bash
cd backend/tools/mcp_servers
python -m comfy_qwen.server
```

### Option C: Docker Deployment

Add to your `compose.yml`:

```yaml
comfy-qwen-mcp:
  build:
    context: ./backend/tools/mcp_servers
    dockerfile: Dockerfile.mcp-server
  image: strieber-mcp-comfy-qwen:latest
  container_name: strieber-mcp-comfy-qwen
  restart: unless-stopped

  ports:
    - "8010:8000"  # Adjust port as needed

  environment:
    - SERVER_MODULE=comfy_qwen
    - COMFY_URL=http://strieber-comfyui:8188
    - OWUI_BASE_URL=http://strieber-open-webui:8080
    - OWUI_API_TOKEN=${OWUI_API_TOKEN}
    - HOST=0.0.0.0
    - PORT=8000

  networks:
    - strieber-network
```

## Tool Usage

### Tool 1: `qwen_image` (Text-to-Image)

Generate images from text prompts.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | *required* | Text description of image to generate |
| `negative_prompt` | str | `""` | What to avoid in the image |
| `width` | int | `1024` | Image width (512-2048) |
| `height` | int | `1024` | Image height (512-2048) |
| `steps` | int | `20` | Sampling steps (1-100) |
| `guidance` | float | `5.0` | Guidance scale/CFG (1.0-20.0) |
| `seed` | int | `null` | Random seed (for reproducibility) |
| `batch_size` | int | `1` | Number of images (1-4) |
| `inline_preview` | bool | `false` | Include inline thumbnail |
| `upload_results_to_openwebui` | bool | `true` | Upload to OWUI Files |

**Example Call (JSON):**

```json
{
  "prompt": "cinematic photo of an astronaut on the moon, highly detailed",
  "negative_prompt": "blurry, low quality, cartoon",
  "width": 768,
  "height": 1024,
  "steps": 28,
  "guidance": 4.5,
  "seed": 12345,
  "batch_size": 1,
  "inline_preview": false,
  "upload_results_to_openwebui": true
}
```

**Returns:**

```
[
  TextContent("Generated 1 image(s)..."),
  EmbeddedResource(url="http://owui/api/v1/files/{id}/content", mimeType="image/png")
]
```

### Tool 2: `qwen_image_edit` (Image Editing)

Edit or modify existing images (img2img/inpaint).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `init_image_file_id` | str | `null` | OWUI file ID of init image |
| `init_image_url` | str | `null` | OWUI URL of init image |
| `prompt` | str | `null` | Changes to make |
| `mask_file_id` | str | `null` | OWUI file ID of mask (inpaint) |
| `mask_image_url` | str | `null` | OWUI URL of mask |
| `strength` | float | `0.7` | Denoising strength (0.0-1.0) |
| `steps` | int | `30` | Sampling steps |
| `guidance` | float | `5.0` | Guidance scale/CFG |
| `seed` | int | `null` | Random seed |
| `inline_preview` | bool | `false` | Include inline thumbnail |
| `upload_results_to_openwebui` | bool | `true` | Upload to OWUI |

**Example Call:**

```json
{
  "prompt": "replace sky with dramatic storm clouds",
  "init_image_file_id": "ab12cd34-ef56-7890",
  "strength": 0.65,
  "steps": 24,
  "guidance": 4.0,
  "seed": 777
}
```

**Returns:** Same format as `qwen_image`

## Open WebUI Filter Setup

The included filter converts inline images to OWUI file references for non-vision models.

### Installation Steps:

1. **Copy filter code:**
   - Open `filter/image_to_file_router.py`
   - Copy entire contents

2. **Add to Open WebUI:**
   - Navigate to Open WebUI → Settings → Functions
   - Click "+" to create new function
   - Paste the filter code
   - Set function type to "Filter"

3. **Configure:**
   - Set `owui_base_url` (e.g., `http://localhost:9200`)
   - Set `owui_api_token` (your API token)
   - Enable `enable_for_non_vision_only` (default: `true`)

4. **Activate:**
   - Enable the filter globally or per-model
   - Non-vision models will now convert inline images to file URLs

### How It Works:

1. User sends message with inline images
2. Filter detects non-vision model
3. Extracts `image_url` blocks from message
4. Uploads each image to OWUI Files API
5. Replaces image blocks with text references
6. Adds hint to use `qwen_image_edit` tool

**Before Filter:**
```
User: [image] Edit this to add a sunset
```

**After Filter (for non-vision model):**
```
User: Edit this to add a sunset

Images uploaded to Open WebUI Files:
- http://owui/api/v1/files/abc123/content

Note: If you want to edit these images, use the qwen_image_edit tool
with init_image_url set to one of the URLs above.
```

## Workflow Configuration

### Workflow Templates

The server uses workflow JSON files in `workflows/`:

- `qwen_image_api.json` - Text-to-image workflow
- `qwen_edit_api.json` - Image editing workflow

### Node ID Mapping

The server modifies these workflows by node ID:

**Text-to-Image (qwen_image_api.json):**
- Node `"3"` - Positive prompt
- Node `"4"` - Negative prompt
- Node `"2"` - Latent size (width, height, batch)
- Node `"5"` - Sampler (seed, steps, cfg, denoise)

**Image Edit (qwen_edit_api.json):**
- Node `"2"` - Load init image
- Node `"4"` - Positive prompt
- Node `"5"` - Negative prompt
- Node `"6"` - Sampler (seed, steps, cfg, denoise/strength)
- Node `"9"` - Load mask (optional)

### Customizing Workflows

To use your own ComfyUI workflows:

1. **Export from ComfyUI:**
   - Load your workflow in ComfyUI
   - Click "Save (API Format)"
   - Save as JSON

2. **Update node IDs in server.py:**
   ```python
   # Update these constants to match your workflow:
   TXT2IMG_NODE_POSITIVE_PROMPT = "3"  # Your prompt node ID
   TXT2IMG_NODE_SAMPLER = "5"          # Your sampler node ID
   # etc.
   ```

3. **Replace workflow file:**
   - Copy your exported JSON to `workflows/`
   - Update filename in `server.py` if changed

## Security

### SSRF Protection

The OWUI client enforces strict URL validation:

- Only downloads from configured `OWUI_BASE_URL` domain
- Rejects arbitrary external URLs
- Size limits enforced (30 MB default)
- Content-type validation (images only)

### Authentication

- All OWUI API calls require Bearer token
- Token passed via `OWUI_API_TOKEN` environment variable
- Never expose tokens in logs or error messages

### Size Limits

- Max image download: 30 MB (configurable)
- Max image upload: 30 MB (configurable)
- Enforced during streaming to prevent memory exhaustion

## Troubleshooting

### Issue: "OWUI client not configured"

**Solution:** Set `OWUI_BASE_URL` and `OWUI_API_TOKEN` environment variables.

### Issue: "ComfyUI connection refused"

**Solution:**
- Check ComfyUI is running: `curl http://127.0.0.1:8188`
- Verify `COMFY_URL` matches your ComfyUI instance
- Check firewall/network settings

### Issue: "Execution error: Model not found"

**Solution:**
- Verify Qwen model is loaded in ComfyUI
- Update `ckpt_name` in workflow JSON files
- Check ComfyUI logs for model loading errors

### Issue: "SSRF protection: URL domain does not match"

**Solution:**
- Only OWUI-hosted URLs are allowed by design
- Upload images to OWUI first via Files API
- Update `OWUI_BASE_URL` to match your instance

### Issue: "WebSocket connection failed"

**Solution:**
- Not critical - server automatically falls back to polling
- Check ComfyUI WebSocket endpoint is accessible
- Verify no proxy/firewall blocking WebSocket connections

## Development

### Running Tests

```bash
cd backend/tools/mcp_servers
pytest comfy_qwen/
```

### Logging

Set log level via environment:

```bash
export LOG_LEVEL=DEBUG
python -m comfy_qwen.server
```

### Local Development

```bash
# Install in editable mode
pip install -e backend/tools/mcp_servers

# Run with live reload
uvicorn comfy_qwen.server:mcp.streamable_http_app --reload --host 0.0.0.0 --port 8000
```

## License

Same as parent project (strieber-gpt-3).

## Credits

- Built with [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- ComfyUI integration
- Open WebUI Files API
- Qwen-VL models
