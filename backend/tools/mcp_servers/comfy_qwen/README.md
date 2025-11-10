# ComfyUI Qwen MCP Server

Production-ready MCP server that fronts ComfyUI workflows for **Qwen Image** generation and editing. Designed for seamless integration with Open WebUI, supporting both vision and non-vision models.

## Features

- **Two MCP Tools:**
  - `qwen_image`: Text-to-image generation (txt2img)
  - `qwen_image_edit`: Image editing and inpainting (img2img)

- **Streamable HTTP Transport:** Ready for Docker deployment with progress streaming

- **Open WebUI Integration:**
  - Automatic file uploads to OWUI Files API
  - Returns resource links (works with non-vision models)
  - Optional inline image previews
  - Pre-model filter for converting inline images to file URLs

- **Progress Tracking:** Real-time progress notifications via MCP

- **Security:** SSRF guards, file size limits, content-type validation

## Architecture

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Open WebUI  │─MCP─→│  MCP Server  │─HTTP→│   ComfyUI    │
│   (Client)   │      │ (this server)│      │  (Workflow)  │
└──────────────┘      └──────────────┘      └──────────────┘
                             │
                             ├─ Upload results
                             ↓
                      ┌──────────────┐
                      │  OWUI Files  │
                      │     API      │
                      └──────────────┘
```

## Directory Structure

```
comfy_qwen/
├── server.py                       # Main MCP server with tools
├── comfy_client.py                 # ComfyUI API client
├── owui_client.py                  # Open WebUI Files API client
├── workflows/
│   ├── qwen_image_api.json         # txt2img workflow template
│   └── qwen_edit_api.json          # img2img workflow template
├── filter/
│   └── image_to_file_router.py     # Open WebUI pre-model filter
└── README.md
```

## Prerequisites

1. **ComfyUI** running with Qwen models loaded
2. **Open WebUI** with Files API enabled
3. **Python 3.10+** with required packages (see `requirements.txt`)

## Setup

### 1. Install Dependencies

From the `backend/tools/mcp_servers` directory:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file or export these variables:

```bash
# ComfyUI Configuration
export COMFY_URL="http://127.0.0.1:8188"  # Or your ComfyUI URL

# Open WebUI Configuration
export OWUI_BASE_URL="https://webui.example.com"  # Your OWUI URL
export OWUI_API_TOKEN="your_api_token_here"       # Bearer token for Files API

# Server Configuration (optional)
export HOST="0.0.0.0"
export PORT="8000"
```

**How to get OWUI_API_TOKEN:**
1. Log into Open WebUI
2. Go to Settings → Account → API Keys
3. Create a new API key
4. Copy the token value

### 3. Export ComfyUI Workflows

You need to export your actual ComfyUI workflows in **API format**:

1. Open ComfyUI and load your Qwen Image workflow
2. Click **"Save (API Format)"** (not regular save!)
3. Replace `workflows/qwen_image_api.json` with the exported file
4. Repeat for your Qwen Edit workflow → `workflows/qwen_edit_api.json`

### 4. Update Node ID Mappings

Edit `server.py` and update the node ID mappings to match your workflows:

```python
# Around line 43-49 for qwen_image
QWEN_IMAGE_NODES = {
    "positive_prompt": "2",  # Update with your CLIPTextEncode node ID
    "negative_prompt": "3",  # Update with your negative prompt node ID
    "empty_latent": "4",     # Update with your EmptyLatentImage node ID
    "sampler": "5",          # Update with your KSampler node ID
}

# Around line 52-58 for qwen_edit
QWEN_EDIT_NODES = {
    "load_image": "2",       # Update with your LoadImage node ID
    "positive_prompt": "3",  # Update with your CLIPTextEncode node ID
    "negative_prompt": "4",  # Update with your negative prompt node ID
    "sampler": "6",          # Update with your KSampler node ID
}
```

**How to find node IDs:**
- Open the exported JSON workflow file
- Node IDs are the top-level keys (e.g., `"1"`, `"2"`, `"3"`)
- Match them to node titles in the `_meta.title` field

## Running the Server

### Option 1: Standalone (for testing)

```bash
cd backend/tools/mcp_servers/comfy_qwen
python server.py
```

Server will start on `http://0.0.0.0:8000` by default.

### Option 2: Via Launcher (production)

```bash
cd backend/tools/mcp_servers
export SERVER_MODULE="comfy_qwen"
export PORT="8000"
python launcher.py
```

### Option 3: Docker (recommended)

Add to your `docker-compose.yml`:

```yaml
services:
  mcp-comfy-qwen:
    build:
      context: ./backend/tools/mcp_servers
      dockerfile: Dockerfile.mcp-server
    container_name: mcp-comfy-qwen
    restart: unless-stopped
    ports:
      - "9050:8000"
    environment:
      - SERVER_MODULE=comfy_qwen
      - COMFY_URL=http://comfyui:8188
      - OWUI_BASE_URL=http://open-webui:8080
      - OWUI_API_TOKEN=${OWUI_API_TOKEN}
    depends_on:
      - comfyui
      - open-webui
```

Then run:

```bash
docker-compose up -d mcp-comfy-qwen
```

## Connecting to Open WebUI

### 1. Add MCP Server

In Open WebUI:

1. Go to **Settings → Connections → MCP Servers**
2. Click **"Add Server"**
3. Configure:
   - **Name:** `comfy_qwen`
   - **URL:** `http://localhost:9050` (or your server URL)
   - **Transport:** `streamable-http`
4. Click **"Test Connection"** to verify
5. Click **"Save"**

### 2. Install the Image Filter (Optional but Recommended)

For non-vision model support:

1. Go to **Settings → Functions**
2. Click **"+ Add Function"**
3. Click **"Import from file"**
4. Upload `filter/image_to_file_router.py`
5. Enable the filter for your workspace

**What the filter does:**
- Detects when non-vision models receive images
- Uploads images to OWUI Files API
- Converts inline images to file URLs
- Adds hint to use `qwen_image_edit` tool

## Usage Examples

### Example 1: Text-to-Image Generation

In Open WebUI chat:

```
Generate a cinematic photo of an astronaut on the moon,
high detail, dramatic lighting
```

Behind the scenes, the AI will call:

```json
{
  "tool": "qwen_image",
  "arguments": {
    "prompt": "cinematic photo of an astronaut on the moon, high detail, dramatic lighting",
    "negative_prompt": "blurry, low quality, distorted",
    "width": 1024,
    "height": 1024,
    "steps": 28,
    "guidance": 5.0
  }
}
```

**Result:** Text summary + resource links to generated images

### Example 2: Image Editing

1. Upload an image in the chat
2. Say: `Replace the sky with dramatic storm clouds`

The filter will:
1. Upload your image to OWUI Files
2. Convert it to a file URL
3. Suggest using `qwen_image_edit`

The AI will then call:

```json
{
  "tool": "qwen_image_edit",
  "arguments": {
    "prompt": "Replace the sky with dramatic storm clouds",
    "init_image_url": "https://webui.example.com/api/v1/files/abc123/content",
    "strength": 0.7,
    "steps": 30,
    "guidance": 5.0
  }
}
```

**Result:** Text summary + resource links to edited images

### Example 3: Advanced Generation

```
Create 2 images of a futuristic cityscape at sunset,
size 768x1024, 35 sampling steps, high quality
```

Parameters extracted:
- `batch_size: 2`
- `width: 768`, `height: 1024`
- `steps: 35`
- `prompt: "futuristic cityscape at sunset, high quality"`

### Example 4: Inpainting (with mask)

1. Upload an image and a mask
2. Say: `Fill the masked area with a mountain landscape`

The tool will use the mask for targeted editing.

## Tool Reference

### qwen_image

**Purpose:** Generate images from text descriptions

**Parameters:**
- `prompt` (required): Text description of desired image
- `negative_prompt` (optional): Things to avoid
- `width` (default: 1024): Image width (512-2048)
- `height` (default: 1024): Image height (512-2048)
- `steps` (default: 20): Sampling steps (1-150)
- `guidance` (default: 5.0): Guidance/CFG scale (1.0-30.0)
- `seed` (optional): Random seed for reproducibility
- `batch_size` (default: 1): Number of images (1-4)
- `inline_preview` (default: false): Include thumbnail
- `upload_results_to_openwebui` (default: true): Upload to Files

**Returns:**
- `TextContent`: Generation summary
- `ResourceLink`: Links to full-resolution images
- `ImageContent` (if `inline_preview=true`): Thumbnail

### qwen_image_edit

**Purpose:** Edit/modify existing images

**Parameters:**
- `prompt` (optional): Editing instruction
- `init_image_file_id` (one required): OWUI file ID
- `init_image_url` (one required): Image URL
- `mask_file_id` (optional): Mask file ID (for inpainting)
- `mask_image_url` (optional): Mask URL
- `strength` (default: 0.7): Denoising strength (0.0-1.0)
- `steps` (default: 30): Sampling steps
- `guidance` (default: 5.0): Guidance scale
- `seed` (optional): Random seed
- `inline_preview` (default: false): Include thumbnail
- `upload_results_to_openwebui` (default: true): Upload to Files

**Returns:** Same as `qwen_image`

## Troubleshooting

### Issue: "Failed to queue workflow"

**Causes:**
- ComfyUI not running
- Wrong `COMFY_URL`
- Workflow JSON has errors

**Solutions:**
1. Check ComfyUI is accessible: `curl http://localhost:8188`
2. Verify `COMFY_URL` in environment
3. Test workflow manually in ComfyUI first
4. Check server logs for detailed errors

### Issue: "Failed to upload file to Open WebUI"

**Causes:**
- OWUI Files API not enabled
- Invalid `OWUI_API_TOKEN`
- Wrong `OWUI_BASE_URL`

**Solutions:**
1. Verify OWUI is running and accessible
2. Check Files API is enabled in OWUI settings
3. Generate a new API token in OWUI
4. Ensure `OWUI_BASE_URL` matches your OWUI URL exactly

### Issue: "No outputs found in workflow history"

**Causes:**
- Workflow failed silently
- Wrong node IDs in mappings
- Model not loaded in ComfyUI

**Solutions:**
1. Check ComfyUI logs for errors
2. Verify node ID mappings in `server.py`
3. Ensure Qwen model is loaded: check ComfyUI UI
4. Test workflow manually in ComfyUI first

### Issue: "URL not from Open WebUI domain"

**Cause:** SSRF protection blocking external URLs

**Solution:**
- Only use OWUI-hosted file URLs
- Images must be uploaded to OWUI Files first
- The filter handles this automatically

### Issue: Progress stuck at 90%

**Cause:** Workflow taking longer than expected

**Solution:**
- This is normal for high-resolution images
- Wait for completion (check ComfyUI UI)
- Increase timeout in `comfy_client.py` if needed

## Development & Testing

### Run Tests

```bash
cd backend/tools/mcp_servers
pytest comfy_qwen/tests/ -v
```

### Manual Testing

```bash
# Start server
python comfy_qwen/server.py

# In another terminal, test with curl
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "qwen_image",
    "arguments": {
      "prompt": "test image",
      "width": 512,
      "height": 512,
      "steps": 10
    }
  }'
```

### Enable Debug Logging

```python
# In server.py, change:
logging.basicConfig(level=logging.DEBUG)
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFY_URL` | `http://127.0.0.1:8188` | ComfyUI base URL |
| `OWUI_BASE_URL` | *(required)* | Open WebUI base URL |
| `OWUI_API_TOKEN` | *(required)* | Open WebUI API token |
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `8000` | Server port |

### Filter Configuration (Valves)

In Open WebUI filter settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `priority` | `0` | Filter execution order |
| `owui_base_url` | *(auto)* | Override OWUI URL |
| `owui_api_token` | *(auto)* | Override API token |
| `enabled_for_non_vision_only` | `true` | Only run for non-vision models |
| `add_tool_hint` | `true` | Add usage hint for tools |
| `max_image_size_mb` | `30` | Max image size limit |

## Security Considerations

### SSRF Protection

- Only OWUI-hosted URLs are allowed by default
- External URLs are blocked to prevent SSRF attacks
- Override with caution in trusted environments

### File Size Limits

- Images limited to 30 MB by default
- Configurable via filter valves
- Prevents DoS via large file uploads

### Authentication

- All OWUI API calls require valid Bearer token
- Tokens should be kept secret (use env vars)
- Rotate tokens periodically

### Content Validation

- All downloaded files verified as images
- Content-Type headers checked
- Malformed data rejected

## Performance Tips

1. **Batch Processing:** Use `batch_size` for multiple variations
2. **Workflow Optimization:** Simplify ComfyUI workflows where possible
3. **Caching:** ComfyUI caches models - keep them loaded
4. **Resolution:** Lower resolution = faster generation
5. **Steps:** 20-30 steps often sufficient for good quality

## Contributing

When modifying this server:

1. Update node mappings in `server.py` for your workflows
2. Test with various parameter combinations
3. Update this README with new features
4. Add tests for new functionality

## License

MIT License - see repository root for details

## Support

For issues:
1. Check ComfyUI logs
2. Check MCP server logs
3. Enable debug logging
4. Verify environment variables
5. Test workflows manually in ComfyUI
6. Open an issue with logs and configuration

## Credits

Built with:
- [FastMCP](https://github.com/modelcontextprotocol/python-sdk) - MCP Python SDK
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Workflow engine
- [Open WebUI](https://github.com/open-webui/open-webui) - Web interface
- [Qwen Models](https://github.com/QwenLM) - Image generation models
