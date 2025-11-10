# ComfyUI Qwen MCP Server

Production-ready MCP server that fronts ComfyUI workflows for **Qwen Image** generation and editing. Designed for seamless integration with Open WebUI, supporting both vision and non-vision models.

## Features

- **Two MCP Tools:**
  - `qwen_image`: Text-to-image generation (txt2img)
  - `qwen_image_edit`: Image editing and inpainting (img2img)

- **Quality Presets with Lightning LoRA:**
  - `fast`: 8-step Lightning LoRA (2-3 seconds, 12-25× faster)
  - `standard`: 20-step standard generation (balanced)
  - `high`: 50-step maximum quality (best detail)

- **OpenAI-Style API:** Simple, familiar parameters (`quality`, `size`, `n`) with advanced overrides

- **Streamable HTTP Transport:** Ready for Docker deployment with progress streaming

- **Open WebUI Integration:**
  - Automatic file uploads to OWUI Files API
  - Returns resource links (works with non-vision models)
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

## What's New: Lightning LoRA Support

**Lightning LoRAs** provide 12-25× speed improvements with minimal quality loss:

- **Default behavior**: `quality="fast"` uses 8-step Lightning LoRA
- **Automatic switching**: Quality presets automatically enable/disable LoRA
- **Manual override**: Advanced users can force `use_lightning=true/false`

### Speed Comparison

| Quality | Steps | LoRA | Time (estimate) | Use Case |
|---------|-------|------|-----------------|----------|
| `fast` | 8 | ✓ Lightning | ~2-3 sec | Quick iterations, previews |
| `standard` | 20 | ✗ None | ~8-10 sec | Balanced quality/speed |
| `high` | 50 | ✗ None | ~20-30 sec | Final renders, maximum detail |

## Prerequisites

1. **ComfyUI** running with Qwen models loaded
2. **Lightning LoRA files** (optional, for `fast` quality):
   - Download from [HuggingFace](https://huggingface.co/lightx2v/Qwen-Image-Lightning)
   - Place in `ComfyUI/models/loras/`:
     - `Qwen-Image-Lightning-8steps-V1.1.safetensors` (txt2img)
     - `Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors` (img2img)
3. **Open WebUI** with Files API enabled
4. **Python 3.10+** with required packages (see `requirements.txt`)

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

### 3. Download Lightning LoRA Models (Optional)

For `fast` quality preset:

```bash
# Navigate to ComfyUI models directory
cd /path/to/ComfyUI/models/loras

# Download txt2img Lightning LoRA
wget https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V1.1.safetensors

# Download img2img/edit Lightning LoRA
wget https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors
```

**Note:** If you skip this step, `quality="fast"` will fail. Use `quality="standard"` or `quality="high"` instead.

### 4. Export ComfyUI Workflows

You need to export your actual ComfyUI workflows in **API format**:

1. Open ComfyUI and load your Qwen Image workflow
2. **Important:** Add a `LoraLoaderModelOnly` node between the checkpoint and sampler
3. Click **"Save (API Format)"** (not regular save!)
4. Replace `workflows/qwen_image_api.json` with the exported file
5. Repeat for your Qwen Edit workflow → `workflows/qwen_edit_api.json`

### 5. Update Node ID Mappings

Edit `server.py` and update the node ID mappings to match your workflows:

```python
# Around line 108-115 for qwen_image
QWEN_IMAGE_NODES = {
    "checkpoint_loader": "1",    # Update with your CheckpointLoader node ID
    "lora_loader": "10",         # Update with your LoraLoaderModelOnly node ID
    "positive_prompt": "2",      # Update with your positive CLIPTextEncode node ID
    "negative_prompt": "3",      # Update with your negative CLIPTextEncode node ID
    "empty_latent": "4",         # Update with your EmptyLatentImage node ID
    "sampler": "5",              # Update with your KSampler node ID
}

# Around line 119-126 for qwen_edit
QWEN_EDIT_NODES = {
    "checkpoint_loader": "1",    # Update with your CheckpointLoader node ID
    "lora_loader": "10",         # Update with your LoraLoaderModelOnly node ID
    "load_image": "2",           # Update with your LoadImage node ID
    "positive_prompt": "3",      # Update with your positive CLIPTextEncode node ID
    "negative_prompt": "4",      # Update with your negative CLIPTextEncode node ID
    "sampler": "6",              # Update with your KSampler node ID
}
```

**How to find node IDs:**
- Open the exported JSON workflow file
- Node IDs are the top-level keys (e.g., `"1"`, `"2"`, `"10"`)
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

### Example 1: Fast Text-to-Image (Default)

In Open WebUI chat:

```
Generate a sunset over mountains with dramatic lighting
```

Behind the scenes, the AI will call:

```json
{
  "tool": "qwen_image",
  "arguments": {
    "prompt": "sunset over mountains with dramatic lighting",
    "quality": "fast"
  }
}
```

**Result:** 8-step Lightning LoRA generation (~2-3 seconds)

### Example 2: High-Quality Generation

```
Create a detailed portrait of a wise elder, high quality, sharp details
```

AI calls:

```json
{
  "tool": "qwen_image",
  "arguments": {
    "prompt": "detailed portrait of a wise elder, sharp details",
    "quality": "high",
    "negative_prompt": "blurry, low quality, distorted"
  }
}
```

**Result:** 50-step standard generation (~20-30 seconds)

### Example 3: Custom Size and Multiple Images

```
Generate 2 wide landscape images of a futuristic city, size 1344x768
```

AI calls:

```json
{
  "tool": "qwen_image",
  "arguments": {
    "prompt": "futuristic city",
    "quality": "fast",
    "size": "1344x768",
    "n": 2
  }
}
```

**Result:** 2 images at 1344x768 resolution

### Example 4: Image Editing

1. Upload an image in the chat
2. Say: `Make the sky more dramatic with storm clouds`

The filter will:
1. Upload your image to OWUI Files
2. Convert it to a file URL

The AI will then call:

```json
{
  "tool": "qwen_image_edit",
  "arguments": {
    "prompt": "make the sky more dramatic with storm clouds",
    "init_image_url": "https://webui.example.com/api/v1/files/abc123/content",
    "quality": "fast",
    "strength": 0.7
  }
}
```

**Result:** Fast 8-step edit with Lightning LoRA

### Example 5: Advanced - Reproducible Generation

```
Generate an astronaut on the moon, seed 42, standard quality
```

AI calls:

```json
{
  "tool": "qwen_image",
  "arguments": {
    "prompt": "astronaut on the moon",
    "quality": "standard",
    "seed": 42
  }
}
```

**Result:** Reproducible 20-step generation (same seed = same image)

## Tool Reference

### qwen_image

**Purpose:** Generate images from text descriptions

**Simple Parameters:**
- `prompt` (required): Text description of desired image
- `quality` (default: `"fast"`): `"fast"` | `"standard"` | `"high"`
- `size` (default: `"1024x1024"`): Image size (see sizes below)
- `n` (default: `1`): Number of images to generate (1-4)
- `negative_prompt` (default: `""`): What to avoid
- `seed` (optional): Random seed for reproducibility

**Available Sizes:**
- `"1024x1024"` - Square (default)
- `"1024x768"` - Landscape 4:3
- `"768x1024"` - Portrait 3:4
- `"1280x720"` - Landscape 16:9
- `"720x1280"` - Portrait 9:16
- `"1344x768"` - Wide landscape
- `"768x1344"` - Tall portrait

**Advanced Parameters (optional):**
- `steps` - Override sampling steps
- `guidance` - Override CFG scale
- `use_lightning` - Force Lightning LoRA on/off
- `upload_results_to_openwebui` (default: `true`)

**Returns:**
- `TextContent`: Generation summary with settings
- `ResourceLink`: Links to full-resolution images on OWUI

### qwen_image_edit

**Purpose:** Edit or transform existing images

**Simple Parameters:**
- `prompt`: Editing instruction
- `init_image_file_id` | `init_image_url` (one required): Base image source
- `quality` (default: `"fast"`): `"fast"` | `"standard"` | `"high"`
- `strength` (default: `0.7`): Denoising strength (0.0-1.0)
- `mask_file_id` | `mask_image_url` (optional): Inpainting mask
- `negative_prompt` (default: `""`): What to avoid
- `seed` (optional): Random seed for reproducibility

**Advanced Parameters (optional):**
- `steps` - Override sampling steps
- `guidance` - Override CFG scale
- `use_lightning` - Force Lightning LoRA on/off
- `upload_results_to_openwebui` (default: `true`)

**Returns:** Same as `qwen_image`

## Quality Preset Details

### Fast (Default)

```python
{
  "use_lightning": True,
  "lora_file": "Qwen-Image-Lightning-8steps-V1.1.safetensors",
  "steps": 8,
  "guidance": 2.5
}
```

- **Speed:** ~2-3 seconds
- **Use case:** Quick iterations, previews, most generations
- **Quality:** Excellent (12-25× faster with minimal loss)

### Standard

```python
{
  "use_lightning": False,
  "steps": 20,
  "guidance": 5.0
}
```

- **Speed:** ~8-10 seconds
- **Use case:** Balanced quality/speed when LoRA unavailable
- **Quality:** Good baseline

### High

```python
{
  "use_lightning": False,
  "steps": 50,
  "guidance": 5.0
}
```

- **Speed:** ~20-30 seconds
- **Use case:** Final renders, maximum detail
- **Quality:** Best possible

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

### Issue: "Lightning LoRA file not found"

**Cause:** Lightning LoRA files not in `ComfyUI/models/loras/`

**Solutions:**
1. Download Lightning LoRAs (see Setup step 3)
2. Place files in correct directory
3. Or use `quality="standard"` or `quality="high"` instead

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
- LoRA node misconfigured

**Solutions:**
1. Check ComfyUI logs for errors
2. Verify node ID mappings in `server.py`
3. Ensure Qwen model is loaded: check ComfyUI UI
4. Verify LoRA node is properly connected in workflow
5. Test workflow manually in ComfyUI first

### Issue: "URL not from Open WebUI domain"

**Cause:** SSRF protection blocking external URLs

**Solution:**
- Only use OWUI-hosted file URLs
- Images must be uploaded to OWUI Files first
- The filter handles this automatically

## Performance Tips

1. **Use fast quality by default:** Lightning LoRA provides 95%+ quality at 12-25× speed
2. **Batch when possible:** Use `n=2` or `n=3` for multiple variations
3. **Optimize workflows:** Simplify ComfyUI workflows where possible
4. **Cache models:** Keep Qwen models loaded in ComfyUI
5. **Right-size images:** Lower resolution = faster generation
6. **Adjust strength:** For edits, lower strength (0.5-0.6) = faster

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

# In another terminal, test with example script
python comfy_qwen/example_usage.py
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

## License

MIT License - see repository root for details

## Credits

Built with:
- [FastMCP](https://github.com/modelcontextprotocol/python-sdk) - MCP Python SDK
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Workflow engine
- [Open WebUI](https://github.com/open-webui/open-webui) - Web interface
- [Qwen Models](https://github.com/QwenLM) - Image generation models
- [Qwen Lightning LoRA](https://huggingface.co/lightx2v/Qwen-Image-Lightning) - Fast generation
