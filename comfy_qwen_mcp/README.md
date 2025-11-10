# ComfyUI Qwen MCP Server

Production-ready MCP (Model Context Protocol) server for ComfyUI Qwen image generation and editing workflows. Provides seamless integration with Open WebUI and supports both vision and non-vision language models.

## Features

- **Two MCP Tools:**
  - `qwen_image`: Text-to-image generation
  - `qwen_image_edit`: Image-to-image editing and inpainting

- **Streamable HTTP Transport:** Standards-compliant MCP server with progress streaming

- **Open WebUI Integration:**
  - Automatic file upload/download via OWUI Files API
  - Returns resource links (works with non-vision models)
  - Optional inline preview thumbnails (for vision models)

- **Pre-model Filter:** Converts inline images to file uploads for non-vision models

- **Robust Error Handling:** Clear error messages with remediation hints

- **Progress Streaming:** Real-time progress updates via WebSocket or polling fallback

- **Security:** SSRF protection, MIME type validation, size limits

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌───────────┐
│  Open WebUI │◄────►│  MCP Server  │◄────►│  ComfyUI  │
│             │      │   (FastMCP)  │      │           │
└─────────────┘      └──────────────┘      └───────────┘
       │                     │
       │                     │
       ▼                     ▼
  OWUI Files API      Workflow Execution
  (file storage)      (image generation)
```

## Installation

### Prerequisites

- Python 3.10+
- ComfyUI running and accessible (default: `http://127.0.0.1:8188`)
- Open WebUI running (optional, for file hosting)
- Qwen model loaded in ComfyUI

### Setup

1. **Clone and install dependencies:**

```bash
cd comfy_qwen_mcp
pip install -r requirements.txt
```

Or with pyproject.toml:

```bash
pip install -e .
```

2. **Configure environment variables:**

Create a `.env` file in the project directory:

```bash
# ComfyUI configuration
COMFY_URL=http://127.0.0.1:8188

# Open WebUI configuration (for file uploads)
OWUI_BASE_URL=https://your-webui.example.com
OWUI_API_TOKEN=your_bearer_token_here

# Server configuration
HOST=0.0.0.0
PORT=8000
```

3. **Prepare ComfyUI workflows:**

The workflows in `workflows/` are templates. You need to:

- Ensure your ComfyUI has the Qwen model checkpoint loaded
- Update `ckpt_name` in the workflow JSONs to match your checkpoint filename
- Verify node IDs match your ComfyUI version (see mapping comments in JSONs)

4. **Get Open WebUI API Token:**

To upload results to Open WebUI:

1. Log into Open WebUI
2. Go to Settings → Account → API Keys
3. Generate a new API key
4. Add it to `.env` as `OWUI_API_TOKEN`

## Running the Server

### Method 1: Direct Python

```bash
cd comfy_qwen_mcp
python server.py
```

The server will start on `http://0.0.0.0:8000` (or your configured `HOST:PORT`).

### Method 2: Uvicorn

```bash
uvicorn server:mcp.get_asgi_app --host 0.0.0.0 --port 8000 --reload
```

### Method 3: Docker (TODO)

A Dockerfile will be provided in future updates.

## Usage

### Tool 1: qwen_image (Text-to-Image)

Generate images from text prompts.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | *required* | Text description of image to generate |
| `negative_prompt` | string | `""` | What to avoid in the image |
| `width` | int | `1024` | Image width (64-4096) |
| `height` | int | `1024` | Image height (64-4096) |
| `steps` | int | `20` | Sampling steps (1-150) |
| `guidance` | float | `5.0` | CFG scale (1.0-30.0) |
| `seed` | int | `null` | Random seed (null for random) |
| `batch_size` | int | `1` | Number of images (1-4) |
| `inline_preview` | bool | `false` | Include base64 thumbnail |
| `upload_results_to_openwebui` | bool | `true` | Upload to OWUI Files |

**Example Call:**

```json
{
  "prompt": "cinematic photo of an astronaut on the moon, highly detailed, 8k",
  "negative_prompt": "blurry, low quality, distorted",
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
  {
    "type": "text",
    "text": "Generated 1 image(s) with prompt: 'cinematic photo...'\nSettings: 768x1024, 28 steps, guidance=4.5, seed=12345"
  },
  {
    "type": "resource",
    "resource": {
      "uri": "https://your-webui.example.com/api/v1/files/abc123/content",
      "mimeType": "image/png",
      "text": "Generated image 1: qwen_txt2img_0_ComfyUI_00001.png"
    }
  }
]
```

### Tool 2: qwen_image_edit (Image-to-Image)

Edit existing images with optional text guidance.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `init_image_file_id` | string | `null` | OWUI file ID for init image |
| `init_image_url` | string | `null` | URL to init image (alternative) |
| `prompt` | string | `null` | Text prompt for edits (optional) |
| `mask_file_id` | string | `null` | OWUI file ID for mask (inpainting) |
| `mask_image_url` | string | `null` | URL to mask image |
| `strength` | float | `0.7` | Denoising strength (0.0-1.0) |
| `steps` | int | `30` | Sampling steps (1-150) |
| `guidance` | float | `5.0` | CFG scale (1.0-30.0) |
| `seed` | int | `null` | Random seed |
| `inline_preview` | bool | `false` | Include base64 thumbnail |
| `upload_results_to_openwebui` | bool | `true` | Upload to OWUI Files |

**Example Call:**

```json
{
  "prompt": "replace sky with dramatic storm clouds",
  "init_image_file_id": "ab12cd34",
  "strength": 0.65,
  "steps": 24,
  "guidance": 4.0,
  "seed": 777,
  "inline_preview": false,
  "upload_results_to_openwebui": true
}
```

**Returns:**

Similar to `qwen_image`, with text summary and resource links to edited images.

## Open WebUI Filter Setup

The pre-model filter converts inline images to file uploads, enabling non-vision models to work with image-based workflows.

### Installation in Open WebUI:

1. **Navigate to Functions:**
   - Open WebUI → Admin Panel → Functions

2. **Create New Function:**
   - Click "Add Function"
   - Name: "Image to File Router"
   - Type: "Filter" (Inlet)

3. **Paste Code:**
   - Copy the entire contents of `filter/image_to_file_router.py`
   - Paste into the function editor

4. **Configure:**
   - Set `owui_base_url` to your Open WebUI URL
   - Set `owui_api_token` to your API key (from Settings → Account → API Keys)
   - Enable `convert_for_nonvision_models`

5. **Activate:**
   - Save and enable the function
   - Set priority (0 for first execution)

### How It Works:

1. User sends message with inline images to a non-vision model
2. Filter intercepts the message
3. Uploads images to OWUI Files API
4. Replaces inline images with text containing file URLs
5. Appends hint to use `qwen_image_edit` tool with the URLs
6. Non-vision model receives only text, but can call tools with the URLs

**Before Filter:**
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Edit this image"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]
}
```

**After Filter:**
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Edit this image"},
    {
      "type": "text",
      "text": "\n\n[Images uploaded to Open WebUI Files]\n1. https://webui.example.com/api/v1/files/xyz789/content\n\nTo edit these images, use the qwen_image_edit tool with init_image_url set to the desired image URL."
    }
  ]
}
```

## Workflow Customization

The workflow templates in `workflows/` are structured for typical Qwen setups. To customize:

### 1. Export Your Workflow from ComfyUI:

- Load your desired workflow in ComfyUI
- Go to Settings → Developer → Enable Dev Mode
- Click "Save (API Format)"
- Save as JSON

### 2. Identify Node IDs:

Open the exported JSON and note the node IDs for:

- **Text encoders** (positive/negative prompts)
- **Latent image** (size, batch)
- **Sampler** (steps, cfg, seed, denoise)
- **Checkpoint loader** (model name)
- **Image loaders** (for editing workflows)
- **Save/output nodes**

### 3. Update Workflow Templates:

Replace the node IDs and structure in `workflows/qwen_image_api.json` or `workflows/qwen_edit_api.json` to match your exported workflow.

### 4. Update Patching Functions:

In `server.py`, update `patch_txt2img_workflow()` and `patch_edit_workflow()` to map parameters to the correct node IDs:

```python
# Example: If your prompt encoder is node "10" instead of "1"
workflow["10"]["inputs"]["text"] = params.prompt
```

## Node Mapping Reference

### qwen_image (txt2img):

| Node ID | Class Type | Purpose | Patched Fields |
|---------|------------|---------|----------------|
| 1 | CLIPTextEncode | Positive prompt | `text` |
| 2 | CLIPTextEncode | Negative prompt | `text` |
| 3 | EmptyLatentImage | Canvas size | `width`, `height`, `batch_size` |
| 4 | KSampler | Sampling | `seed`, `steps`, `cfg` |
| 5 | CheckpointLoaderSimple | Model | `ckpt_name` |
| 6 | VAEDecode | Decode latents | (auto) |
| 7 | SaveImage | Output | `filename_prefix` |

### qwen_edit (img2img):

| Node ID | Class Type | Purpose | Patched Fields |
|---------|------------|---------|----------------|
| 1 | CLIPTextEncode | Positive prompt | `text` |
| 2 | CLIPTextEncode | Negative prompt | `text` |
| 3 | LoadImage | Init image | `image` (filename) |
| 4 | VAEEncode | Encode init | (auto) |
| 5 | KSampler | Sampling | `seed`, `steps`, `cfg`, `denoise` |
| 6 | CheckpointLoaderSimple | Model | `ckpt_name` |
| 7 | VAEDecode | Decode latents | (auto) |
| 8 | SaveImage | Output | `filename_prefix` |

## Troubleshooting

### ComfyUI Connection Issues

**Problem:** `Connection refused` to ComfyUI

**Solution:**
- Verify ComfyUI is running: `http://127.0.0.1:8188`
- Check firewall settings
- Update `COMFY_URL` in `.env`

### Workflow Execution Fails

**Problem:** Workflow execution errors in ComfyUI

**Solution:**
- Check ComfyUI logs for detailed error
- Verify checkpoint name in workflow JSON matches loaded model
- Ensure all custom nodes are installed
- Test workflow manually in ComfyUI web UI first

### Open WebUI Upload Fails

**Problem:** File upload to OWUI returns 401/403

**Solution:**
- Verify `OWUI_API_TOKEN` is correct
- Check token hasn't expired
- Ensure API endpoints are accessible (not behind additional auth)

### Progress Streaming Not Working

**Problem:** No progress updates during generation

**Solution:**
- Check WebSocket connection (browser dev tools)
- Server falls back to polling automatically
- Verify network allows WebSocket connections

### Image Too Large Error

**Problem:** `File too large` error

**Solution:**
- Default limit is 30 MB
- Resize images before upload
- Adjust limit in `owui_client.py` if needed

### Non-Vision Model Not Calling Tools

**Problem:** Model doesn't use image edit tool even after filter conversion

**Solution:**
- Ensure filter is installed and enabled
- Check filter execution order (should run first)
- Verify model has tool-calling capabilities
- Add explicit instruction in user message: "Use qwen_image_edit tool to process this image"

## API Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `COMFY_URL` | No | `http://127.0.0.1:8188` | ComfyUI base URL |
| `OWUI_BASE_URL` | Yes* | `http://localhost:8080` | Open WebUI base URL |
| `OWUI_API_TOKEN` | Yes* | - | OWUI API bearer token |
| `HOST` | No | `0.0.0.0` | Server bind host |
| `PORT` | No | `8000` | Server bind port |

*Required only if using Open WebUI integration (`upload_results_to_openwebui=true`)

### MCP Content Types

The server returns standard MCP content blocks:

**TextContent:**
```json
{
  "type": "text",
  "text": "Summary of operation..."
}
```

**ImageContent** (optional, when `inline_preview=true`):
```json
{
  "type": "image",
  "data": "data:image/png;base64,...",
  "mimeType": "image/png"
}
```

**EmbeddedResource** (file links):
```json
{
  "type": "resource",
  "resource": {
    "uri": "https://webui.example.com/api/v1/files/abc123/content",
    "mimeType": "image/png",
    "text": "Generated image 1: filename.png"
  }
}
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (when available)
pytest tests/
```

### Code Formatting

```bash
# Format with black
black comfy_qwen_mcp/

# Lint with ruff
ruff check comfy_qwen_mcp/
```

### Adding New Tools

1. Define Pydantic input model
2. Create tool function with `@mcp.tool()` decorator
3. Load and patch appropriate workflow
4. Queue, monitor, and collect results
5. Build and return MCP content blocks

Example skeleton:

```python
class MyToolInput(BaseModel):
    param1: str
    param2: int = 10

@mcp.tool()
async def my_tool(param1: str, param2: int = 10) -> list:
    """Tool description for MCP clients."""
    params = MyToolInput(param1=param1, param2=param2)

    # Load workflow
    workflow = load_workflow("my_workflow.json")

    # Patch workflow
    workflow = patch_my_workflow(workflow, params)

    # Execute
    prompt_id = await comfy_client.queue_prompt(workflow)
    async for progress in comfy_client.progress(prompt_id):
        logger.info(f"Progress: {progress}%")

    # Collect and process outputs
    outputs = await comfy_client.collect_output_files(prompt_id)

    # Build content blocks
    return [
        TextContent(type="text", text="Result summary"),
        # ... more blocks
    ]
```

## Security Considerations

- **SSRF Protection:** Only OWUI URLs are auto-authenticated; external URLs require explicit allowlist
- **Input Validation:** All parameters validated via Pydantic models
- **File Size Limits:** 30 MB default max upload size
- **MIME Type Checks:** Verifies content is actually an image
- **Token Security:** Bearer tokens should be kept secret; use env vars, not hardcoded

## License

[Specify your license here]

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code passes linting (black, ruff)
5. Submit a pull request

## Support

For issues and questions:

- GitHub Issues: [Your repo URL]
- Documentation: This README
- ComfyUI Docs: https://github.com/comfyanonymous/ComfyUI
- MCP Spec: https://spec.modelcontextprotocol.io/

## Changelog

### v0.1.0 (Initial Release)

- MCP server with Streamable HTTP transport
- `qwen_image` and `qwen_image_edit` tools
- Open WebUI integration with file uploads
- Pre-model filter for non-vision models
- Progress streaming via WebSocket and polling
- Comprehensive error handling and logging
