"""
MCP server for ComfyUI Qwen image generation and editing.

Provides two tools:
- qwen_image: Text-to-image generation
- qwen_image_edit: Image-to-image editing and inpainting

Uses Streamable HTTP transport and returns MCP content blocks.
"""

import base64
import json
import logging
import os
import random
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP
from mcp.types import TextContent, ImageContent, EmbeddedResource
from PIL import Image
from pydantic import BaseModel, HttpUrl, Field

from comfy_client import ComfyUIClient
from owui_client import OWUIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("comfy-qwen-mcp")

# Initialize clients (will use env config)
comfy_client = ComfyUIClient()
owui_client = OWUIClient()

# Load workflow templates
WORKFLOWS_DIR = Path(__file__).parent / "workflows"


def load_workflow(name: str) -> dict:
    """Load a workflow JSON template."""
    path = WORKFLOWS_DIR / name
    with open(path, "r") as f:
        return json.load(f)


# Tool input models
class QwenImageInput(BaseModel):
    """Input parameters for qwen_image (txt2img) tool."""

    prompt: str = Field(..., description="Text prompt describing the image to generate")
    negative_prompt: str = Field(
        default="", description="Negative prompt (what to avoid)"
    )
    width: int = Field(default=1024, ge=64, le=4096, description="Image width in pixels")
    height: int = Field(default=1024, ge=64, le=4096, description="Image height in pixels")
    steps: int = Field(default=20, ge=1, le=150, description="Number of sampling steps")
    guidance: float = Field(
        default=5.0, ge=1.0, le=30.0, description="Guidance scale (CFG)"
    )
    seed: Optional[int] = Field(default=None, description="Random seed (None for random)")
    batch_size: int = Field(default=1, ge=1, le=4, description="Number of images to generate")
    inline_preview: bool = Field(
        default=False,
        description="Include inline base64 thumbnail (only if model supports images)",
    )
    upload_results_to_openwebui: bool = Field(
        default=True, description="Upload results to Open WebUI Files API"
    )


class QwenImageEditInput(BaseModel):
    """Input parameters for qwen_image_edit (img2img/inpaint) tool."""

    prompt: Optional[str] = Field(
        default=None, description="Text prompt for edits (optional)"
    )
    init_image_file_id: Optional[str] = Field(
        default=None, description="Open WebUI file ID for init image"
    )
    init_image_url: Optional[HttpUrl] = Field(
        default=None, description="URL to init image (alternative to file_id)"
    )
    mask_file_id: Optional[str] = Field(
        default=None, description="Open WebUI file ID for mask image (optional)"
    )
    mask_image_url: Optional[HttpUrl] = Field(
        default=None, description="URL to mask image (optional)"
    )
    strength: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Denoising strength (0=no change, 1=full regeneration)",
    )
    steps: int = Field(default=30, ge=1, le=150, description="Number of sampling steps")
    guidance: float = Field(
        default=5.0, ge=1.0, le=30.0, description="Guidance scale (CFG)"
    )
    seed: Optional[int] = Field(default=None, description="Random seed (None for random)")
    inline_preview: bool = Field(
        default=False,
        description="Include inline base64 thumbnail (only if model supports images)",
    )
    upload_results_to_openwebui: bool = Field(
        default=True, description="Upload results to Open WebUI Files API"
    )


def create_thumbnail(image_bytes: bytes, max_size: int = 512) -> str:
    """Create a base64-encoded thumbnail from image bytes."""
    img = Image.open(BytesIO(image_bytes))

    # Resize to fit within max_size while maintaining aspect ratio
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{b64}"


def patch_txt2img_workflow(workflow: dict, params: QwenImageInput) -> dict:
    """
    Patch txt2img workflow with user parameters.

    Node mappings (see workflows/qwen_image_api.json):
    - Node 1: Positive prompt
    - Node 2: Negative prompt
    - Node 3: Latent size and batch
    - Node 4: Sampler settings (seed, steps, cfg)
    """
    # Generate seed if not provided
    seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)

    # Patch prompts
    workflow["1"]["inputs"]["text"] = params.prompt
    workflow["2"]["inputs"]["text"] = params.negative_prompt

    # Patch size and batch
    workflow["3"]["inputs"]["width"] = params.width
    workflow["3"]["inputs"]["height"] = params.height
    workflow["3"]["inputs"]["batch_size"] = params.batch_size

    # Patch sampler
    workflow["4"]["inputs"]["seed"] = seed
    workflow["4"]["inputs"]["steps"] = params.steps
    workflow["4"]["inputs"]["cfg"] = params.guidance

    return workflow


async def patch_edit_workflow(workflow: dict, params: QwenImageEditInput) -> dict:
    """
    Patch img2img workflow with user parameters.

    Node mappings (see workflows/qwen_edit_api.json):
    - Node 1: Positive prompt
    - Node 2: Negative prompt
    - Node 3: Load init image
    - Node 5: Sampler settings (seed, steps, cfg, denoise)
    """
    # Validate input image is provided
    if not params.init_image_file_id and not params.init_image_url:
        raise ValueError("Either init_image_file_id or init_image_url must be provided")

    # Download init image
    if params.init_image_file_id:
        logger.info(f"Downloading init image from OWUI file {params.init_image_file_id}")
        init_bytes = await owui_client.download_file_content(params.init_image_file_id)
    else:
        logger.info(f"Downloading init image from URL {params.init_image_url}")
        init_bytes = await owui_client.download_url(str(params.init_image_url))

    # Upload to ComfyUI
    init_filename = await comfy_client.upload_image(
        init_bytes, "init_image.png", kind="input"
    )

    # Handle optional mask
    if params.mask_file_id or params.mask_image_url:
        if params.mask_file_id:
            logger.info(f"Downloading mask from OWUI file {params.mask_file_id}")
            mask_bytes = await owui_client.download_file_content(params.mask_file_id)
        else:
            logger.info(f"Downloading mask from URL {params.mask_image_url}")
            mask_bytes = await owui_client.download_url(str(params.mask_image_url))

        # Upload mask to ComfyUI (note: may need special handling depending on workflow)
        await comfy_client.upload_image(mask_bytes, "mask.png", kind="input")

    # Generate seed if not provided
    seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)

    # Patch prompts (optional for editing)
    workflow["1"]["inputs"]["text"] = params.prompt or ""
    workflow["2"]["inputs"]["text"] = ""

    # Patch init image
    workflow["3"]["inputs"]["image"] = init_filename

    # Patch sampler
    workflow["5"]["inputs"]["seed"] = seed
    workflow["5"]["inputs"]["steps"] = params.steps
    workflow["5"]["inputs"]["cfg"] = params.guidance
    workflow["5"]["inputs"]["denoise"] = params.strength

    return workflow


@mcp.tool()
async def qwen_image(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    guidance: float = 5.0,
    seed: Optional[int] = None,
    batch_size: int = 1,
    inline_preview: bool = False,
    upload_results_to_openwebui: bool = True,
) -> list:
    """
    Generate images using Qwen text-to-image model.

    Args:
        prompt: Text prompt describing the image to generate
        negative_prompt: Negative prompt (what to avoid)
        width: Image width in pixels (64-4096)
        height: Image height in pixels (64-4096)
        steps: Number of sampling steps (1-150)
        guidance: Guidance scale/CFG (1.0-30.0)
        seed: Random seed (None for random)
        batch_size: Number of images to generate (1-4)
        inline_preview: Include inline base64 thumbnail (use only with vision models)
        upload_results_to_openwebui: Upload results to Open WebUI Files API

    Returns:
        List of MCP content blocks with text summary, optional inline preview,
        and resource links to generated images
    """
    params = QwenImageInput(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        seed=seed,
        batch_size=batch_size,
        inline_preview=inline_preview,
        upload_results_to_openwebui=upload_results_to_openwebui,
    )

    logger.info(f"Starting txt2img generation: {params.prompt[:50]}...")

    # Load and patch workflow
    workflow = load_workflow("qwen_image_api.json")
    workflow = patch_txt2img_workflow(workflow, params)

    # Queue workflow
    prompt_id = await comfy_client.queue_prompt(workflow)

    # Monitor progress with notifications
    # Note: In FastMCP, we can use context for progress
    # For now, we'll just monitor internally
    async for progress in comfy_client.progress(prompt_id):
        logger.info(f"Progress: {progress}%")
        # In a full MCP implementation with context:
        # await ctx.send_progress_notification(progress, 100, f"Generating: {progress}%")

    # Collect outputs
    output_files = await comfy_client.collect_output_files(prompt_id)

    logger.info(f"Generated {len(output_files)} images")

    # Build content blocks
    content_blocks = []

    # Text summary
    summary = (
        f"Generated {len(output_files)} image(s) with prompt: '{params.prompt}'\n"
        f"Settings: {params.width}x{params.height}, {params.steps} steps, "
        f"guidance={params.guidance}, seed={params.seed or 'random'}"
    )
    content_blocks.append(TextContent(type="text", text=summary))

    # Process each output
    for idx, (filename, file_bytes) in enumerate(output_files):
        # Upload to OWUI if requested
        if upload_results_to_openwebui:
            file_id, content_url = await owui_client.upload_file(
                file_bytes,
                f"qwen_txt2img_{idx}_{filename}",
                mime_type="image/png",
            )

            # Add resource link
            content_blocks.append(
                EmbeddedResource(
                    type="resource",
                    resource={
                        "uri": content_url,
                        "mimeType": "image/png",
                        "text": f"Generated image {idx + 1}: {filename}",
                    },
                )
            )

        # Optionally add inline preview
        if inline_preview:
            thumbnail_b64 = create_thumbnail(file_bytes)
            content_blocks.append(
                ImageContent(
                    type="image",
                    data=thumbnail_b64,
                    mimeType="image/png",
                )
            )

    logger.info(f"Completed txt2img generation with {len(content_blocks)} content blocks")
    return content_blocks


@mcp.tool()
async def qwen_image_edit(
    init_image_file_id: Optional[str] = None,
    init_image_url: Optional[str] = None,
    prompt: Optional[str] = None,
    mask_file_id: Optional[str] = None,
    mask_image_url: Optional[str] = None,
    strength: float = 0.7,
    steps: int = 30,
    guidance: float = 5.0,
    seed: Optional[int] = None,
    inline_preview: bool = False,
    upload_results_to_openwebui: bool = True,
) -> list:
    """
    Edit images using Qwen image-to-image model.

    Args:
        init_image_file_id: Open WebUI file ID for init image
        init_image_url: URL to init image (alternative to file_id)
        prompt: Text prompt for edits (optional)
        mask_file_id: Open WebUI file ID for mask image (optional, for inpainting)
        mask_image_url: URL to mask image (optional, for inpainting)
        strength: Denoising strength (0=no change, 1=full regeneration)
        steps: Number of sampling steps (1-150)
        guidance: Guidance scale/CFG (1.0-30.0)
        seed: Random seed (None for random)
        inline_preview: Include inline base64 thumbnail (use only with vision models)
        upload_results_to_openwebui: Upload results to Open WebUI Files API

    Returns:
        List of MCP content blocks with text summary, optional inline preview,
        and resource links to edited images
    """
    params = QwenImageEditInput(
        prompt=prompt,
        init_image_file_id=init_image_file_id,
        init_image_url=init_image_url,
        mask_file_id=mask_file_id,
        mask_image_url=mask_image_url,
        strength=strength,
        steps=steps,
        guidance=guidance,
        seed=seed,
        inline_preview=inline_preview,
        upload_results_to_openwebui=upload_results_to_openwebui,
    )

    logger.info(f"Starting img2img edit with strength={params.strength}")

    # Load and patch workflow
    workflow = load_workflow("qwen_edit_api.json")
    workflow = await patch_edit_workflow(workflow, params)

    # Queue workflow
    prompt_id = await comfy_client.queue_prompt(workflow)

    # Monitor progress
    async for progress in comfy_client.progress(prompt_id):
        logger.info(f"Progress: {progress}%")
        # In a full MCP implementation with context:
        # await ctx.send_progress_notification(progress, 100, f"Editing: {progress}%")

    # Collect outputs
    output_files = await comfy_client.collect_output_files(prompt_id)

    logger.info(f"Edited {len(output_files)} images")

    # Build content blocks
    content_blocks = []

    # Text summary
    summary = (
        f"Edited {len(output_files)} image(s)"
        + (f" with prompt: '{params.prompt}'" if params.prompt else "")
        + f"\nSettings: strength={params.strength}, {params.steps} steps, "
        f"guidance={params.guidance}, seed={params.seed or 'random'}"
    )
    content_blocks.append(TextContent(type="text", text=summary))

    # Process each output
    for idx, (filename, file_bytes) in enumerate(output_files):
        # Upload to OWUI if requested
        if upload_results_to_openwebui:
            file_id, content_url = await owui_client.upload_file(
                file_bytes,
                f"qwen_edit_{idx}_{filename}",
                mime_type="image/png",
            )

            # Add resource link
            content_blocks.append(
                EmbeddedResource(
                    type="resource",
                    resource={
                        "uri": content_url,
                        "mimeType": "image/png",
                        "text": f"Edited image {idx + 1}: {filename}",
                    },
                )
            )

        # Optionally add inline preview
        if inline_preview:
            thumbnail_b64 = create_thumbnail(file_bytes)
            content_blocks.append(
                ImageContent(
                    type="image",
                    data=thumbnail_b64,
                    mimeType="image/png",
                )
            )

    logger.info(f"Completed img2img edit with {len(content_blocks)} content blocks")
    return content_blocks


# Main entry point for Streamable HTTP
if __name__ == "__main__":
    import uvicorn

    # Get config from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    logger.info(f"Starting MCP server on {host}:{port}")
    logger.info(f"ComfyUI URL: {comfy_client.base_url}")
    logger.info(f"OWUI URL: {owui_client.base_url}")

    # Run with streamable HTTP transport
    # FastMCP provides the ASGI app
    uvicorn.run(
        mcp.get_asgi_app(),
        host=host,
        port=port,
        log_level="info",
    )
