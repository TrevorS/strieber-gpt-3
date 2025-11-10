"""ABOUTME: MCP server for ComfyUI Qwen Image generation workflows.

Provides two tools:
- qwen_image: Text-to-image generation using Qwen models
- qwen_image_edit: Image editing/inpainting using Qwen models

Both tools support:
- Progress streaming via MCP notifications
- Automatic upload to Open WebUI Files
- Optional inline image previews
- Proper MCP content blocks (TextContent, ResourceLink, ImageContent)
"""

import asyncio
import base64
import json
import logging
import os
import random
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ImageContent, ResourceLink, TextContent
from PIL import Image
from pydantic import BaseModel, Field, HttpUrl

# Import our clients
from comfy_qwen.comfy_client import ComfyUIClient
from comfy_qwen.owui_client import OpenWebUIClient


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize MCP server
mcp = FastMCP("comfy_qwen")


# Load workflow templates
WORKFLOWS_DIR = Path(__file__).parent / "workflows"

with open(WORKFLOWS_DIR / "qwen_image_api.json") as f:
    QWEN_IMAGE_WORKFLOW = json.load(f)

with open(WORKFLOWS_DIR / "qwen_edit_api.json") as f:
    QWEN_EDIT_WORKFLOW = json.load(f)


# Node ID mappings for qwen_image workflow
# Update these to match your actual ComfyUI workflow node IDs
QWEN_IMAGE_NODES = {
    "positive_prompt": "2",  # CLIPTextEncode node for positive prompt
    "negative_prompt": "3",  # CLIPTextEncode node for negative prompt
    "empty_latent": "4",     # EmptyLatentImage node (width, height, batch_size)
    "sampler": "5",          # KSampler node (seed, steps, cfg)
}

# Node ID mappings for qwen_edit workflow
# Update these to match your actual ComfyUI workflow node IDs
QWEN_EDIT_NODES = {
    "load_image": "2",       # LoadImage node
    "positive_prompt": "3",  # CLIPTextEncode node for positive prompt
    "negative_prompt": "4",  # CLIPTextEncode node for negative prompt
    "sampler": "6",          # KSampler node (seed, steps, cfg, denoise)
}


# Initialize clients
comfy_client = ComfyUIClient()
owui_client = OpenWebUIClient()


# ============================================================================
# Pydantic Models for Tool Inputs
# ============================================================================

class QwenImageInput(BaseModel):
    """Input schema for qwen_image tool."""

    prompt: str = Field(..., description="Positive prompt describing the desired image")
    negative_prompt: str = Field("", description="Negative prompt (things to avoid)")
    width: int = Field(1024, ge=512, le=2048, description="Image width in pixels")
    height: int = Field(1024, ge=512, le=2048, description="Image height in pixels")
    steps: int = Field(20, ge=1, le=150, description="Number of sampling steps")
    guidance: float = Field(5.0, ge=1.0, le=30.0, description="Guidance scale (CFG)")
    seed: Optional[int] = Field(None, description="Random seed (None for random)")
    batch_size: int = Field(1, ge=1, le=4, description="Number of images to generate")
    inline_preview: bool = Field(False, description="Include small inline image preview")
    upload_results_to_openwebui: bool = Field(True, description="Upload results to Open WebUI Files")


class QwenImageEditInput(BaseModel):
    """Input schema for qwen_image_edit tool."""

    prompt: Optional[str] = Field(None, description="Editing prompt (what to change)")
    init_image_file_id: Optional[str] = Field(None, description="Open WebUI file ID for init image")
    init_image_url: Optional[HttpUrl] = Field(None, description="URL to init image")
    mask_file_id: Optional[str] = Field(None, description="Open WebUI file ID for mask")
    mask_image_url: Optional[HttpUrl] = Field(None, description="URL to mask image")
    strength: float = Field(0.7, ge=0.0, le=1.0, description="Denoising strength (0=no change, 1=full)")
    steps: int = Field(30, ge=1, le=150, description="Number of sampling steps")
    guidance: float = Field(5.0, ge=1.0, le=30.0, description="Guidance scale (CFG)")
    seed: Optional[int] = Field(None, description="Random seed (None for random)")
    inline_preview: bool = Field(False, description="Include small inline image preview")
    upload_results_to_openwebui: bool = Field(True, description="Upload results to Open WebUI Files")


# ============================================================================
# Helper Functions
# ============================================================================

def create_thumbnail(image_bytes: bytes, max_size: int = 512) -> str:
    """Create a base64-encoded thumbnail from image bytes.

    Args:
        image_bytes: Raw image bytes
        max_size: Maximum dimension for thumbnail

    Returns:
        Base64-encoded thumbnail data URL
    """
    img = Image.open(BytesIO(image_bytes))

    # Resize maintaining aspect ratio
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Convert to PNG and encode
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{b64_data}"


async def download_init_image(
    file_id: Optional[str],
    url: Optional[HttpUrl],
) -> bytes:
    """Download init image from file ID or URL.

    Args:
        file_id: Open WebUI file ID
        url: Image URL

    Returns:
        Raw image bytes

    Raises:
        ValueError: If neither or both sources provided, or download fails
    """
    if file_id and url:
        raise ValueError("Provide either init_image_file_id OR init_image_url, not both")

    if not file_id and not url:
        raise ValueError("Must provide either init_image_file_id or init_image_url")

    if file_id:
        logger.info(f"Downloading init image from OWUI file: {file_id}")
        return await owui_client.download_file_content(file_id)

    if url:
        logger.info(f"Downloading init image from URL: {url}")
        return await owui_client.download_url(str(url))


# ============================================================================
# MCP Tools
# ============================================================================

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
    ctx: Context = None,
) -> List:
    """Generate images from text using Qwen Image models (txt2img).

    This tool creates new images from text descriptions using ComfyUI's Qwen
    workflow. Results are uploaded to Open WebUI Files and returned as resource
    links by default, making them compatible with non-vision models.

    Args:
        prompt: Positive prompt describing the desired image
        negative_prompt: Negative prompt (things to avoid)
        width: Image width in pixels (512-2048)
        height: Image height in pixels (512-2048)
        steps: Number of sampling steps (1-150)
        guidance: Guidance scale/CFG (1.0-30.0)
        seed: Random seed (None for random)
        batch_size: Number of images to generate (1-4)
        inline_preview: Include small inline image preview (default: False)
        upload_results_to_openwebui: Upload to OWUI Files (default: True)
        ctx: MCP context for progress notifications

    Returns:
        List of MCP content blocks (TextContent, optional ImageContent, ResourceLinks)

    Example:
        {
          "prompt": "cinematic photo of an astronaut on the moon",
          "negative_prompt": "blurry, low quality",
          "width": 768,
          "height": 1024,
          "steps": 28,
          "guidance": 4.5,
          "seed": 12345
        }
    """
    logger.info(f"qwen_image called: prompt='{prompt[:50]}...', size={width}x{height}")

    try:
        # Generate random seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Prepare workflow
        workflow = json.loads(json.dumps(QWEN_IMAGE_WORKFLOW))  # Deep copy

        # Update workflow nodes with parameters
        workflow[QWEN_IMAGE_NODES["positive_prompt"]]["inputs"]["text"] = prompt
        workflow[QWEN_IMAGE_NODES["negative_prompt"]]["inputs"]["text"] = negative_prompt
        workflow[QWEN_IMAGE_NODES["empty_latent"]]["inputs"]["width"] = width
        workflow[QWEN_IMAGE_NODES["empty_latent"]]["inputs"]["height"] = height
        workflow[QWEN_IMAGE_NODES["empty_latent"]]["inputs"]["batch_size"] = batch_size
        workflow[QWEN_IMAGE_NODES["sampler"]]["inputs"]["seed"] = seed
        workflow[QWEN_IMAGE_NODES["sampler"]]["inputs"]["steps"] = steps
        workflow[QWEN_IMAGE_NODES["sampler"]]["inputs"]["cfg"] = guidance

        # Queue workflow
        prompt_id = await comfy_client.queue_prompt(workflow)

        # Track progress
        if ctx:
            async for progress in comfy_client.progress(prompt_id):
                await ctx.send_progress_notification(
                    progress=progress,
                    total=100,
                    message=f"Generating image... {progress}%"
                )

        # Collect outputs
        output_files = await comfy_client.collect_output_files(prompt_id)

        # Build response content blocks
        content_blocks = []

        # 1. Text summary
        summary_text = (
            f"Generated {len(output_files)} image(s) using Qwen Image.\n"
            f"Prompt: {prompt}\n"
            f"Size: {width}x{height}, Steps: {steps}, Guidance: {guidance}, Seed: {seed}"
        )
        content_blocks.append(TextContent(type="text", text=summary_text))

        # 2. Upload to OWUI and create resource links
        if upload_results_to_openwebui:
            for idx, (filename, img_bytes) in enumerate(output_files):
                try:
                    file_id, content_url = await owui_client.upload_file(
                        img_bytes,
                        f"qwen_image_{prompt_id}_{idx}.png",
                        "image/png",
                    )

                    # Add resource link
                    content_blocks.append(
                        ResourceLink(
                            type="resource",
                            resource={
                                "uri": content_url,
                                "mimeType": "image/png",
                                "name": f"Image {idx + 1}",
                            }
                        )
                    )

                    # Optional: add inline preview
                    if inline_preview:
                        thumbnail = create_thumbnail(img_bytes)
                        content_blocks.append(
                            ImageContent(
                                type="image",
                                data=thumbnail,
                                mimeType="image/png",
                            )
                        )

                except Exception as e:
                    logger.error(f"Failed to upload image {idx}: {e}")
                    content_blocks.append(
                        TextContent(
                            type="text",
                            text=f"⚠️ Failed to upload image {idx + 1}: {str(e)}"
                        )
                    )
        else:
            # Just report success without uploading
            content_blocks.append(
                TextContent(
                    type="text",
                    text=f"✓ Generated {len(output_files)} image(s) (not uploaded to OWUI)"
                )
            )

        logger.info(f"qwen_image completed: {len(output_files)} image(s)")
        return content_blocks

    except Exception as e:
        logger.error(f"qwen_image error: {e}", exc_info=True)
        error_msg = (
            f"Failed to generate image: {str(e)}\n\n"
            f"Troubleshooting:\n"
            f"- Ensure ComfyUI is running at {comfy_client.base_url}\n"
            f"- Check that the Qwen model is loaded in ComfyUI\n"
            f"- Verify workflow node IDs in server.py match your ComfyUI setup"
        )
        return [TextContent(type="text", text=error_msg)]


@mcp.tool()
async def qwen_image_edit(
    prompt: Optional[str] = None,
    init_image_file_id: Optional[str] = None,
    init_image_url: Optional[HttpUrl] = None,
    mask_file_id: Optional[str] = None,
    mask_image_url: Optional[HttpUrl] = None,
    strength: float = 0.7,
    steps: int = 30,
    guidance: float = 5.0,
    seed: Optional[int] = None,
    inline_preview: bool = False,
    upload_results_to_openwebui: bool = True,
    ctx: Context = None,
) -> List:
    """Edit images using Qwen Image Edit (img2img/inpaint).

    This tool modifies existing images using text prompts and optional masks.
    Accepts images via OWUI file IDs or URLs. Results are uploaded to Open WebUI
    Files and returned as resource links by default.

    Args:
        prompt: Editing instruction (what to change)
        init_image_file_id: OWUI file ID for the base image
        init_image_url: URL to the base image (OWUI URLs only for security)
        mask_file_id: OWUI file ID for mask (optional, for inpainting)
        mask_image_url: URL to mask image (optional)
        strength: Denoising strength (0.0=no change, 1.0=full regeneration)
        steps: Number of sampling steps (1-150)
        guidance: Guidance scale/CFG (1.0-30.0)
        seed: Random seed (None for random)
        inline_preview: Include small inline image preview (default: False)
        upload_results_to_openwebui: Upload to OWUI Files (default: True)
        ctx: MCP context for progress notifications

    Returns:
        List of MCP content blocks (TextContent, optional ImageContent, ResourceLinks)

    Example:
        {
          "prompt": "replace sky with dramatic storm clouds",
          "init_image_file_id": "ab12cd34",
          "strength": 0.65,
          "steps": 24,
          "guidance": 4.0,
          "seed": 777
        }
    """
    logger.info(f"qwen_image_edit called: prompt='{prompt}'")

    try:
        # Validate inputs
        if not init_image_file_id and not init_image_url:
            raise ValueError("Must provide either init_image_file_id or init_image_url")

        # Generate random seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Download init image
        if ctx:
            await ctx.send_progress_notification(0, 100, "Downloading init image...")

        init_image_bytes = await download_init_image(init_image_file_id, init_image_url)

        # Upload init image to ComfyUI
        if ctx:
            await ctx.send_progress_notification(10, 100, "Uploading to ComfyUI...")

        init_filename = await comfy_client.upload_image(
            init_image_bytes,
            "init_image.png",
            kind="image",
        )

        # Handle mask if provided
        mask_filename = None
        if mask_file_id or mask_image_url:
            mask_bytes = await download_init_image(mask_file_id, mask_image_url)
            mask_filename = await comfy_client.upload_image(
                mask_bytes,
                "mask.png",
                kind="mask",
            )

        # Prepare workflow
        workflow = json.loads(json.dumps(QWEN_EDIT_WORKFLOW))  # Deep copy

        # Update workflow nodes with parameters
        workflow[QWEN_EDIT_NODES["load_image"]]["inputs"]["image"] = init_filename
        workflow[QWEN_EDIT_NODES["positive_prompt"]]["inputs"]["text"] = prompt or ""
        workflow[QWEN_EDIT_NODES["sampler"]]["inputs"]["seed"] = seed
        workflow[QWEN_EDIT_NODES["sampler"]]["inputs"]["steps"] = steps
        workflow[QWEN_EDIT_NODES["sampler"]]["inputs"]["cfg"] = guidance
        workflow[QWEN_EDIT_NODES["sampler"]]["inputs"]["denoise"] = strength

        # Queue workflow
        if ctx:
            await ctx.send_progress_notification(20, 100, "Queueing workflow...")

        prompt_id = await comfy_client.queue_prompt(workflow)

        # Track progress
        if ctx:
            async for progress in comfy_client.progress(prompt_id):
                # Map 0-100 to 25-95 range (reserve 0-25 for prep, 95-100 for upload)
                scaled_progress = 25 + int(progress * 0.70)
                await ctx.send_progress_notification(
                    progress=scaled_progress,
                    total=100,
                    message=f"Editing image... {progress}%"
                )

        # Collect outputs
        output_files = await comfy_client.collect_output_files(prompt_id)

        # Build response content blocks
        content_blocks = []

        # 1. Text summary
        summary_text = (
            f"Edited {len(output_files)} image(s) using Qwen Image Edit.\n"
            f"Prompt: {prompt or '(none)'}\n"
            f"Strength: {strength}, Steps: {steps}, Guidance: {guidance}, Seed: {seed}"
        )
        content_blocks.append(TextContent(type="text", text=summary_text))

        # 2. Upload to OWUI and create resource links
        if upload_results_to_openwebui:
            if ctx:
                await ctx.send_progress_notification(95, 100, "Uploading results...")

            for idx, (filename, img_bytes) in enumerate(output_files):
                try:
                    file_id, content_url = await owui_client.upload_file(
                        img_bytes,
                        f"qwen_edit_{prompt_id}_{idx}.png",
                        "image/png",
                    )

                    # Add resource link
                    content_blocks.append(
                        ResourceLink(
                            type="resource",
                            resource={
                                "uri": content_url,
                                "mimeType": "image/png",
                                "name": f"Edited Image {idx + 1}",
                            }
                        )
                    )

                    # Optional: add inline preview
                    if inline_preview:
                        thumbnail = create_thumbnail(img_bytes)
                        content_blocks.append(
                            ImageContent(
                                type="image",
                                data=thumbnail,
                                mimeType="image/png",
                            )
                        )

                except Exception as e:
                    logger.error(f"Failed to upload edited image {idx}: {e}")
                    content_blocks.append(
                        TextContent(
                            type="text",
                            text=f"⚠️ Failed to upload image {idx + 1}: {str(e)}"
                        )
                    )
        else:
            # Just report success without uploading
            content_blocks.append(
                TextContent(
                    type="text",
                    text=f"✓ Edited {len(output_files)} image(s) (not uploaded to OWUI)"
                )
            )

        if ctx:
            await ctx.send_progress_notification(100, 100, "Complete!")

        logger.info(f"qwen_image_edit completed: {len(output_files)} image(s)")
        return content_blocks

    except Exception as e:
        logger.error(f"qwen_image_edit error: {e}", exc_info=True)
        error_msg = (
            f"Failed to edit image: {str(e)}\n\n"
            f"Troubleshooting:\n"
            f"- Ensure ComfyUI is running at {comfy_client.base_url}\n"
            f"- Check that the Qwen model is loaded in ComfyUI\n"
            f"- Verify init image source (file_id or URL)\n"
            f"- Ensure OWUI_BASE_URL and OWUI_API_TOKEN are set correctly\n"
            f"- Verify workflow node IDs in server.py match your ComfyUI setup"
        )
        return [TextContent(type="text", text=error_msg)]


# ============================================================================
# Server Instance (for launcher.py)
# ============================================================================

class ComfyQwenServer:
    """Wrapper class for launcher.py integration."""

    def __init__(self):
        self.mcp = mcp
        self.logger = logger

    def get_mcp(self):
        """Get the FastMCP instance."""
        return self.mcp


# Export server instance
server = ComfyQwenServer()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # For standalone testing
    mcp.run(transport="streamable-http")
