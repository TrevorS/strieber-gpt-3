"""ABOUTME: MCP server for ComfyUI Qwen workflows (text-to-image and image editing).

Exposes two tools via MCP:
- qwen_image: Generate images from text prompts
- qwen_image_edit: Edit/modify existing images with prompts

Returns results as MCP content blocks (TextContent + ResourceLink) for compatibility
with non-vision models in Open WebUI.
"""

import asyncio
import json
import logging
import os
import random
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict, Any

from PIL import Image
from pydantic import BaseModel, Field, HttpUrl

# MCP SDK imports
from mcp.server.fastmcp import FastMCP, Context
from mcp.types import TextContent, ImageContent, EmbeddedResource

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.mcp_base import MCPServerBase, setup_logging
from comfy_qwen.comfy_client import ComfyUIClient, create_comfy_client
from comfy_qwen.owui_client import OpenWebUIClient, create_owui_client

logger = setup_logging(__name__)

# Node IDs for workflow JSON manipulation
# These map to the node IDs in workflows/*.json files

# Text-to-image workflow nodes
TXT2IMG_NODE_POSITIVE_PROMPT = "3"
TXT2IMG_NODE_NEGATIVE_PROMPT = "4"
TXT2IMG_NODE_LATENT_SIZE = "2"
TXT2IMG_NODE_SAMPLER = "5"

# Image-edit workflow nodes
EDIT_NODE_LOAD_IMAGE = "2"
EDIT_NODE_POSITIVE_PROMPT = "4"
EDIT_NODE_NEGATIVE_PROMPT = "5"
EDIT_NODE_SAMPLER = "6"
EDIT_NODE_LOAD_MASK = "9"


class QwenImageInput(BaseModel):
    """Input parameters for qwen_image tool (text-to-image)."""

    prompt: str = Field(..., description="Text prompt describing the image to generate")
    negative_prompt: str = Field(
        default="",
        description="Negative prompt (what to avoid in the image)"
    )
    width: int = Field(default=1024, ge=512, le=2048, description="Image width in pixels")
    height: int = Field(default=1024, ge=512, le=2048, description="Image height in pixels")
    steps: int = Field(default=20, ge=1, le=100, description="Number of sampling steps")
    guidance: float = Field(default=5.0, ge=1.0, le=20.0, description="Guidance scale (CFG)")
    seed: Optional[int] = Field(default=None, description="Random seed (for reproducibility)")
    batch_size: int = Field(default=1, ge=1, le=4, description="Number of images to generate")
    inline_preview: bool = Field(
        default=False,
        description="Include inline image preview (thumbnail). Default false for non-vision model compatibility"
    )
    upload_results_to_openwebui: bool = Field(
        default=True,
        description="Upload results to Open WebUI and return OWUI-hosted links"
    )


class QwenImageEditInput(BaseModel):
    """Input parameters for qwen_image_edit tool (img2img/inpaint)."""

    prompt: Optional[str] = Field(
        default=None,
        description="Text prompt describing changes to make"
    )
    init_image_file_id: Optional[str] = Field(
        default=None,
        description="Open WebUI file ID of the init image"
    )
    init_image_url: Optional[HttpUrl] = Field(
        default=None,
        description="URL of the init image (OWUI-hosted only)"
    )
    mask_file_id: Optional[str] = Field(
        default=None,
        description="Open WebUI file ID of the mask image (for inpainting)"
    )
    mask_image_url: Optional[HttpUrl] = Field(
        default=None,
        description="URL of the mask image (OWUI-hosted only)"
    )
    strength: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Denoising strength (0=no change, 1=full regeneration)"
    )
    steps: int = Field(default=30, ge=1, le=100, description="Number of sampling steps")
    guidance: float = Field(default=5.0, ge=1.0, le=20.0, description="Guidance scale (CFG)")
    seed: Optional[int] = Field(default=None, description="Random seed")
    inline_preview: bool = Field(
        default=False,
        description="Include inline image preview (thumbnail)"
    )
    upload_results_to_openwebui: bool = Field(
        default=True,
        description="Upload results to Open WebUI and return OWUI-hosted links"
    )


class ComfyQwenServer(MCPServerBase):
    """MCP server for ComfyUI Qwen image generation and editing."""

    def __init__(self):
        """Initialize the Comfy Qwen MCP server."""
        super().__init__("comfy_qwen")

        # Load workflow templates
        workflows_dir = Path(__file__).parent / "workflows"
        with open(workflows_dir / "qwen_image_api.json") as f:
            self.txt2img_workflow = json.load(f)
        with open(workflows_dir / "qwen_edit_api.json") as f:
            self.edit_workflow = json.load(f)

        # Remove metadata
        self.txt2img_workflow.pop("_meta", None)
        self.edit_workflow.pop("_meta", None)

        logger.info("ComfyQwen MCP server initialized")



# Instantiate server
server = ComfyQwenServer()
mcp = server.get_mcp()


def _create_thumbnail(image_bytes: bytes, max_size: int = 512) -> str:
    """Create a base64-encoded thumbnail for inline preview.

    Args:
        image_bytes: Original image bytes
        max_size: Maximum dimension for thumbnail

    Returns:
        Base64-encoded PNG data URL
    """
    import base64

    img = Image.open(BytesIO(image_bytes))

    # Calculate thumbnail size maintaining aspect ratio
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Convert to PNG bytes
    thumb_io = BytesIO()
    img.save(thumb_io, format="PNG")
    thumb_bytes = thumb_io.getvalue()

    # Encode as base64 data URL
    b64 = base64.b64encode(thumb_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


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
    ctx: Context = None
) -> List[Any]:
    """Generate images from text prompts using Qwen Image model.

    Returns a list of MCP content blocks:
    - TextContent: Summary of generation
    - ImageContent: Optional inline thumbnail (only if inline_preview=true)
    - EmbeddedResource: Links to full-resolution images (OWUI-hosted if enabled)

    Args:
        prompt: Text description of the image to generate
        negative_prompt: Things to avoid in the image
        width: Image width in pixels (512-2048)
        height: Image height in pixels (512-2048)
        steps: Number of sampling steps (1-100)
        guidance: Guidance scale/CFG (1.0-20.0)
        seed: Random seed for reproducibility (optional)
        batch_size: Number of images to generate (1-4)
        inline_preview: Include inline thumbnail preview (default: false)
        upload_results_to_openwebui: Upload to OWUI and return OWUI links (default: true)

    Returns:
        List of MCP content blocks (TextContent, ImageContent, EmbeddedResource)
    """
    server.log_tool_start(
        "qwen_image",
        prompt=prompt[:50],
        width=width,
        height=height,
        steps=steps
    )

    try:
        # Create clients
        comfy = await create_comfy_client()
        owui = None
        if upload_results_to_openwebui:
            try:
                owui = await create_owui_client()
            except ValueError as e:
                logger.warning(f"OWUI client unavailable: {e}")
                upload_results_to_openwebui = False

        # Generate seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Prepare workflow
        workflow = json.loads(json.dumps(server.txt2img_workflow))

        # Set parameters
        workflow[TXT2IMG_NODE_POSITIVE_PROMPT]["inputs"]["text"] = prompt
        workflow[TXT2IMG_NODE_NEGATIVE_PROMPT]["inputs"]["text"] = negative_prompt
        workflow[TXT2IMG_NODE_LATENT_SIZE]["inputs"]["width"] = width
        workflow[TXT2IMG_NODE_LATENT_SIZE]["inputs"]["height"] = height
        workflow[TXT2IMG_NODE_LATENT_SIZE]["inputs"]["batch_size"] = batch_size
        workflow[TXT2IMG_NODE_SAMPLER]["inputs"]["seed"] = seed
        workflow[TXT2IMG_NODE_SAMPLER]["inputs"]["steps"] = steps
        workflow[TXT2IMG_NODE_SAMPLER]["inputs"]["cfg"] = guidance

        # Queue workflow
        prompt_id = await comfy.queue_prompt(workflow)

        # Track progress
        if ctx:
            async for progress, msg in comfy.track_progress_ws(prompt_id, prompt_id):
                await ctx.send_progress_notification(
                    progress=progress,
                    total=100,
                    progressToken=msg
                )

        # Retrieve outputs
        outputs = await comfy.get_output_images(prompt_id)

        # Build result content blocks
        content_blocks = []

        # Text summary
        summary = (
            f"Generated {len(outputs)} image(s) using Qwen Image model.\n"
            f"Prompt: {prompt}\n"
            f"Size: {width}x{height}, Steps: {steps}, Guidance: {guidance}, Seed: {seed}"
        )
        content_blocks.append(TextContent(type="text", text=summary))

        # Process each output
        for idx, (filename, image_bytes) in enumerate(outputs):
            # Upload to OWUI if enabled
            if upload_results_to_openwebui and owui:
                try:
                    file_id, content_url = await owui.upload_file(
                        image_bytes,
                        f"qwen_txt2img_{seed}_{idx}.png",
                        "image/png"
                    )

                    # Add resource link
                    content_blocks.append(
                        EmbeddedResource(
                            type="resource",
                            resource=TextContent(
                                type="text",
                                text=content_url,
                                mimeType="image/png"
                            )
                        )
                    )

                    logger.info(f"Uploaded to OWUI: {content_url}")

                except Exception as e:
                    logger.error(f"Failed to upload to OWUI: {e}")
                    # Fall back to local reference
                    content_blocks.append(
                        TextContent(
                            type="text",
                            text=f"Generated image: {filename} (upload failed: {e})"
                        )
                    )
            else:
                # No OWUI upload
                content_blocks.append(
                    TextContent(type="text", text=f"Generated image: {filename}")
                )

            # Add inline thumbnail if requested
            if inline_preview:
                try:
                    thumbnail_data = _create_thumbnail(image_bytes)
                    content_blocks.append(
                        ImageContent(type="image", data=thumbnail_data, mimeType="image/png")
                    )
                except Exception as e:
                    logger.warning(f"Failed to create thumbnail: {e}")

        server.log_tool_complete("qwen_image", images=len(outputs))
        return content_blocks

    except Exception as e:
        server.log_tool_error("qwen_image", "GENERATION_FAILED", str(e))
        return [TextContent(
            type="text",
            text=f"Image generation failed: {str(e)}\n\n"
                 f"Please check that ComfyUI is running and the Qwen model is loaded."
        )]


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
    ctx: Context = None
) -> List[Any]:
    """Edit or modify images using Qwen Image Edit model.

    Supports img2img and inpainting workflows. Returns MCP content blocks
    compatible with non-vision models (text + resource links).

    Args:
        init_image_file_id: OWUI file ID of the init image
        init_image_url: URL of the init image (OWUI-hosted only)
        prompt: Text description of changes to make
        mask_file_id: OWUI file ID of mask for inpainting (optional)
        mask_image_url: URL of mask image (optional)
        strength: Denoising strength (0=no change, 1=full regen)
        steps: Number of sampling steps (1-100)
        guidance: Guidance scale/CFG (1.0-20.0)
        seed: Random seed (optional)
        inline_preview: Include inline thumbnail (default: false)
        upload_results_to_openwebui: Upload to OWUI (default: true)

    Returns:
        List of MCP content blocks
    """
    server.log_tool_start(
        "qwen_image_edit",
        prompt=prompt[:50] if prompt else "no prompt",
        strength=strength,
        steps=steps
    )

    try:
        # Validate inputs
        if not init_image_file_id and not init_image_url:
            raise ValueError("Either init_image_file_id or init_image_url must be provided")

        # Create clients
        comfy = await create_comfy_client()
        owui = None
        if upload_results_to_openwebui or init_image_file_id or mask_file_id:
            try:
                owui = await create_owui_client()
            except ValueError as e:
                logger.warning(f"OWUI client unavailable: {e}")
                if init_image_file_id or mask_file_id:
                    raise ValueError(
                        f"OWUI client required for file_id inputs: {e}"
                    )

        # Download init image
        if init_image_file_id:
            if not owui:
                raise ValueError("OWUI client required for file_id input")
            init_bytes = await owui.download_file_content(init_image_file_id)
        else:
            if not owui:
                raise ValueError("OWUI client required for URL input")
            init_bytes = await owui.download_url(str(init_image_url))

        # Upload to ComfyUI
        init_filename = await comfy.upload_image(
            init_bytes,
            "init_image.png",
            "input"
        )

        # Handle mask if provided
        mask_filename = None
        if mask_file_id or mask_image_url:
            if mask_file_id:
                if not owui:
                    raise ValueError("OWUI client required for mask file_id")
                mask_bytes = await owui.download_file_content(mask_file_id)
            else:
                if not owui:
                    raise ValueError("OWUI client required for mask URL")
                mask_bytes = await owui.download_url(str(mask_image_url))

            mask_filename = await comfy.upload_mask(mask_bytes, "mask.png")

        # Generate seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Prepare workflow
        workflow = json.loads(json.dumps(server.edit_workflow))

        # Set parameters
        workflow[EDIT_NODE_LOAD_IMAGE]["inputs"]["image"] = init_filename
        workflow[EDIT_NODE_POSITIVE_PROMPT]["inputs"]["text"] = prompt or "high quality"
        workflow[EDIT_NODE_NEGATIVE_PROMPT]["inputs"]["text"] = "blurry, low quality"
        workflow[EDIT_NODE_SAMPLER]["inputs"]["seed"] = seed
        workflow[EDIT_NODE_SAMPLER]["inputs"]["steps"] = steps
        workflow[EDIT_NODE_SAMPLER]["inputs"]["cfg"] = guidance
        workflow[EDIT_NODE_SAMPLER]["inputs"]["denoise"] = strength

        # Add mask if provided
        if mask_filename:
            workflow[EDIT_NODE_LOAD_MASK]["inputs"]["image"] = mask_filename
            # Note: connecting mask to sampler requires workflow graph modification
            # For simplicity, this assumes mask is auto-detected by LoadImage

        # Remove mask node if not used
        if not mask_filename and EDIT_NODE_LOAD_MASK in workflow:
            del workflow[EDIT_NODE_LOAD_MASK]

        # Queue workflow
        prompt_id = await comfy.queue_prompt(workflow)

        # Track progress
        if ctx:
            async for progress, msg in comfy.track_progress_ws(prompt_id, prompt_id):
                await ctx.send_progress_notification(
                    progress=progress,
                    total=100,
                    progressToken=msg
                )

        # Retrieve outputs
        outputs = await comfy.get_output_images(prompt_id)

        # Build result content blocks
        content_blocks = []

        # Text summary
        summary = (
            f"Edited {len(outputs)} image(s) using Qwen Image Edit model.\n"
            f"Prompt: {prompt or 'None'}\n"
            f"Strength: {strength}, Steps: {steps}, Guidance: {guidance}, Seed: {seed}"
        )
        content_blocks.append(TextContent(type="text", text=summary))

        # Process each output
        for idx, (filename, image_bytes) in enumerate(outputs):
            # Upload to OWUI if enabled
            if upload_results_to_openwebui and owui:
                try:
                    file_id, content_url = await owui.upload_file(
                        image_bytes,
                        f"qwen_edit_{seed}_{idx}.png",
                        "image/png"
                    )

                    # Add resource link
                    content_blocks.append(
                        EmbeddedResource(
                            type="resource",
                            resource=TextContent(
                                type="text",
                                text=content_url,
                                mimeType="image/png"
                            )
                        )
                    )

                    logger.info(f"Uploaded to OWUI: {content_url}")

                except Exception as e:
                    logger.error(f"Failed to upload to OWUI: {e}")
                    content_blocks.append(
                        TextContent(
                            type="text",
                            text=f"Edited image: {filename} (upload failed: {e})"
                        )
                    )
            else:
                content_blocks.append(
                    TextContent(type="text", text=f"Edited image: {filename}")
                )

            # Add inline thumbnail if requested
            if inline_preview:
                try:
                    thumbnail_data = _create_thumbnail(image_bytes)
                    content_blocks.append(
                        ImageContent(type="image", data=thumbnail_data, mimeType="image/png")
                    )
                except Exception as e:
                    logger.warning(f"Failed to create thumbnail: {e}")

        server.log_tool_complete("qwen_image_edit", images=len(outputs))
        return content_blocks

    except Exception as e:
        server.log_tool_error("qwen_image_edit", "EDIT_FAILED", str(e))
        return [TextContent(
            type="text",
            text=f"Image editing failed: {str(e)}\n\n"
                 f"Please check that ComfyUI is running and the Qwen model is loaded.\n"
                 f"Ensure the init image is accessible from Open WebUI Files API."
        )]


# Entry point for direct execution
if __name__ == "__main__":
    import asyncio

    logger.info("Starting ComfyQwen MCP server (streamable-http)")
    server.run(transport="streamable-http")
