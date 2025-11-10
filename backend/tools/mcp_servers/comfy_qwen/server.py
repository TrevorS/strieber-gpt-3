"""ABOUTME: MCP server for ComfyUI Qwen Image generation workflows.

Provides two tools:
- qwen_image: Text-to-image generation using Qwen models
- qwen_image_edit: Image editing/inpainting using Qwen models

Both tools support:
- Quality presets (fast/standard/high) with automatic Lightning LoRA
- Progress streaming via MCP notifications
- Automatic upload to Open WebUI Files
- Proper MCP content blocks (TextContent, ResourceLink)
"""

import asyncio
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ResourceLink, TextContent
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


# ============================================================================
# Quality Presets Configuration
# ============================================================================

QualityLevel = Literal["fast", "standard", "high"]
ImageSize = Literal[
    "1024x1024",  # Square
    "1024x768",   # Landscape 4:3
    "768x1024",   # Portrait 3:4
    "1280x720",   # Landscape 16:9
    "720x1280",   # Portrait 9:16
    "1344x768",   # Wide landscape
    "768x1344",   # Tall portrait
]

# Quality presets define steps, cfg, and Lightning LoRA usage
QUALITY_PRESETS: Dict[QualityLevel, Dict] = {
    "fast": {
        "use_lightning": True,
        "lora_file": "Qwen-Image-Lightning-8steps-V1.1.safetensors",
        "steps": 8,
        "guidance": 2.5,
        "description": "8-step Lightning LoRA - fastest generation with great quality",
    },
    "standard": {
        "use_lightning": False,
        "lora_file": None,
        "steps": 20,
        "guidance": 5.0,
        "description": "20-step standard generation - balanced speed and quality",
    },
    "high": {
        "use_lightning": False,
        "lora_file": None,
        "steps": 50,
        "guidance": 5.0,
        "description": "50-step high-quality generation - maximum detail",
    },
}

# Size presets map string sizes to (width, height) tuples
SIZE_PRESETS: Dict[ImageSize, Tuple[int, int]] = {
    "1024x1024": (1024, 1024),
    "1024x768": (1024, 768),
    "768x1024": (768, 1024),
    "1280x720": (1280, 720),
    "720x1280": (720, 1280),
    "1344x768": (1344, 768),
    "768x1344": (768, 1344),
}


# ============================================================================
# Node ID Mappings
# ============================================================================

# Node ID mappings for qwen_image workflow
# Update these to match your actual ComfyUI workflow node IDs
QWEN_IMAGE_NODES = {
    "checkpoint_loader": "1",    # CheckpointLoader node
    "lora_loader": "10",         # LoraLoaderModelOnly node (NEW)
    "positive_prompt": "2",      # CLIPTextEncode node for positive prompt
    "negative_prompt": "3",      # CLIPTextEncode node for negative prompt
    "empty_latent": "4",         # EmptyLatentImage node (width, height, batch_size)
    "sampler": "5",              # KSampler node (seed, steps, cfg)
}

# Node ID mappings for qwen_edit workflow
# Update these to match your actual ComfyUI workflow node IDs
QWEN_EDIT_NODES = {
    "checkpoint_loader": "1",    # CheckpointLoader node
    "lora_loader": "10",         # LoraLoaderModelOnly node (NEW)
    "load_image": "2",           # LoadImage node
    "positive_prompt": "3",      # CLIPTextEncode node for positive prompt
    "negative_prompt": "4",      # CLIPTextEncode node for negative prompt
    "sampler": "6",              # KSampler node (seed, steps, cfg, denoise)
}


# Initialize clients
comfy_client = ComfyUIClient()
owui_client = OpenWebUIClient()


# ============================================================================
# Pydantic Models for Tool Inputs
# ============================================================================

class QwenImageInput(BaseModel):
    """Input schema for qwen_image tool.

    Provides OpenAI-style quality presets with advanced overrides for power users.
    """

    prompt: str = Field(
        ...,
        description="Text description of the image to generate"
    )
    quality: QualityLevel = Field(
        "fast",
        description="Generation quality: 'fast' (8 steps, Lightning LoRA), 'standard' (20 steps), or 'high' (50 steps)"
    )
    size: ImageSize = Field(
        "1024x1024",
        description="Output image size as WIDTHxHEIGHT (e.g., '1024x1024', '1280x720')"
    )
    n: int = Field(
        1,
        ge=1,
        le=4,
        description="Number of images to generate (1-4)"
    )
    negative_prompt: str = Field(
        "",
        description="Negative prompt describing what to avoid in the generated image"
    )
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducible generation (omit for random)"
    )

    # Advanced overrides (optional)
    steps: Optional[int] = Field(
        None,
        ge=1,
        le=150,
        description="Override number of sampling steps (advanced users only)"
    )
    guidance: Optional[float] = Field(
        None,
        ge=1.0,
        le=30.0,
        description="Override CFG/guidance scale (advanced users only)"
    )
    use_lightning: Optional[bool] = Field(
        None,
        description="Explicitly enable/disable Lightning LoRA (overrides quality preset)"
    )
    upload_results_to_openwebui: bool = Field(
        True,
        description="Upload generated images to Open WebUI Files API"
    )


class QwenImageEditInput(BaseModel):
    """Input schema for qwen_image_edit tool.

    Edit or inpaint existing images using Qwen models with quality presets.
    """

    prompt: Optional[str] = Field(
        None,
        description="Text description of the desired edit or transformation"
    )
    init_image_file_id: Optional[str] = Field(
        None,
        description="Open WebUI file ID for the base image to edit"
    )
    init_image_url: Optional[HttpUrl] = Field(
        None,
        description="URL to the base image (must be from Open WebUI domain)"
    )
    quality: QualityLevel = Field(
        "fast",
        description="Generation quality: 'fast' (8 steps, Lightning LoRA), 'standard' (30 steps), or 'high' (50 steps)"
    )
    strength: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Denoising strength: 0.0 = no change, 1.0 = complete regeneration"
    )
    mask_file_id: Optional[str] = Field(
        None,
        description="Open WebUI file ID for inpainting mask (optional)"
    )
    mask_image_url: Optional[HttpUrl] = Field(
        None,
        description="URL to inpainting mask image (optional)"
    )
    negative_prompt: str = Field(
        "",
        description="Negative prompt describing what to avoid"
    )
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducible edits (omit for random)"
    )

    # Advanced overrides (optional)
    steps: Optional[int] = Field(
        None,
        ge=1,
        le=150,
        description="Override number of sampling steps (advanced users only)"
    )
    guidance: Optional[float] = Field(
        None,
        ge=1.0,
        le=30.0,
        description="Override CFG/guidance scale (advanced users only)"
    )
    use_lightning: Optional[bool] = Field(
        None,
        description="Explicitly enable/disable Lightning LoRA (overrides quality preset)"
    )
    upload_results_to_openwebui: bool = Field(
        True,
        description="Upload edited images to Open WebUI Files API"
    )


# ============================================================================
# Helper Functions
# ============================================================================

def resolve_quality_settings(
    quality: QualityLevel,
    steps_override: Optional[int],
    guidance_override: Optional[float],
    use_lightning_override: Optional[bool],
) -> Dict:
    """Resolve quality preset with optional advanced overrides.

    Args:
        quality: Quality preset level
        steps_override: User-provided steps override
        guidance_override: User-provided guidance override
        use_lightning_override: User-provided Lightning LoRA override

    Returns:
        Dictionary with resolved settings (steps, guidance, use_lightning, lora_file)
    """
    preset = QUALITY_PRESETS[quality].copy()

    # Apply overrides
    if steps_override is not None:
        preset["steps"] = steps_override
        logger.info(f"Overriding steps: {steps_override}")

    if guidance_override is not None:
        preset["guidance"] = guidance_override
        logger.info(f"Overriding guidance: {guidance_override}")

    if use_lightning_override is not None:
        preset["use_lightning"] = use_lightning_override
        if use_lightning_override and not preset.get("lora_file"):
            # Enable Lightning LoRA even if preset doesn't have it
            preset["lora_file"] = "Qwen-Image-Lightning-8steps-V1.1.safetensors"
        logger.info(f"Overriding Lightning LoRA: {use_lightning_override}")

    return preset


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
    quality: QualityLevel = "fast",
    size: ImageSize = "1024x1024",
    n: int = 1,
    negative_prompt: str = "",
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    guidance: Optional[float] = None,
    use_lightning: Optional[bool] = None,
    upload_results_to_openwebui: bool = True,
    ctx: Context = None,
) -> List:
    """Generate images from text using Qwen Image models.

    This tool creates images from text descriptions with three quality presets:
    - 'fast': 8-step Lightning LoRA (2-3 seconds, great quality)
    - 'standard': 20-step standard generation (balanced)
    - 'high': 50-step maximum quality (slower, best detail)

    Results are uploaded to Open WebUI Files and returned as resource links,
    making them compatible with both vision and non-vision models.

    Args:
        prompt: Text description of the image to generate
        quality: Generation quality preset (default: "fast")
        size: Output image size (default: "1024x1024")
        n: Number of images to generate (default: 1)
        negative_prompt: What to avoid in the image (default: "")
        seed: Random seed for reproducibility (default: random)
        steps: Override sampling steps (advanced, default: from preset)
        guidance: Override CFG scale (advanced, default: from preset)
        use_lightning: Force Lightning LoRA on/off (advanced, default: from preset)
        upload_results_to_openwebui: Upload to OWUI Files (default: True)
        ctx: MCP context for progress notifications

    Returns:
        List of MCP content blocks:
        - TextContent: Generation summary
        - ResourceLink: Links to full-resolution images on OWUI

    Example:
        Generate a fast image:
        {
          "prompt": "sunset over mountains, dramatic lighting",
          "quality": "fast",
          "size": "1280x720"
        }

        High-quality with custom seed:
        {
          "prompt": "portrait of a wise elder",
          "quality": "high",
          "negative_prompt": "blurry, distorted",
          "seed": 42
        }
    """
    logger.info(f"qwen_image called: prompt='{prompt[:50]}...', quality={quality}, size={size}")

    try:
        # Resolve quality settings
        settings = resolve_quality_settings(quality, steps, guidance, use_lightning)
        final_steps = settings["steps"]
        final_guidance = settings["guidance"]
        use_lora = settings["use_lightning"]
        lora_file = settings.get("lora_file")

        # Parse size
        width, height = SIZE_PRESETS[size]

        # Generate random seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Prepare workflow
        workflow = json.loads(json.dumps(QWEN_IMAGE_WORKFLOW))  # Deep copy

        # Configure LoRA node
        if use_lora and lora_file:
            # Enable LoRA loader
            workflow[QWEN_IMAGE_NODES["lora_loader"]]["inputs"]["lora_name"] = lora_file
            workflow[QWEN_IMAGE_NODES["lora_loader"]]["inputs"]["strength_model"] = 1.0
            workflow[QWEN_IMAGE_NODES["lora_loader"]].pop("mode", None)  # Ensure enabled
            logger.info(f"Lightning LoRA enabled: {lora_file}")
        else:
            # Bypass LoRA loader
            workflow[QWEN_IMAGE_NODES["lora_loader"]]["mode"] = 4
            logger.info("Lightning LoRA bypassed")

        # Update workflow nodes with parameters
        workflow[QWEN_IMAGE_NODES["positive_prompt"]]["inputs"]["text"] = prompt
        workflow[QWEN_IMAGE_NODES["negative_prompt"]]["inputs"]["text"] = negative_prompt
        workflow[QWEN_IMAGE_NODES["empty_latent"]]["inputs"]["width"] = width
        workflow[QWEN_IMAGE_NODES["empty_latent"]]["inputs"]["height"] = height
        workflow[QWEN_IMAGE_NODES["empty_latent"]]["inputs"]["batch_size"] = n
        workflow[QWEN_IMAGE_NODES["sampler"]]["inputs"]["seed"] = seed
        workflow[QWEN_IMAGE_NODES["sampler"]]["inputs"]["steps"] = final_steps
        workflow[QWEN_IMAGE_NODES["sampler"]]["inputs"]["cfg"] = final_guidance

        # Queue workflow
        prompt_id = await comfy_client.queue_prompt(workflow)

        # Track progress
        if ctx:
            async for progress in comfy_client.progress(prompt_id):
                quality_label = f"quality={quality}"
                if use_lora:
                    quality_label += f" (Lightning {final_steps}-step)"
                await ctx.send_progress_notification(
                    progress=progress,
                    total=100,
                    message=f"Generating image ({quality_label})... {progress}%"
                )

        # Collect outputs
        output_files = await comfy_client.collect_output_files(prompt_id)

        # Build response content blocks
        content_blocks = []

        # 1. Text summary
        summary_parts = [
            f"Generated {len(output_files)} image(s) using Qwen Image.",
            f"Quality: {quality} ({final_steps} steps, CFG {final_guidance})",
        ]
        if use_lora:
            summary_parts.append(f"Lightning LoRA: {lora_file}")
        summary_parts.extend([
            f"Size: {width}x{height}",
            f"Seed: {seed}",
            f"Prompt: {prompt}",
        ])

        summary_text = "\n".join(summary_parts)
        content_blocks.append(TextContent(type="text", text=summary_text))

        # 2. Upload to OWUI and create resource links
        if upload_results_to_openwebui:
            for idx, (filename, img_bytes) in enumerate(output_files):
                try:
                    file_id, content_url = await owui_client.upload_file(
                        img_bytes,
                        f"qwen_image_{quality}_{prompt_id}_{idx}.png",
                        "image/png",
                    )

                    # Add resource link
                    content_blocks.append(
                        ResourceLink(
                            type="resource",
                            resource={
                                "uri": content_url,
                                "mimeType": "image/png",
                                "name": f"Image {idx + 1} ({size})",
                            }
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

        logger.info(f"qwen_image completed: {len(output_files)} image(s), quality={quality}")
        return content_blocks

    except Exception as e:
        logger.error(f"qwen_image error: {e}", exc_info=True)
        error_msg = (
            f"Failed to generate image: {str(e)}\n\n"
            f"Troubleshooting:\n"
            f"- Ensure ComfyUI is running at {comfy_client.base_url}\n"
            f"- Check that the Qwen model is loaded in ComfyUI\n"
            f"- If using Lightning LoRA, ensure {lora_file if 'lora_file' in locals() else 'LoRA file'} is in ComfyUI/models/loras/\n"
            f"- Verify workflow node IDs in server.py match your ComfyUI setup"
        )
        return [TextContent(type="text", text=error_msg)]


@mcp.tool()
async def qwen_image_edit(
    prompt: Optional[str] = None,
    init_image_file_id: Optional[str] = None,
    init_image_url: Optional[HttpUrl] = None,
    quality: QualityLevel = "fast",
    strength: float = 0.7,
    mask_file_id: Optional[str] = None,
    mask_image_url: Optional[HttpUrl] = None,
    negative_prompt: str = "",
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    guidance: Optional[float] = None,
    use_lightning: Optional[bool] = None,
    upload_results_to_openwebui: bool = True,
    ctx: Context = None,
) -> List:
    """Edit or transform images using Qwen Image Edit models.

    This tool modifies existing images using text prompts and optional masks.
    Supports three quality presets with automatic Lightning LoRA for speed:
    - 'fast': 8-step Lightning LoRA (2-3 seconds)
    - 'standard': 30-step standard editing (balanced)
    - 'high': 50-step maximum quality (best detail)

    Accepts images via OWUI file IDs or URLs. Results are uploaded to Open WebUI
    Files and returned as resource links by default.

    Args:
        prompt: Text description of the desired edit
        init_image_file_id: OWUI file ID for the base image
        init_image_url: URL to the base image (must be from OWUI domain)
        quality: Edit quality preset (default: "fast")
        strength: Denoising strength, 0.0-1.0 (default: 0.7)
        mask_file_id: OWUI file ID for inpainting mask (optional)
        mask_image_url: URL to mask image (optional)
        negative_prompt: What to avoid (default: "")
        seed: Random seed for reproducibility (default: random)
        steps: Override sampling steps (advanced, default: from preset)
        guidance: Override CFG scale (advanced, default: from preset)
        use_lightning: Force Lightning LoRA on/off (advanced, default: from preset)
        upload_results_to_openwebui: Upload to OWUI Files (default: True)
        ctx: MCP context for progress notifications

    Returns:
        List of MCP content blocks:
        - TextContent: Edit summary
        - ResourceLink: Links to edited images on OWUI

    Example:
        Fast edit with file ID:
        {
          "prompt": "make the sky more dramatic with storm clouds",
          "init_image_file_id": "abc123",
          "quality": "fast",
          "strength": 0.65
        }

        High-quality edit with URL:
        {
          "prompt": "convert to autumn colors",
          "init_image_url": "https://webui.example.com/api/v1/files/xyz789/content",
          "quality": "high",
          "strength": 0.8,
          "seed": 42
        }
    """
    logger.info(f"qwen_image_edit called: prompt='{prompt}', quality={quality}")

    try:
        # Validate inputs
        if not init_image_file_id and not init_image_url:
            raise ValueError("Must provide either init_image_file_id or init_image_url")

        # Resolve quality settings
        settings = resolve_quality_settings(quality, steps, guidance, use_lightning)
        final_steps = settings["steps"]
        final_guidance = settings["guidance"]
        use_lora = settings["use_lightning"]
        lora_file = settings.get("lora_file")

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

        # Configure LoRA node
        if use_lora and lora_file:
            # Enable LoRA loader
            workflow[QWEN_EDIT_NODES["lora_loader"]]["inputs"]["lora_name"] = lora_file
            workflow[QWEN_EDIT_NODES["lora_loader"]]["inputs"]["strength_model"] = 1.0
            workflow[QWEN_EDIT_NODES["lora_loader"]].pop("mode", None)  # Ensure enabled
            logger.info(f"Lightning LoRA enabled: {lora_file}")
        else:
            # Bypass LoRA loader
            workflow[QWEN_EDIT_NODES["lora_loader"]]["mode"] = 4
            logger.info("Lightning LoRA bypassed")

        # Update workflow nodes with parameters
        workflow[QWEN_EDIT_NODES["load_image"]]["inputs"]["image"] = init_filename
        workflow[QWEN_EDIT_NODES["positive_prompt"]]["inputs"]["text"] = prompt or ""
        workflow[QWEN_EDIT_NODES["negative_prompt"]]["inputs"]["text"] = negative_prompt
        workflow[QWEN_EDIT_NODES["sampler"]]["inputs"]["seed"] = seed
        workflow[QWEN_EDIT_NODES["sampler"]]["inputs"]["steps"] = final_steps
        workflow[QWEN_EDIT_NODES["sampler"]]["inputs"]["cfg"] = final_guidance
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
                quality_label = f"quality={quality}"
                if use_lora:
                    quality_label += f" (Lightning {final_steps}-step)"
                await ctx.send_progress_notification(
                    progress=scaled_progress,
                    total=100,
                    message=f"Editing image ({quality_label})... {progress}%"
                )

        # Collect outputs
        output_files = await comfy_client.collect_output_files(prompt_id)

        # Build response content blocks
        content_blocks = []

        # 1. Text summary
        summary_parts = [
            f"Edited {len(output_files)} image(s) using Qwen Image Edit.",
            f"Quality: {quality} ({final_steps} steps, CFG {final_guidance})",
        ]
        if use_lora:
            summary_parts.append(f"Lightning LoRA: {lora_file}")
        summary_parts.extend([
            f"Strength: {strength}",
            f"Seed: {seed}",
            f"Prompt: {prompt or '(none)'}",
        ])

        summary_text = "\n".join(summary_parts)
        content_blocks.append(TextContent(type="text", text=summary_text))

        # 2. Upload to OWUI and create resource links
        if upload_results_to_openwebui:
            if ctx:
                await ctx.send_progress_notification(95, 100, "Uploading results...")

            for idx, (filename, img_bytes) in enumerate(output_files):
                try:
                    file_id, content_url = await owui_client.upload_file(
                        img_bytes,
                        f"qwen_edit_{quality}_{prompt_id}_{idx}.png",
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

        logger.info(f"qwen_image_edit completed: {len(output_files)} image(s), quality={quality}")
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
            f"- If using Lightning LoRA, ensure {lora_file if 'lora_file' in locals() else 'LoRA file'} is in ComfyUI/models/loras/\n"
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
