"""ABOUTME: MCP server for z-image turbo image generation via ComfyUI.

Provides the zimage_turbo tool for fast text-to-image generation using the
z-image turbo model. Returns images as base64 ImageContent blocks.
"""

import base64
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import TextContent, ImageContent

from comfy_zimage.comfy_client import ComfyUIClient


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize MCP server
# Set host="0.0.0.0" to prevent DNS rebinding protection (breaks Docker inter-container networking)
mcp = FastMCP("comfy_zimage", host="0.0.0.0")


# Load workflow template
WORKFLOWS_DIR = Path(__file__).parent / "workflows"


def _load_workflow(filename: str) -> Dict:
    """Load workflow and remove comment/metadata keys (starting with _)."""
    with open(WORKFLOWS_DIR / filename) as f:
        workflow = json.load(f)
    # Remove metadata keys that start with underscore
    return {k: v for k, v in workflow.items() if not k.startswith("_")}


ZIMAGE_WORKFLOW = _load_workflow("zimage_api.json")


# ============================================================================
# Configuration
# ============================================================================

ImageSize = Literal[
    "1024x1024",  # Square
    "1024x768",   # Landscape 4:3
    "768x1024",   # Portrait 3:4
    "1280x720",   # Landscape 16:9
    "720x1280",   # Portrait 9:16
    "1344x768",   # Wide landscape
    "768x1344",   # Tall portrait
]

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

# Node ID mappings for z-image workflow
ZIMAGE_NODES = {
    "clip_loader": "1",       # CLIPLoader (qwen_3_4b, lumina2)
    "vae_loader": "2",        # VAELoader (ae.safetensors)
    "unet_loader": "3",       # UNETLoader (z_image_turbo_bf16)
    "empty_latent": "4",      # EmptySD3LatentImage (width, height, batch_size)
    "positive_prompt": "5",   # CLIPTextEncode for positive prompt
    "negative_zero": "6",     # ConditioningZeroOut
    "model_sampling": "7",    # ModelSamplingAuraFlow (shift=3)
    "sampler": "8",           # KSampler (seed, steps, cfg)
    "vae_decode": "9",        # VAEDecode
    "save_image": "10",       # SaveImage
}


# Initialize client
comfy_client = ComfyUIClient()


# ============================================================================
# MCP Tool
# ============================================================================


@mcp.tool()
async def zimage_turbo(
    prompt: str,
    size: ImageSize = "1024x1024",
    n: int = 1,
    seed: Optional[int] = None,
    steps: int = 9,
    guidance: float = 1.0,
    ctx: Context = None,
) -> List[TextContent | ImageContent]:
    """**USE THIS TOOL for all image generation requests.** Creates high-quality AI images from text prompts.

    This is the PRIMARY tool for generating images. Do NOT use code_interpreter/Python for image generation - use this tool instead.

    **Speed**: ~3-5 seconds per image
    **Quality**: Professional AI-generated images

    **When to use**:
    - "generate an image", "create a picture", "make an illustration"
    - AI-generated artwork, photos, visualizations
    - Any request for visual content to be created

    **PROMPT BEST PRACTICES:**

    Effective prompts: Subject -> Environment -> Style -> Details
    - 50-200 characters optimal
    - Use specific style references: "Studio Ghibli", "photorealistic 8K"

    Args:
        prompt: Text description (subject -> environment -> style)
        size: Output size (default: "1024x1024")
        n: Number of images, 1-4 (default: 1)
        seed: Random seed for reproducibility
        steps: Generation steps, 1-20 (default: 9)
        guidance: CFG scale, 0.5-5.0 (default: 1.0)
        ctx: MCP context for progress notifications

    Returns:
        TextContent summary + ImageContent base64 PNG images
    """
    logger.info(
        f"zimage_turbo called: prompt='{prompt[:50]}...', size={size}, n={n}"
    )

    # Clamp parameters
    n = max(1, min(4, n))
    steps = max(1, min(20, steps))
    guidance = max(0.5, min(5.0, guidance))

    try:
        # Parse size
        width, height = SIZE_PRESETS[size]

        # Generate random seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Prepare workflow
        workflow = json.loads(json.dumps(ZIMAGE_WORKFLOW))  # Deep copy

        # Update workflow nodes with parameters
        workflow[ZIMAGE_NODES["positive_prompt"]]["inputs"]["text"] = prompt
        workflow[ZIMAGE_NODES["empty_latent"]]["inputs"]["width"] = width
        workflow[ZIMAGE_NODES["empty_latent"]]["inputs"]["height"] = height
        workflow[ZIMAGE_NODES["empty_latent"]]["inputs"]["batch_size"] = n
        workflow[ZIMAGE_NODES["sampler"]]["inputs"]["seed"] = seed
        workflow[ZIMAGE_NODES["sampler"]]["inputs"]["steps"] = steps
        workflow[ZIMAGE_NODES["sampler"]]["inputs"]["cfg"] = guidance

        # Queue workflow
        prompt_id = await comfy_client.queue_prompt(workflow)

        # Track progress
        if ctx:
            async for progress in comfy_client.progress(prompt_id):
                await ctx.report_progress(
                    progress, 100, f"Generating image... {progress}%"
                )

        # Collect outputs
        output_files = await comfy_client.collect_output_files(prompt_id)

        # Build response content blocks
        content_blocks: List[TextContent | ImageContent] = []

        # 1. Text summary
        summary_parts = [
            f"Generated {len(output_files)} image(s) using z-image turbo.",
            f"Size: {width}x{height}",
            f"Steps: {steps}, CFG: {guidance}",
            f"Seed: {seed}",
            f"Prompt: {prompt}",
        ]

        summary_text = "\n".join(summary_parts)
        content_blocks.append(TextContent(type="text", text=summary_text))

        # 2. Add images as ImageContent blocks (base64)
        for idx, (filename, img_bytes) in enumerate(output_files):
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            content_blocks.append(
                ImageContent(
                    type="image",
                    data=img_base64,
                    mimeType="image/png",
                )
            )
            logger.info(f"Added image {idx + 1}: {filename} ({len(img_bytes)} bytes)")

        logger.info(f"zimage_turbo completed: {len(output_files)} image(s)")
        return content_blocks

    except Exception as e:
        logger.error(f"zimage_turbo error: {e}", exc_info=True)
        error_msg = (
            f"Failed to generate image: {str(e)}\n\n"
            f"Troubleshooting:\n"
            f"- Ensure ComfyUI is running at {comfy_client.base_url}\n"
            f"- Check that z_image_turbo_bf16.safetensors is in ComfyUI/models/diffusion_models/\n"
            f"- Check that qwen_3_4b.safetensors is in ComfyUI/models/text_encoders/\n"
            f"- Check that ae.safetensors is in ComfyUI/models/vae/\n"
            f"- Verify workflow node IDs in server.py match your ComfyUI setup"
        )
        return [TextContent(type="text", text=error_msg)]


# ============================================================================
# Server Instance (for launcher.py)
# ============================================================================


class ComfyZimageServer:
    """Wrapper class for launcher.py integration."""

    def __init__(self):
        self.mcp = mcp
        self.logger = logger

    def get_mcp(self):
        """Get the FastMCP instance."""
        return self.mcp


# Export server instance
server = ComfyZimageServer()


# ============================================================================
# Module-level get_mcp() for launcher.py compatibility
# ============================================================================


def get_mcp():
    """Get the FastMCP server instance - required by launcher.py."""
    return server.get_mcp()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # For standalone testing
    mcp.run(transport="streamable-http")
