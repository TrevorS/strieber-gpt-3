"""ABOUTME: MCP server for z-image turbo image generation via ComfyUI.

Provides the zimage_turbo tool for fast text-to-image generation using the
z-image turbo model. Returns images as base64 ImageContent blocks.
"""

import base64
import json
import logging
import random
import uuid
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
ZIMAGE_CONTROLNET_WORKFLOW = _load_workflow("zimage_controlnet_api.json")


# ============================================================================
# Configuration
# ============================================================================

ImageSize = Literal[
    "1024x1024",  # Square
    "1024x768",  # Landscape 4:3
    "768x1024",  # Portrait 3:4
    "1280x720",  # Landscape 16:9
    "720x1280",  # Portrait 9:16
    "1344x768",  # Wide landscape
    "768x1344",  # Tall portrait
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
    "clip_loader": "1",  # CLIPLoader (qwen_3_4b, lumina2)
    "vae_loader": "2",  # VAELoader (ae.safetensors)
    "unet_loader": "3",  # UNETLoader (z_image_turbo_bf16)
    "empty_latent": "4",  # EmptySD3LatentImage (width, height, batch_size)
    "positive_prompt": "5",  # CLIPTextEncode for positive prompt
    "negative_zero": "6",  # ConditioningZeroOut
    "model_sampling": "7",  # ModelSamplingAuraFlow (shift=3)
    "sampler": "8",  # KSampler (seed, steps, cfg)
    "vae_decode": "9",  # VAEDecode
    "save_image": "10",  # SaveImage
}

# Node ID mappings for z-image ControlNet workflow
ZIMAGE_CONTROLNET_NODES = {
    "clip_loader": "1",
    "vae_loader": "2",
    "unet_loader": "3",
    "model_patch_loader": "4",
    "load_image": "5",
    "preprocessor": "6",
    "positive_prompt": "7",
    "negative_zero": "8",
    "model_sampling": "9",
    "controlnet": "10",
    "empty_latent": "11",
    "sampler": "12",
    "vae_decode": "13",
    "save_image": "14",
}

# Map control types to AIO_Preprocessor settings
PREPROCESSOR_MAP = {
    "canny": "CannyEdgePreprocessor",
    "depth": "DepthAnythingV2Preprocessor",
    "pose": "DWPreprocessor",
    "hed": "HEDPreprocessor",
    "mlsd": "M-LSDPreprocessor",
}

ControlType = Literal["canny", "depth", "pose", "hed", "mlsd"]


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
    steps: int = 8,
    ctx: Context = None,
) -> List[TextContent | ImageContent]:
    """Generate images from text descriptions.

    ALWAYS enhance user prompts - this model excels with detailed descriptions.

    STRENGTHS:
    - Fast, high-quality text-to-image generation
    - Excellent at rendering text and typography
    - Works best with rich, descriptive prompts (100-300 words ideal)

    PROMPT ENHANCEMENT:
    Transform simple requests into detailed prompts. Describe what you WANT to see.

    Example - User says "a dog":
    Enhanced: "A golden retriever with fluffy fur, sitting in a sunlit meadow,
    wildflowers in foreground, soft bokeh background, warm afternoon light,
    photorealistic, 8K detail, shallow depth of field"

    Structure: [Subject] + [Details] + [Setting] + [Lighting] + [Style]

    Style keywords:
    - Photo: "photorealistic, 8K, hyperdetailed, DSLR quality"
    - Art: "digital painting, concept art, artstation trending"
    - Anime: "anime style, vibrant colors, clean lines"
    - Cinematic: "cinematic shot, dramatic lighting, film grain"

    SIZE GUIDE:
    - 1024x1024: Square (default, portraits, icons)
    - 1024x768 / 768x1024: Standard photo ratio
    - 1280x720 / 720x1280: Widescreen / phone wallpaper
    - 1344x768 / 768x1344: Ultra-wide / tall banners

    Args:
        prompt: Detailed image description. ALWAYS expand simple requests.
        size: Image dimensions. Default "1024x1024".
        n: Number of images (1-4). Default 1.
        seed: Random seed for reproducibility.
        steps: Denoising steps (1-20). Default 9.

    Returns:
        Generated image(s) as base64 PNG.
    """
    logger.info(f"zimage_turbo called: prompt='{prompt[:50]}...', size={size}, n={n}")

    # Clamp parameters
    n = max(1, min(4, n))
    steps = max(1, min(20, steps))

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
        workflow[ZIMAGE_NODES["sampler"]]["inputs"]["cfg"] = 1.0

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

        # 1. Text summary with all parameters
        summary_parts = [
            f"Generated {len(output_files)} image(s).",
            "Parameters:",
            f"  prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
            f"  size: {size} ({width}x{height})",
            f"  n: {n}",
            f"  steps: {steps}",
            f"  seed: {seed}",
            "  cfg: 1.0",
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


@mcp.tool()
async def zimage_controlnet(
    prompt: str,
    image_data: str,
    control_type: ControlType = "canny",
    control_strength: float = 0.75,
    size: ImageSize = "1024x1024",
    seed: Optional[int] = None,
    steps: int = 8,
    ctx: Context = None,
) -> List[TextContent | ImageContent]:
    """Transform images while preserving structure.

    Use when user wants to restyle, redraw, or transform an existing image.

    CONTROL TYPES - Choose based on what to preserve:
    - canny: Edges/outlines (faces, objects, drawings)
    - depth: Spatial layout (foreground/background arrangement)
    - pose: Human body position (generate different person, same pose)
    - hed: Soft boundaries (artistic, painterly edges)
    - mlsd: Straight lines (architecture, interiors)

    WHEN TO USE:
    - "Redraw this as anime" → canny
    - "Same pose, different person" → pose
    - "Same scene, different style" → depth

    IMAGE INPUT:
    When the user has attached an image to their message, use "attached" or "image_0"
    as the image_data value. The system will automatically inject the attached image.
    For multiple attached images, use "image_0", "image_1", etc.

    Args:
        prompt: Describe the OUTPUT style/content you want.
        image_data: Use "attached" or "image_0" for user's attached image. System injects actual data.
        control_type: What structure to preserve. Default "canny".
        control_strength: 0.65 (loose) to 0.80 (strict). Default 0.75.
        size: Output dimensions. Default "1024x1024".
        seed: Random seed for reproducibility.
        steps: Denoising steps (1-20). Default 9.

    Returns:
        Generated image(s) as base64 PNG.
    """
    logger.info(
        f"zimage_controlnet called: control_type={control_type}, "
        f"strength={control_strength}, size={size}"
    )

    # Clamp parameters
    steps = max(1, min(20, steps))
    control_strength = max(0.5, min(1.0, control_strength))

    try:
        # Parse size
        width, height = SIZE_PRESETS[size]

        # Generate random seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Upload input image to ComfyUI via API
        img_bytes = base64.b64decode(image_data)
        input_filename = await comfy_client.upload_image(
            img_bytes, f"controlnet_input_{uuid.uuid4().hex[:8]}.png"
        )

        # Prepare workflow
        workflow = json.loads(json.dumps(ZIMAGE_CONTROLNET_WORKFLOW))  # Deep copy

        # Update workflow nodes with parameters
        workflow[ZIMAGE_CONTROLNET_NODES["load_image"]]["inputs"]["image"] = (
            input_filename
        )
        workflow[ZIMAGE_CONTROLNET_NODES["preprocessor"]]["inputs"]["preprocessor"] = (
            PREPROCESSOR_MAP[control_type]
        )
        workflow[ZIMAGE_CONTROLNET_NODES["positive_prompt"]]["inputs"]["text"] = prompt
        workflow[ZIMAGE_CONTROLNET_NODES["controlnet"]]["inputs"]["strength"] = (
            control_strength
        )
        workflow[ZIMAGE_CONTROLNET_NODES["empty_latent"]]["inputs"]["width"] = width
        workflow[ZIMAGE_CONTROLNET_NODES["empty_latent"]]["inputs"]["height"] = height
        workflow[ZIMAGE_CONTROLNET_NODES["sampler"]]["inputs"]["seed"] = seed
        workflow[ZIMAGE_CONTROLNET_NODES["sampler"]]["inputs"]["steps"] = steps

        # Queue workflow
        prompt_id = await comfy_client.queue_prompt(workflow)

        # Track progress
        if ctx:
            async for progress in comfy_client.progress(prompt_id):
                await ctx.report_progress(
                    progress,
                    100,
                    f"Generating with {control_type} control... {progress}%",
                )

        # Collect outputs
        output_files = await comfy_client.collect_output_files(prompt_id)

        # Build response content blocks
        content_blocks: List[TextContent | ImageContent] = []

        # 1. Text summary with all parameters
        summary_parts = [
            f"Generated {len(output_files)} image(s) with {control_type} ControlNet.",
            "Parameters:",
            f"  prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
            f"  control_type: {control_type}",
            f"  control_strength: {control_strength}",
            f"  size: {size} ({width}x{height})",
            f"  steps: {steps}",
            f"  seed: {seed}",
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

        logger.info(f"zimage_controlnet completed: {len(output_files)} image(s)")
        return content_blocks

    except Exception as e:
        logger.error(f"zimage_controlnet error: {e}", exc_info=True)
        error_msg = (
            f"Failed to generate image with ControlNet: {str(e)}\n\n"
            f"Troubleshooting:\n"
            f"- Ensure ComfyUI is running at {comfy_client.base_url}\n"
            f"- Check that Z-Image-Turbo-Fun-Controlnet-Union.safetensors is in ComfyUI/models/model_patches/\n"
            f"- Check that comfyui_controlnet_aux custom node is installed\n"
            f"- Verify the input image is valid base64 PNG/JPEG"
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
