# ABOUTME: MCP server for LoRA training management.
# Provides tools for dataset creation, image upload, training control, and checkpoint promotion.

import base64
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Literal, Optional

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ImageContent, TextContent

from lora_trainer.captioner import VisionCaptioner
from lora_trainer.dataset_manager import DatasetManager
from lora_trainer.image_utils import (
    ImageFetchError,
    ImageProcessingError,
    fetch_image,
    fetch_images_batch,
    smart_crop,
)
from lora_trainer.job_store import create_job_store
from lora_trainer.job_store_base import JobStoreBase
from lora_trainer.models import LoRAType, TrainingConfig, TrainingStatus
from lora_trainer.training_runner import TrainingRunner


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _to_json(obj: Any) -> str:
    """Serialize object to JSON string for UI consumption."""

    def default(o: Any) -> Any:
        if hasattr(o, "model_dump"):  # Pydantic
            return o.model_dump()
        if hasattr(o, "value"):  # Enum
            return o.value
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    return json.dumps(obj, default=default)


# Initialize MCP server
mcp = FastMCP("lora_trainer", host="0.0.0.0")

# Lazy initialization - only create managers when first accessed
BASE_PATH = Path("/data")
_dataset_manager: Optional[DatasetManager] = None
_job_store: Optional["JobStoreBase"] = None
_training_runner: Optional[TrainingRunner] = None


def _get_dataset_manager() -> DatasetManager:
    """Lazy initialization of dataset manager."""
    global _dataset_manager
    if _dataset_manager is None:
        _dataset_manager = DatasetManager(BASE_PATH / "datasets")
    return _dataset_manager


def _get_job_store() -> "JobStoreBase":
    """Lazy initialization of job store."""
    global _job_store
    if _job_store is None:
        _job_store = create_job_store(BASE_PATH)
    return _job_store


def _get_training_runner() -> TrainingRunner:
    """Lazy initialization of training runner."""
    global _training_runner
    if _training_runner is None:
        _training_runner = TrainingRunner(
            datasets_path=BASE_PATH / "datasets",
            outputs_path=BASE_PATH / "outputs",
            configs_path=BASE_PATH / "configs",
            job_store=_get_job_store(),
            loras_path=Path("/models/loras"),
        )
    return _training_runner


# ============================================================================
# Dataset Management Tools
# ============================================================================


@mcp.tool()
async def lora_create_dataset(
    name: str,
    trigger_token: str,
    lora_type: Literal["character", "style", "concept"] = "character",
    description: Optional[str] = None,
    output_format: Literal["text", "json"] = "text",
    ctx: Context = None,
) -> List[TextContent]:
    """Create a NEW LoRA training dataset. ONLY use when user explicitly requests a new dataset.

    WHEN TO USE:
    - User says "create a new dataset" or "start a new LoRA"
    - No existing dataset matches their needs

    DO NOT USE when:
    - User mentions an existing dataset name
    - User wants to add images to an existing dataset
    - User wants to train/caption/validate an existing dataset

    TRIGGER TOKEN GUIDELINES:
    - Use unique, non-dictionary words (e.g., "ohwx", "sks", "xyz123")
    - Keep it short (3-6 characters)
    - Avoid common words that appear in training data

    Args:
        name: NEW dataset name (alphanumeric + underscores, starts with letter)
        trigger_token: Unique token to trigger the LoRA (e.g., "ohwx")
        lora_type: character, style, or concept
        description: Optional description
        output_format: "text" for human-readable, "json" for structured data.

    Returns:
        Confirmation with dataset path.
    """
    try:
        dm = _get_dataset_manager()
        metadata = dm.create_dataset(
            name=name,
            trigger_token=trigger_token,
            lora_type=LoRAType(lora_type),
            description=description,
        )

        if output_format == "json":
            return [TextContent(type="text", text=_to_json(metadata))]

        return [
            TextContent(
                type="text",
                text=f"Created dataset '{name}' with trigger token '{trigger_token}'.\n"
                f"Type: {lora_type}\n"
                f"Path: {dm.get_dataset_path(name)}\n\n"
                f"Next: Upload training images with lora_upload_images.",
            )
        ]
    except ValueError as e:
        if output_format == "json":
            return [TextContent(type="text", text=_to_json({"error": str(e)}))]
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@mcp.tool()
async def lora_delete_dataset(
    name: str,
    output_format: Literal["text", "json"] = "text",
    ctx: Context = None,
) -> List[TextContent]:
    """Delete a dataset and all its contents.

    WHEN TO USE:
    - User explicitly asks to delete a dataset
    - Cleaning up unused datasets

    Args:
        name: Dataset name to delete.
        output_format: "text" for human-readable, "json" for structured data.

    Returns:
        Confirmation of deletion.
    """
    try:
        dm = _get_dataset_manager()
        dm.delete_dataset(name)

        if output_format == "json":
            return [TextContent(type="text", text=_to_json({"deleted": name}))]

        return [TextContent(type="text", text=f"Deleted dataset '{name}'.")]
    except Exception as e:
        if output_format == "json":
            return [TextContent(type="text", text=_to_json({"error": str(e)}))]
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@mcp.tool()
async def lora_upload_images(
    dataset_name: str,
    images: List[str],
    captions: Optional[List[str]] = None,
    ctx: Context = None,
) -> List[TextContent]:
    """Upload training images to a dataset.

    IMAGE REQUIREMENTS:
    - Resolution: 1024x1024 or 1536x1536 optimal
    - Format: PNG or JPEG (base64 encoded)
    - Count: 5-15 images minimum (9 sufficient for identity)
    - Diversity: Vary poses, expressions, lighting

    CAPTIONS (optional):
    - If provided, include trigger token in each caption
    - Example: "ohwx, portrait photo, studio lighting"

    Args:
        dataset_name: Name of existing dataset
        images: List of base64-encoded images
        captions: Optional list of captions (same length as images)

    Returns:
        Upload summary with validation results.
    """
    try:
        if ctx:
            await ctx.report_progress(0, len(images), "Starting upload...")

        results = []
        for i, img_data in enumerate(images):
            # Handle data URL format
            if "," in img_data:
                img_data = img_data.split(",", 1)[1]

            # Decode and save image
            img_bytes = base64.b64decode(img_data)
            caption = captions[i] if captions and i < len(captions) else None
            dm = _get_dataset_manager()
            filename = dm.add_image(dataset_name, img_bytes, caption)
            results.append(filename)

            if ctx:
                await ctx.report_progress(i + 1, len(images), f"Uploaded {filename}")

        metadata = _get_dataset_manager().get_metadata(dataset_name)
        return [
            TextContent(
                type="text",
                text=f"Uploaded {len(results)} images to '{dataset_name}'.\n"
                f"Total images: {metadata.image_count}\n"
                f"Has captions: {metadata.has_captions}\n\n"
                f"Use lora_validate_dataset to check readiness.",
            )
        ]
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@mcp.tool()
async def lora_validate_dataset(
    dataset_name: str,
    ctx: Context = None,
) -> List[TextContent]:
    """Validate dataset readiness for training.

    Checks:
    - Image count (minimum 5)
    - Image resolutions
    - Caption format (if present)
    - Trigger token consistency

    Args:
        dataset_name: Name of dataset to validate

    Returns:
        Validation report with any issues found.
    """
    try:
        report = _get_dataset_manager().validate_dataset(dataset_name)
        return [TextContent(type="text", text=report)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@mcp.tool()
async def lora_list_datasets(
    output_format: Literal["text", "json"] = "text",
    ctx: Context = None,
) -> List[TextContent]:
    """List all available training datasets.

    WHEN TO USE:
    - User asks "what datasets exist?" or "show me datasets"
    - You need to verify a dataset exists before using it

    DO NOT USE as a default/fallback action. If user gives a specific task, do that task.

    Args:
        output_format: "text" for human-readable, "json" for structured data.

    Returns:
        List of datasets with metadata (name, trigger, type, image count).
    """
    datasets = _get_dataset_manager().list_datasets()

    if output_format == "json":
        return [TextContent(type="text", text=_to_json({"datasets": datasets}))]

    if not datasets:
        return [TextContent(type="text", text="No datasets found.")]

    lines = ["Available datasets:\n"]
    for ds in datasets:
        lines.append(f"- {ds.name}")
        lines.append(f"  Trigger: {ds.trigger_token}")
        lines.append(f"  Type: {ds.lora_type.value}")
        lines.append(f"  Images: {ds.image_count}")
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


@mcp.tool()
async def lora_get_dataset(
    name: str,
    output_format: Literal["text", "json"] = "text",
    ctx: Context = None,
) -> List[TextContent]:
    """Get detailed information about a dataset including all images.

    WHEN TO USE:
    - User asks about a specific dataset's contents
    - You need to see what images are in a dataset
    - Before editing captions or managing images

    Args:
        name: Dataset name.
        output_format: "text" for human-readable, "json" for structured data.

    Returns:
        Dataset metadata and list of images with captions.
    """
    dm = _get_dataset_manager()
    metadata = dm.get_metadata(name)
    dataset_path = dm.get_dataset_path(name)
    images_dir = dataset_path / "images"
    captions_dir = dataset_path / "captions"

    # Collect images with captions
    images = []
    if images_dir.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in sorted(images_dir.glob(ext)):
                caption_path = captions_dir / f"{img_path.stem}.txt"
                caption = None
                if caption_path.exists():
                    caption = caption_path.read_text().strip()
                images.append(
                    {
                        "filename": img_path.name,
                        "caption": caption,
                    }
                )

    if output_format == "json":
        return [
            TextContent(
                type="text",
                text=_to_json(
                    {
                        "name": metadata.name,
                        "trigger_token": metadata.trigger_token,
                        "lora_type": metadata.lora_type.value,
                        "description": metadata.description,
                        "image_count": metadata.image_count,
                        "has_captions": metadata.has_captions,
                        "created_at": metadata.created_at,
                        "images": images,
                    }
                ),
            )
        ]

    # Text format
    lines = [
        f"Dataset: {metadata.name}",
        f"Trigger: {metadata.trigger_token}",
        f"Type: {metadata.lora_type.value}",
        f"Images: {metadata.image_count}",
        f"Has captions: {metadata.has_captions}",
        "",
        "Images:",
    ]
    for img in images:
        caption_preview = (
            img["caption"][:100] + "..."
            if img["caption"] and len(img["caption"]) > 100
            else img["caption"]
        )
        lines.append(f"- {img['filename']}: {caption_preview or '(no caption)'}")

    return [TextContent(type="text", text="\n".join(lines))]


@mcp.tool()
async def lora_get_image(
    dataset_name: str,
    filename: str,
    output_format: Literal["text", "json"] = "text",
    ctx: Context = None,
) -> List[TextContent | ImageContent]:
    """Get a specific image from a dataset.

    WHEN TO USE:
    - User wants to see a specific image from a dataset
    - Reviewing individual images before training

    Args:
        dataset_name: Name of the dataset.
        filename: Image filename (e.g., "001_abc123.png").
        output_format: "text" for LLM (returns ImageContent), "json" for UI (returns base64 in JSON).

    Returns:
        The image and its caption if available.
    """
    dm = _get_dataset_manager()
    if not dm.dataset_exists(dataset_name):
        if output_format == "json":
            return [
                TextContent(
                    type="text",
                    text=_to_json({"error": f"Dataset '{dataset_name}' not found."}),
                )
            ]
        return [TextContent(type="text", text=f"Dataset '{dataset_name}' not found.")]

    dataset_path = dm.get_dataset_path(dataset_name)
    image_path = dataset_path / "images" / filename

    if not image_path.exists():
        if output_format == "json":
            return [
                TextContent(
                    type="text",
                    text=_to_json({"error": f"Image '{filename}' not found."}),
                )
            ]
        return [TextContent(type="text", text=f"Image '{filename}' not found.")]

    # Get caption if exists
    caption_path = dataset_path / "captions" / f"{image_path.stem}.txt"
    caption = None
    if caption_path.exists():
        caption = caption_path.read_text().strip()

    # Read and encode image
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    suffix = image_path.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/png")

    if output_format == "json":
        return [
            TextContent(
                type="text",
                text=_to_json(
                    {
                        "filename": filename,
                        "data": image_data,
                        "content_type": media_type,
                        "caption": caption,
                    }
                ),
            )
        ]

    # Text format - return ImageContent for LLM
    result: List[TextContent | ImageContent] = []
    if caption:
        result.append(TextContent(type="text", text=f"Caption: {caption}"))
    result.append(ImageContent(type="image", data=image_data, mimeType=media_type))
    return result


@mcp.tool()
async def lora_delete_image(
    dataset_name: str,
    filename: str,
    output_format: Literal["text", "json"] = "text",
    ctx: Context = None,
) -> List[TextContent]:
    """Delete an image from a dataset.

    WHEN TO USE:
    - User wants to remove a specific image
    - Cleaning up bad images from a dataset

    Args:
        dataset_name: Name of the dataset.
        filename: Image filename to delete.
        output_format: "text" for human-readable, "json" for structured data.

    Returns:
        Confirmation of deletion.
    """
    dm = _get_dataset_manager()
    if not dm.dataset_exists(dataset_name):
        if output_format == "json":
            return [
                TextContent(
                    type="text",
                    text=_to_json({"error": f"Dataset '{dataset_name}' not found."}),
                )
            ]
        return [TextContent(type="text", text=f"Dataset '{dataset_name}' not found.")]

    dataset_path = dm.get_dataset_path(dataset_name)
    image_path = dataset_path / "images" / filename

    if not image_path.exists():
        if output_format == "json":
            return [
                TextContent(
                    type="text",
                    text=_to_json({"error": f"Image '{filename}' not found."}),
                )
            ]
        return [TextContent(type="text", text=f"Image '{filename}' not found.")]

    # Delete image
    image_path.unlink()

    # Delete caption if exists
    caption_path = dataset_path / "captions" / f"{image_path.stem}.txt"
    if caption_path.exists():
        caption_path.unlink()

    # Update metadata
    dm._update_metadata(dataset_name)

    if output_format == "json":
        return [
            TextContent(
                type="text",
                text=_to_json({"deleted": filename, "dataset": dataset_name}),
            )
        ]

    return [TextContent(type="text", text=f"Deleted '{filename}' from {dataset_name}.")]


@mcp.tool()
async def lora_update_caption(
    dataset_name: str,
    filename: str,
    caption: str,
    output_format: Literal["text", "json"] = "text",
    ctx: Context = None,
) -> List[TextContent]:
    """Update the caption for a specific image.

    WHEN TO USE:
    - User wants to edit or fix a caption
    - Manually setting captions for specific images

    Args:
        dataset_name: Name of the dataset.
        filename: Image filename (e.g., "001_abc123.png").
        caption: New caption text.
        output_format: "text" for human-readable, "json" for structured data.

    Returns:
        Confirmation of update.
    """
    dm = _get_dataset_manager()
    if not dm.dataset_exists(dataset_name):
        if output_format == "json":
            return [
                TextContent(
                    type="text",
                    text=_to_json({"error": f"Dataset '{dataset_name}' not found."}),
                )
            ]
        return [TextContent(type="text", text=f"Dataset '{dataset_name}' not found.")]

    dataset_path = dm.get_dataset_path(dataset_name)
    image_path = dataset_path / "images" / filename

    if not image_path.exists():
        if output_format == "json":
            return [
                TextContent(
                    type="text",
                    text=_to_json({"error": f"Image '{filename}' not found."}),
                )
            ]
        return [TextContent(type="text", text=f"Image '{filename}' not found.")]

    # Write caption
    captions_dir = dataset_path / "captions"
    captions_dir.mkdir(exist_ok=True)
    caption_path = captions_dir / f"{image_path.stem}.txt"
    caption_path.write_text(caption)

    # Update metadata
    dm._update_metadata(dataset_name)

    if output_format == "json":
        return [
            TextContent(
                type="text",
                text=_to_json(
                    {"filename": filename, "dataset": dataset_name, "caption": caption}
                ),
            )
        ]

    return [
        TextContent(
            type="text",
            text=f"Updated caption for '{filename}':\n{caption[:200]}{'...' if len(caption) > 200 else ''}",
        )
    ]


# ============================================================================
# Image Acquisition Tools
# ============================================================================


@mcp.tool()
async def lora_fetch_image(
    dataset_name: str,
    url: str,
    caption: Optional[str] = None,
    preprocess: bool = True,
    crop_mode: Literal["center", "smart", "none"] = "smart",
    ctx: Context = None,
) -> List[TextContent]:
    """Download and add a NEW image from a URL to a dataset.

    WHEN TO USE:
    - User provides a URL to download and add to the dataset
    - Building a dataset from web images

    DO NOT USE for captioning. To caption images already in a dataset, use lora_caption instead.

    For multiple images, use lora_fetch_images_batch instead.

    REQUIRES: The dataset_name must already exist.

    Args:
        dataset_name: Name of EXISTING dataset to add image to
        url: URL of image to fetch (PNG, JPEG, WebP)
        caption: Optional caption (should include trigger token)
        preprocess: Apply smart crop and resize (recommended: true)
        crop_mode: "smart" for face detection, "center", or "none"

    Returns:
        Confirmation with filename.
    """
    try:
        # Fetch image from URL
        if ctx:
            await ctx.report_progress(0, 2, "Fetching image...")

        image_bytes = await fetch_image(url)

        # Optionally preprocess
        if preprocess:
            if ctx:
                await ctx.report_progress(1, 2, "Preprocessing...")
            image_bytes = smart_crop(image_bytes, target_size=1024, crop_mode=crop_mode)

        # Add to dataset
        dm = _get_dataset_manager()
        filename = dm.add_image(dataset_name, image_bytes, caption)

        if ctx:
            await ctx.report_progress(2, 2, "Done")

        metadata = dm.get_metadata(dataset_name)
        return [
            TextContent(
                type="text",
                text=f"Added image from URL to '{dataset_name}'.\n"
                f"Filename: {filename}\n"
                f"Total images: {metadata.image_count}\n"
                f"Preprocessed: {preprocess} (crop_mode={crop_mode})",
            )
        ]
    except ImageFetchError as e:
        return [TextContent(type="text", text=f"Fetch error: {str(e)}")]
    except ImageProcessingError as e:
        return [TextContent(type="text", text=f"Processing error: {str(e)}")]
    except Exception as e:
        logger.error(f"Fetch image error: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@mcp.tool()
async def lora_fetch_images_batch(
    dataset_name: str,
    urls: List[str],
    auto_caption: bool = False,
    caption_style: Literal["detailed", "simple", "tags"] = "detailed",
    preprocess: bool = True,
    crop_mode: Literal["center", "smart", "none"] = "smart",
    ctx: Context = None,
) -> List[TextContent]:
    """Add images from URLs to an EXISTING dataset. Use this to populate a dataset with images.

    WHEN TO USE:
    - User provides image URLs to add to a dataset
    - User wants to fetch/download images into a dataset
    - Adding multiple images at once

    REQUIRES: The dataset_name must already exist (created with lora_create_dataset).

    Args:
        dataset_name: Name of EXISTING dataset to add images to
        urls: List of image URLs to fetch (PNG, JPEG, WebP)
        auto_caption: Generate captions automatically (recommended: true)
        caption_style: "detailed" for full descriptions, "simple" for short, "tags" for comma-separated
        preprocess: Apply smart crop and resize (recommended: true)
        crop_mode: "smart" for face detection, "center", or "none"

    Returns:
        Summary of fetched images and any failures.
    """
    try:
        dm = _get_dataset_manager()
        metadata = dm.get_metadata(dataset_name)

        total = len(urls)
        if ctx:
            await ctx.report_progress(0, total, "Starting batch fetch...")

        # Fetch all images
        def progress_cb(completed: int, total: int, url: str):
            # Can't call async ctx.report_progress from sync callback
            logger.info(f"Fetched {completed}/{total}: {url[:50]}...")

        results = await fetch_images_batch(
            urls, max_concurrent=5, on_progress=progress_cb
        )

        # Process results
        success_count = 0
        fail_count = 0
        failures = []

        # Prepare captioner if needed
        captioner = None
        if auto_caption:
            captioner = VisionCaptioner()

        try:
            for i, (url, result) in enumerate(results):
                if isinstance(result, Exception):
                    fail_count += 1
                    failures.append(f"- {url[:50]}...: {str(result)[:50]}")
                    continue

                # Preprocess if requested
                image_bytes = result
                if preprocess:
                    try:
                        image_bytes = smart_crop(
                            image_bytes, target_size=1024, crop_mode=crop_mode
                        )
                    except ImageProcessingError as e:
                        fail_count += 1
                        failures.append(f"- {url[:50]}...: preprocessing failed: {e}")
                        continue

                # Generate caption if requested
                caption = None
                if auto_caption and captioner:
                    try:
                        caption = await captioner.caption_image(
                            image_bytes=image_bytes,
                            style=caption_style,
                            trigger_token=metadata.trigger_token,
                        )
                    except Exception as e:
                        logger.warning(f"Caption generation failed for {url}: {e}")
                        # Continue without caption

                # Add to dataset
                dm.add_image(dataset_name, image_bytes, caption)
                success_count += 1

                if ctx:
                    await ctx.report_progress(
                        i + 1, total, f"Processed {i + 1}/{total}"
                    )
        finally:
            if captioner:
                await captioner.close()

        # Build summary
        metadata = dm.get_metadata(dataset_name)
        lines = [
            f"Batch fetch complete for '{dataset_name}':",
            f"  Succeeded: {success_count}/{total}",
            f"  Failed: {fail_count}/{total}",
            f"  Total images now: {metadata.image_count}",
            f"  Auto-captioned: {auto_caption}",
        ]

        if failures:
            lines.append("\nFailures:")
            lines.extend(failures[:10])  # Limit to 10
            if len(failures) > 10:
                lines.append(f"  ... and {len(failures) - 10} more")

        return [TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        logger.error(f"Batch fetch error: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# ============================================================================
# Captioning Tools
# ============================================================================


@mcp.tool()
async def lora_caption(
    dataset_name: str,
    style: Literal["detailed", "simple", "tags"] = "detailed",
    image_name: Optional[str] = None,
    limit: Optional[int] = None,
    overwrite: bool = False,
    custom_prompt: Optional[str] = None,
    ctx: Context = None,
) -> List[TextContent]:
    """Generate captions for images ALREADY IN a dataset using Qwen-VL vision model.

    WHEN TO USE:
    - User asks to "caption", "describe", or "generate captions" for a dataset
    - User wants to regenerate/update existing captions with a different style
    - After uploading images, before training

    DO NOT USE lora_fetch_image to caption - that tool is ONLY for downloading NEW images from URLs.
    This tool captions images that are ALREADY in the dataset on disk.

    Automatically prepends the dataset's trigger token to each caption.

    MODES:
    - Single image: provide image_name (e.g., "001_abc123.png")
    - Limited batch: provide limit (e.g., limit=5 for next 5 uncaptioned)
    - All images: omit both image_name and limit

    STYLES:
    - detailed: Full description with appearance, pose, background, lighting
    - simple: Brief phrase like "portrait photo, studio lighting"
    - tags: Comma-separated tags like "woman, brown_hair, outdoor"

    Args:
        dataset_name: Name of dataset to caption
        style: Caption style (default "detailed")
        image_name: Specific image filename to caption (optional)
        limit: Max images to caption (optional, ignored if image_name set)
        overwrite: Set to true to regenerate/replace existing captions
        custom_prompt: Override style with custom prompt (optional)

    Returns:
        Summary with sample captions.
    """
    try:
        dm = _get_dataset_manager()
        metadata = dm.get_metadata(dataset_name)
        dataset_path = dm.get_dataset_path(dataset_name)
        images_dir = dataset_path / "images"
        captions_dir = dataset_path / "captions"
        captions_dir.mkdir(exist_ok=True)

        if not images_dir.exists():
            return [
                TextContent(
                    type="text", text=f"No images directory found for '{dataset_name}'"
                )
            ]

        # MODE 1: Single image
        if image_name:
            image_path = images_dir / image_name
            if not image_path.exists():
                return [TextContent(type="text", text=f"Image not found: {image_name}")]

            with open(image_path, "rb") as f:
                image_bytes = f.read()

            async with VisionCaptioner() as captioner:
                caption = await captioner.caption_image(
                    image_bytes=image_bytes,
                    style=style,
                    trigger_token=metadata.trigger_token,
                    custom_prompt=custom_prompt,
                )

            caption_path = captions_dir / f"{image_path.stem}.txt"
            caption_path.write_text(caption)
            dm._update_metadata(dataset_name)

            return [
                TextContent(
                    type="text", text=f"Caption for '{image_name}':\n\n{caption}"
                )
            ]

        # MODE 2/3: Batch (all or limited)
        image_files = sorted(
            [
                p
                for p in images_dir.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            ]
        )

        if not image_files:
            return [
                TextContent(type="text", text=f"No images found in '{dataset_name}'")
            ]

        total_images = len(image_files)

        # Filter to uncaptioned unless overwrite is True
        if not overwrite:
            image_files = [
                p for p in image_files if not (captions_dir / f"{p.stem}.txt").exists()
            ]

        # Apply limit
        if limit and limit > 0:
            image_files = image_files[:limit]

        if not image_files:
            return [
                TextContent(
                    type="text",
                    text=f"SUCCESS: All {total_images} images in '{dataset_name}' already have captions. "
                    f"The dataset is ready for training. If the user wants to regenerate captions with a "
                    f"different style, use lora_regenerate_captions instead.",
                )
            ]

        # Caption batch
        results = {}
        async with VisionCaptioner() as captioner:
            for idx, image_path in enumerate(image_files, 1):
                logger.info(f"Captioning {idx}/{len(image_files)}: {image_path.name}")
                if ctx:
                    await ctx.report_progress(idx, len(image_files), image_path.name)

                with open(image_path, "rb") as f:
                    image_bytes = f.read()

                try:
                    caption = await captioner.caption_image(
                        image_bytes=image_bytes,
                        style=style,
                        trigger_token=metadata.trigger_token,
                        custom_prompt=custom_prompt,
                    )

                    caption_path = captions_dir / f"{image_path.stem}.txt"
                    caption_path.write_text(caption)
                    results[image_path.name] = caption
                except Exception as e:
                    logger.error(f"Failed to caption {image_path.name}: {e}")
                    results[image_path.name] = f"[ERROR: {e}]"

        dm._update_metadata(dataset_name)

        lines = [
            f"Captioned {len(results)} images in '{dataset_name}' ({style} style):",
            "",
        ]
        for name, caption in list(results.items())[:3]:
            preview = caption[:60] + "..." if len(caption) > 60 else caption
            lines.append(f"  {name}: {preview}")

        if len(results) > 3:
            lines.append(f"  ... and {len(results) - 3} more")

        return [TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        logger.error(f"Caption error: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@mcp.tool()
async def lora_regenerate_captions(
    dataset_name: str,
    style: Literal["detailed", "simple", "tags"] = "detailed",
    custom_prompt: Optional[str] = None,
    ctx: Context = None,
) -> List[TextContent]:
    """Regenerate ALL captions in a dataset, replacing existing ones.

    WHEN TO USE:
    - User says "regenerate captions" or "re-caption"
    - User wants to change caption style (e.g., from detailed to tags)
    - User wants to overwrite existing captions

    This tool ALWAYS overwrites existing captions. For adding captions to
    uncaptioned images only, use lora_caption instead.

    Args:
        dataset_name: Name of dataset to regenerate captions for
        style: Caption style - detailed, simple, or tags
        custom_prompt: Optional custom prompt for vision model

    Returns:
        Summary with sample regenerated captions.
    """
    # Delegate to lora_caption with overwrite=True
    return await lora_caption(
        dataset_name=dataset_name,
        style=style,
        overwrite=True,
        custom_prompt=custom_prompt,
        ctx=ctx,
    )


# ============================================================================
# Preprocessing Tools
# ============================================================================


@mcp.tool()
async def lora_preprocess_dataset(
    dataset_name: str,
    target_size: int = 1024,
    crop_mode: Literal["center", "smart", "none"] = "smart",
    ctx: Context = None,
) -> List[TextContent]:
    """Preprocess all images in a dataset for optimal training.

    Applies smart cropping (face detection for characters) and resizing
    to ensure consistent, high-quality training images.

    PREPROCESSING STEPS:
    - Detect faces (if crop_mode="smart") and center crop around them
    - Resize to target_size x target_size (square)
    - Convert to PNG format
    - Remove EXIF metadata for privacy

    Args:
        dataset_name: Name of dataset to preprocess
        target_size: Output size in pixels (default 1024, recommended 1024 or 1536)
        crop_mode: "smart" for face detection, "center", or "none"

    Returns:
        Summary of preprocessed images.
    """
    try:
        dm = _get_dataset_manager()
        dataset_path = dm.get_dataset_path(dataset_name)
        images_dir = dataset_path / "images"

        if not images_dir.exists():
            return [
                TextContent(
                    type="text", text=f"No images directory found for '{dataset_name}'"
                )
            ]

        # Find all images
        image_files = sorted(
            [
                p
                for p in images_dir.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            ]
        )
        total = len(image_files)

        if total == 0:
            return [
                TextContent(type="text", text=f"No images found in '{dataset_name}'")
            ]

        if ctx:
            await ctx.report_progress(0, total, "Starting preprocessing...")

        processed = 0
        failed = 0
        failures = []

        for i, image_path in enumerate(image_files):
            try:
                # Read image
                with open(image_path, "rb") as f:
                    image_bytes = f.read()

                # Process
                processed_bytes = smart_crop(
                    image_bytes, target_size=target_size, crop_mode=crop_mode
                )

                # Write back (as PNG)
                output_path = image_path.with_suffix(".png")
                with open(output_path, "wb") as f:
                    f.write(processed_bytes)

                # Remove original if it was different format
                if output_path != image_path:
                    image_path.unlink()

                processed += 1
                logger.info(f"Preprocessed {i + 1}/{total}: {image_path.name}")

            except Exception as e:
                failed += 1
                failures.append(f"- {image_path.name}: {str(e)[:50]}")
                logger.warning(f"Failed to preprocess {image_path.name}: {e}")

            if ctx:
                await ctx.report_progress(i + 1, total, f"Processed {i + 1}/{total}")

        # Update metadata
        dm._update_metadata(dataset_name)

        lines = [
            f"Preprocessing complete for '{dataset_name}':",
            f"  Target size: {target_size}x{target_size}",
            f"  Crop mode: {crop_mode}",
            f"  Processed: {processed}/{total}",
            f"  Failed: {failed}/{total}",
        ]

        if failures:
            lines.append("\nFailures:")
            lines.extend(failures[:5])

        return [TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        logger.error(f"Preprocess error: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# ============================================================================
# Training Control Tools
# ============================================================================


@mcp.tool()
async def lora_start_training(
    dataset_name: str,
    steps: int = 3000,
    learning_rate: float = 0.0001,
    lora_rank: int = 8,
    checkpoint_every: int = 500,
    sample_every: int = 250,
    sample_prompts: Optional[List[str]] = None,
    ctx: Context = None,
) -> List[TextContent]:
    """Start LoRA training job.

    RECOMMENDED SETTINGS:
    - steps: 3000 for 5-15 image datasets
    - learning_rate: 0.0001 (decrease to 0.00005 if overfitting)
    - lora_rank: 8 (increase to 16 for more capacity)
    - checkpoint_every: 500 (for recovery and comparison)
    - sample_every: 250 (monitor convergence)

    SAMPLE PROMPTS:
    - Include trigger token in each prompt
    - Use fixed seeds for consistent comparison
    - Example: ["ohwx, portrait photo", "ohwx on a beach, sunset"]

    Args:
        dataset_name: Name of prepared dataset
        steps: Total training steps (default 3000)
        learning_rate: Learning rate (default 0.0001)
        lora_rank: LoRA rank/dimension (default 8)
        checkpoint_every: Save checkpoint interval (default 500)
        sample_every: Generate sample interval (default 250)
        sample_prompts: Prompts for sample generation

    Returns:
        Job ID for status polling.
    """
    try:
        # Validate dataset exists and is ready
        metadata = _get_dataset_manager().get_metadata(dataset_name)
        if metadata.image_count < 5:
            return [
                TextContent(
                    type="text",
                    text=f"Error: Dataset has {metadata.image_count} images. Minimum 5 required.",
                )
            ]

        # Build config
        config = TrainingConfig(
            dataset=dataset_name,
            steps=steps,
            lr=learning_rate,
            lora_rank=lora_rank,
            checkpoint_every=checkpoint_every,
            sample_every=sample_every,
            sample_prompts=sample_prompts
            or [
                f"{metadata.trigger_token}, portrait, studio lighting",
                f"{metadata.trigger_token}, outdoor, natural light",
            ],
        )

        # Start training
        job = await _get_training_runner().start_training(
            dataset_name=dataset_name,
            trigger_token=metadata.trigger_token,
            config=config,
        )

        return [
            TextContent(
                type="text",
                text=f"Started training job: {job.job_id}\n"
                f"Dataset: {dataset_name}\n"
                f"Steps: {steps}\n"
                f"LoRA rank: {lora_rank}\n"
                f"Learning rate: {learning_rate}\n\n"
                f"Use lora_training_status('{job.job_id}') to monitor progress.",
            )
        ]
    except Exception as e:
        logger.error(f"Training start error: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@mcp.tool()
async def lora_training_status(
    job_id: str,
    ctx: Context = None,
) -> List[TextContent | ImageContent]:
    """Get training job status and progress.

    Args:
        job_id: Job ID from lora_start_training

    Returns:
        Status, progress, loss, and sample images (if available).
    """
    try:
        job = _get_job_store().get_job(job_id)
        if not job:
            return [TextContent(type="text", text=f"Job not found: {job_id}")]

        content: List[TextContent | ImageContent] = []

        # Status text
        progress_pct = (
            f"{job.current_step / job.total_steps * 100:.1f}%"
            if job.total_steps > 0
            else "0%"
        )
        status_lines = [
            f"Job: {job.job_id}",
            f"Dataset: {job.dataset_name}",
            f"Status: {job.status.value}",
            f"Progress: {job.current_step}/{job.total_steps} ({progress_pct})",
        ]

        if job.latest_loss is not None:
            status_lines.append(f"Latest loss: {job.latest_loss:.4f}")

        if job.checkpoints:
            status_lines.append(f"Checkpoints: {len(job.checkpoints)}")
            status_lines.append(f"  Latest: {job.checkpoints[-1]}")

        if job.error_message:
            status_lines.append(f"Error: {job.error_message[:200]}")

        content.append(TextContent(type="text", text="\n".join(status_lines)))

        # Include sample images if available (last 2)
        for sample_path in job.sample_images[-2:]:
            try:
                with open(sample_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode()
                content.append(
                    ImageContent(
                        type="image",
                        data=img_base64,
                        mimeType="image/png",
                    )
                )
            except Exception as e:
                logger.debug(f"Failed to read sample image: {e}")

        return content
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@mcp.tool()
async def lora_stop_training(
    job_id: str,
    ctx: Context = None,
) -> List[TextContent]:
    """Stop a running training job.

    The latest checkpoint will be preserved.

    Args:
        job_id: Job ID to stop

    Returns:
        Confirmation with available checkpoints.
    """
    try:
        job = await _get_training_runner().stop_training(job_id)
        return [
            TextContent(
                type="text",
                text=f"Stopped job: {job_id}\n"
                f"Checkpoints available: {len(job.checkpoints)}\n"
                f"Use lora_list_checkpoints('{job_id}') to see options.",
            )
        ]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@mcp.tool()
async def lora_list_jobs(
    status: Optional[
        Literal["pending", "running", "completed", "failed", "stopped"]
    ] = None,
    ctx: Context = None,
) -> List[TextContent]:
    """List all training jobs.

    Args:
        status: Filter by status (optional)

    Returns:
        List of jobs with summary info.
    """
    jobs = _get_job_store().list_jobs(status=TrainingStatus(status) if status else None)
    if not jobs:
        return [TextContent(type="text", text="No jobs found.")]

    lines = ["Training Jobs:\n"]
    for job in jobs:
        lines.append(f"- {job.job_id} ({job.status.value})")
        lines.append(f"  Dataset: {job.dataset_name}")
        lines.append(f"  Progress: {job.current_step}/{job.total_steps}")
        if job.checkpoints:
            lines.append(f"  Checkpoints: {len(job.checkpoints)}")
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


# ============================================================================
# Checkpoint Management Tools
# ============================================================================


@mcp.tool()
async def lora_list_checkpoints(
    job_id: str,
    ctx: Context = None,
) -> List[TextContent]:
    """List all checkpoints for a training job.

    Args:
        job_id: Job ID

    Returns:
        List of checkpoints with step numbers.
    """
    try:
        job = _get_job_store().get_job(job_id)
        if not job:
            return [TextContent(type="text", text=f"Job not found: {job_id}")]

        if not job.checkpoints:
            return [TextContent(type="text", text="No checkpoints available yet.")]

        lines = [f"Checkpoints for job {job_id}:\n"]
        for ckpt in job.checkpoints:
            lines.append(f"- {ckpt}")

        lines.append("\nUse lora_promote_checkpoint to copy to loras directory.")
        return [TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@mcp.tool()
async def lora_promote_checkpoint(
    job_id: str,
    checkpoint_name: str,
    output_name: Optional[str] = None,
    ctx: Context = None,
) -> List[TextContent]:
    """Promote a checkpoint to the active LoRA directory.

    Copies the checkpoint to ComfyUI's loras directory, making it
    available for inference with zimage_turbo.

    Args:
        job_id: Job ID
        checkpoint_name: Checkpoint filename (from lora_list_checkpoints)
        output_name: Output filename (default: dataset_name)

    Returns:
        Confirmation with usage instructions.
    """
    try:
        job = _get_job_store().get_job(job_id)
        if not job:
            return [TextContent(type="text", text=f"Job not found: {job_id}")]

        output_filename = _get_training_runner().promote_checkpoint(
            job_id=job_id,
            checkpoint_name=checkpoint_name,
            output_name=output_name or job.dataset_name,
        )

        lora_name = Path(output_filename).stem

        return [
            TextContent(
                type="text",
                text=f"Promoted checkpoint to: {output_filename}\n\n"
                f"Usage with zimage_turbo:\n"
                f'  lora_name: "{lora_name}"\n'
                f"  lora_strength: 1.0\n"
                f'  prompt: "{job.trigger_token}, your description here"',
            )
        ]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# ============================================================================
# Server Instance
# ============================================================================


class LoraTrainerServer:
    """Wrapper class for launcher.py integration."""

    def __init__(self):
        self.mcp = mcp
        self.logger = logger

    def get_mcp(self):
        """Get the FastMCP instance."""
        return self.mcp


server = LoraTrainerServer()


def get_mcp():
    """Get the FastMCP server instance - required by launcher.py."""
    return server.get_mcp()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
