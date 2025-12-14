# ABOUTME: MCP server for LoRA training management.
# Provides tools for dataset creation, image upload, training control, and checkpoint promotion.

import base64
import logging
from pathlib import Path
from typing import List, Literal, Optional

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ImageContent, TextContent

from lora_trainer.dataset_manager import DatasetManager
from lora_trainer.job_store import create_job_store
from lora_trainer.job_store_base import JobStoreBase
from lora_trainer.models import LoRAType, TrainingConfig, TrainingStatus
from lora_trainer.training_runner import TrainingRunner


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    ctx: Context = None,
) -> List[TextContent]:
    """Create a new LoRA training dataset.

    TRIGGER TOKEN GUIDELINES:
    - Use unique, non-dictionary words (e.g., "ohwx", "sks", "xyz123")
    - Keep it short (3-6 characters)
    - Avoid common words that appear in training data

    LORA TYPES:
    - character: Person/subject identity (faces, full body)
    - style: Artistic style transfer (painting style, color palette)
    - concept: Object or abstract concept

    Args:
        name: Dataset name (alphanumeric + underscores, starts with letter)
        trigger_token: Unique token to trigger the LoRA (e.g., "ohwx")
        lora_type: Type of LoRA being trained
        description: Optional description

    Returns:
        Confirmation with dataset path.
    """
    try:
        dm = _get_dataset_manager()
        dm.create_dataset(
            name=name,
            trigger_token=trigger_token,
            lora_type=LoRAType(lora_type),
            description=description,
        )
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
    ctx: Context = None,
) -> List[TextContent]:
    """List all available training datasets.

    Returns:
        List of datasets with metadata.
    """
    datasets = _get_dataset_manager().list_datasets()
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
