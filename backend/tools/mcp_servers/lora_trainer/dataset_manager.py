# ABOUTME: Dataset management for LoRA training.
# Handles dataset creation, image uploads, and validation.

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from lora_trainer.models import DatasetMetadata, LoRAType


def _detect_image_type(data: bytes) -> Optional[str]:
    """Detect image type from file header bytes.

    Args:
        data: Raw image bytes.

    Returns:
        'png', 'jpeg', or None if not recognized.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:2] == b"\xff\xd8":
        return "jpeg"
    return None


logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages LoRA training datasets on disk."""

    def __init__(self, datasets_path: Path):
        """Initialize dataset manager.

        Args:
            datasets_path: Base path for all datasets.
        """
        self.datasets_path = Path(datasets_path)
        self.datasets_path.mkdir(parents=True, exist_ok=True)

    def _validate_name(self, name: str) -> None:
        """Validate dataset name (alphanumeric + underscores)."""
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
            raise ValueError(
                f"Invalid name '{name}': must start with a letter, "
                "contain only letters, numbers, and underscores"
            )

    def get_dataset_path(self, name: str) -> Path:
        """Get path to a dataset directory."""
        return self.datasets_path / name

    def dataset_exists(self, name: str) -> bool:
        """Check if a dataset exists."""
        return (self.get_dataset_path(name) / "metadata.json").exists()

    def create_dataset(
        self,
        name: str,
        trigger_token: str,
        lora_type: LoRAType,
        description: Optional[str] = None,
    ) -> DatasetMetadata:
        """Create a new dataset.

        Args:
            name: Dataset name (alphanumeric + underscores).
            trigger_token: Unique token to trigger the LoRA.
            lora_type: Type of LoRA (character, style, concept).
            description: Optional description.

        Returns:
            Created dataset metadata.

        Raises:
            ValueError: If name is invalid or dataset already exists.
        """
        self._validate_name(name)

        dataset_path = self.get_dataset_path(name)
        if dataset_path.exists():
            raise ValueError(f"Dataset '{name}' already exists")

        # Create directory structure
        dataset_path.mkdir(parents=True)
        (dataset_path / "images").mkdir()
        (dataset_path / "captions").mkdir()

        # Create metadata
        metadata = DatasetMetadata(
            name=name,
            trigger_token=trigger_token,
            lora_type=lora_type,
            description=description,
        )

        # Save metadata
        self._save_metadata(name, metadata)

        logger.info(f"Created dataset '{name}' with trigger '{trigger_token}'")
        return metadata

    def _save_metadata(self, name: str, metadata: DatasetMetadata) -> None:
        """Save dataset metadata to disk."""
        metadata_path = self.get_dataset_path(name) / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.model_dump(mode="json"), f, indent=2, default=str)

    def get_metadata(self, name: str) -> DatasetMetadata:
        """Load dataset metadata.

        Args:
            name: Dataset name.

        Returns:
            Dataset metadata.

        Raises:
            ValueError: If dataset doesn't exist.
        """
        metadata_path = self.get_dataset_path(name) / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Dataset '{name}' not found")

        with open(metadata_path) as f:
            data = json.load(f)

        return DatasetMetadata(**data)

    def add_image(
        self,
        name: str,
        image_bytes: bytes,
        caption: Optional[str] = None,
    ) -> str:
        """Add an image to a dataset.

        Args:
            name: Dataset name.
            image_bytes: Raw image bytes (PNG or JPEG).
            caption: Optional caption for the image.

        Returns:
            Saved filename.

        Raises:
            ValueError: If dataset doesn't exist or image format invalid.
        """
        dataset_path = self.get_dataset_path(name)
        if not dataset_path.exists():
            raise ValueError(f"Dataset '{name}' not found")

        # Detect image format
        img_type = _detect_image_type(image_bytes)
        if img_type not in ("png", "jpeg"):
            raise ValueError(f"Invalid image format: {img_type}. Must be PNG or JPEG.")

        ext = "png" if img_type == "png" else "jpg"

        # Generate filename from content hash for deduplication
        content_hash = hashlib.md5(image_bytes).hexdigest()[:8]
        images_dir = dataset_path / "images"
        existing = list(images_dir.glob(f"*_{content_hash}.*"))
        if existing:
            # Already have this image
            return existing[0].name

        # Find next index
        existing_images = list(images_dir.glob("*.*"))
        next_idx = len(existing_images) + 1
        filename = f"{next_idx:03d}_{content_hash}.{ext}"

        # Save image
        image_path = images_dir / filename
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # Save caption if provided
        if caption:
            caption_path = (
                dataset_path / "captions" / f"{filename.rsplit('.', 1)[0]}.txt"
            )
            with open(caption_path, "w") as f:
                f.write(caption)

        # Update metadata
        metadata = self.get_metadata(name)
        metadata.image_count = len(list(images_dir.glob("*.*")))
        metadata.has_captions = bool(list((dataset_path / "captions").glob("*.txt")))
        self._save_metadata(name, metadata)

        logger.info(f"Added image {filename} to dataset '{name}'")
        return filename

    def list_datasets(self) -> List[DatasetMetadata]:
        """List all datasets.

        Returns:
            List of dataset metadata.
        """
        datasets = []
        for path in self.datasets_path.iterdir():
            if path.is_dir() and (path / "metadata.json").exists():
                try:
                    datasets.append(self.get_metadata(path.name))
                except Exception as e:
                    logger.warning(f"Failed to load dataset {path.name}: {e}")
        return datasets

    def validate_dataset(self, name: str) -> str:
        """Validate dataset readiness for training.

        Args:
            name: Dataset name.

        Returns:
            Validation report string.
        """
        dataset_path = self.get_dataset_path(name)
        if not dataset_path.exists():
            return f"ERROR: Dataset '{name}' not found"

        metadata = self.get_metadata(name)
        issues = []
        info = []

        # Check image count
        images_dir = dataset_path / "images"
        images = list(images_dir.glob("*.*"))
        image_count = len(images)

        if image_count < 5:
            issues.append(f"Too few images: {image_count} (minimum 5 required)")
        elif image_count < 9:
            info.append(
                f"Image count is low: {image_count} (9+ recommended for best results)"
            )
        else:
            info.append(f"Image count: {image_count} ✓")

        # Check image sizes
        try:
            from PIL import Image

            sizes = set()
            for img_path in images:
                with Image.open(img_path) as img:
                    sizes.add(img.size)

            if len(sizes) > 1:
                info.append(f"Mixed image sizes: {sizes}")
            else:
                size = next(iter(sizes)) if sizes else (0, 0)
                if size[0] < 512 or size[1] < 512:
                    issues.append(f"Images too small: {size} (512x512 minimum)")
                elif size[0] < 1024 or size[1] < 1024:
                    info.append(f"Image size: {size} (1024x1024 recommended)")
                else:
                    info.append(f"Image size: {size} ✓")
        except ImportError:
            info.append("Pillow not installed, skipping size validation")

        # Check captions
        captions_dir = dataset_path / "captions"
        captions = list(captions_dir.glob("*.txt"))
        if captions:
            # Verify trigger token in captions
            trigger_missing = 0
            for caption_path in captions:
                with open(caption_path) as f:
                    content = f.read()
                    if metadata.trigger_token not in content:
                        trigger_missing += 1

            if trigger_missing > 0:
                issues.append(
                    f"{trigger_missing} captions missing trigger token '{metadata.trigger_token}'"
                )
            else:
                info.append(f"All {len(captions)} captions include trigger token ✓")
        else:
            info.append("No captions (will use trigger token only)")

        # Build report
        report = [f"Dataset: {name}"]
        report.append(f"Trigger: {metadata.trigger_token}")
        report.append(f"Type: {metadata.lora_type.value}")
        report.append("")

        if issues:
            report.append("ISSUES:")
            for issue in issues:
                report.append(f"  ✗ {issue}")
            report.append("")

        report.append("STATUS:")
        for item in info:
            report.append(f"  {item}")

        if issues:
            report.append("")
            report.append("Ready for training: NO")
        else:
            report.append("")
            report.append("Ready for training: YES")

        return "\n".join(report)

    def delete_dataset(self, name: str) -> None:
        """Delete a dataset.

        Args:
            name: Dataset name.

        Raises:
            ValueError: If dataset doesn't exist.
        """
        dataset_path = self.get_dataset_path(name)
        if not dataset_path.exists():
            raise ValueError(f"Dataset '{name}' not found")

        import shutil

        shutil.rmtree(dataset_path)
        logger.info(f"Deleted dataset '{name}'")
