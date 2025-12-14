"""Example usage of VisionCaptioner for LoRA training datasets.

This demonstrates how to auto-caption images in a dataset using the Qwen-VL
vision model.

Run this example:
    uv run python -m lora_trainer.example_caption_usage
"""

import asyncio
from pathlib import Path

from lora_trainer.captioner import VisionCaptioner


async def caption_single_image_example():
    """Example: Caption a single image."""
    print("=" * 60)
    print("Example 1: Caption a single image")
    print("=" * 60)

    async with VisionCaptioner() as captioner:
        # Read an image file
        with open("path/to/image.jpg", "rb") as f:
            image_bytes = f.read()

        # Generate a detailed caption with trigger token
        caption = await captioner.caption_image(
            image_bytes=image_bytes,
            style="detailed",
            trigger_token="ohwx",
        )

        print(f"Caption: {caption}")


async def caption_dataset_example():
    """Example: Caption all images in a dataset."""
    print("\n" + "=" * 60)
    print("Example 2: Caption entire dataset")
    print("=" * 60)

    async with VisionCaptioner() as captioner:
        # Define paths
        dataset_dir = Path("/datasets/my_character")
        images_dir = dataset_dir / "images"
        captions_dir = dataset_dir / "captions"

        # Progress callback
        def on_progress(current: int, total: int, image_name: str):
            print(f"[{current}/{total}] Captioned: {image_name}")

        # Caption all images
        results = await captioner.caption_dataset(
            images_dir=images_dir,
            captions_dir=captions_dir,
            trigger_token="ohwx",
            style="detailed",
            overwrite=False,  # Skip images that already have captions
            on_progress=on_progress,
        )

        print(f"\nCaptioned {len(results)} images")


async def caption_styles_example():
    """Example: Different caption styles."""
    print("\n" + "=" * 60)
    print("Example 3: Different caption styles")
    print("=" * 60)

    async with VisionCaptioner() as captioner:
        with open("path/to/image.jpg", "rb") as f:
            image_bytes = f.read()

        # Detailed style (default)
        detailed = await captioner.caption_image(
            image_bytes, style="detailed", trigger_token="ohwx"
        )
        print(f"Detailed: {detailed}")

        # Simple style (short phrase)
        simple = await captioner.caption_image(
            image_bytes, style="simple", trigger_token="ohwx"
        )
        print(f"Simple: {simple}")

        # Tags style (comma-separated, booru-style)
        tags = await captioner.caption_image(
            image_bytes, style="tags", trigger_token="ohwx"
        )
        print(f"Tags: {tags}")


async def main():
    """Run all examples."""
    try:
        await caption_single_image_example()
    except FileNotFoundError:
        print("(Skipped - image file not found)")

    try:
        await caption_dataset_example()
    except ValueError:
        print("(Skipped - dataset directory not found)")

    try:
        await caption_styles_example()
    except FileNotFoundError:
        print("(Skipped - image file not found)")


if __name__ == "__main__":
    asyncio.run(main())
