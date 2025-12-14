# ABOUTME: Helper script for uploading images to lora_trainer datasets.
# ABOUTME: Can use existing images from a directory or generate synthetic test images.

import argparse
import asyncio
import base64
import io
import json
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.mcp_client import MCPClient


def generate_synthetic_images(count: int = 5, size: int = 512) -> list[tuple[str, str]]:
    """Generate synthetic test images with random colors.

    Returns list of (base64_data, caption) tuples.
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL not installed. Install with: pip install Pillow")
        sys.exit(1)

    images = []

    for i in range(count):
        # Create image with random color gradient
        img = Image.new("RGB", (size, size))

        # Simple gradient fill
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        for y in range(size):
            r = int(128 + 64 * (y / size))
            g = int(64 + 128 * ((i + 1) / count))
            b = int(192 - 64 * (y / size))
            draw.line([(0, y), (size, y)], fill=(r, g, b))

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode()

        caption = f"ohwx test image {i + 1} with gradient pattern"
        images.append((b64, caption))

    return images


def load_images_from_directory(
    directory: Path, trigger_token: str
) -> list[tuple[str, str]]:
    """Load images from a directory and encode as base64.

    Supports:
    - metadata.jsonl (HuggingFace format) with "TOK" placeholder
    - Individual .txt caption files
    - Falls back to trigger + filename

    Returns list of (base64_data, caption) tuples.
    """
    images = []

    # Check for HuggingFace-style metadata.jsonl
    metadata_path = directory / "metadata.jsonl"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            for line in f:
                entry = json.loads(line)
                # Replace TOK placeholder with actual trigger token
                prompt = entry.get("prompt", "").replace("TOK", trigger_token)
                metadata[entry["file_name"]] = prompt
        print(f"Loaded {len(metadata)} captions from metadata.jsonl")

    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for img_path in sorted(directory.glob(ext)):
            # Read and encode image
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            # Priority: metadata.jsonl > .txt file > default
            if img_path.name in metadata:
                caption = metadata[img_path.name]
            else:
                caption_path = img_path.with_suffix(".txt")
                if caption_path.exists():
                    caption = caption_path.read_text().strip()
                else:
                    caption = f"{trigger_token} {img_path.stem}"

            images.append((b64, caption))

    return images


async def upload_images(
    dataset_name: str,
    images: list[tuple[str, str]],
    server: str = "lora_trainer",
) -> None:
    """Upload images to a dataset via MCP."""
    client = MCPClient(server)

    # Prepare arguments
    image_data = [img[0] for img in images]
    captions = [img[1] for img in images]

    print(f"Uploading {len(images)} images to dataset '{dataset_name}'...")

    result = await client.call_tool(
        "lora_upload_images",
        {
            "dataset_name": dataset_name,
            "images": image_data,
            "captions": captions,
        },
    )

    if result.is_error:
        print(f"Error: {result.text()}")
        sys.exit(1)

    print(result.text())


async def main():
    parser = argparse.ArgumentParser(
        description="Upload images to lora_trainer dataset"
    )
    parser.add_argument("dataset_name", help="Name of the dataset to upload to")
    parser.add_argument(
        "--source",
        "-s",
        help="Source directory with images (default: generate synthetic)",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=5,
        help="Number of synthetic images to generate (default: 5)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Size of synthetic images (default: 512)",
    )
    parser.add_argument(
        "--server",
        default="lora_trainer",
        help="MCP server to use (default: lora_trainer)",
    )
    parser.add_argument(
        "--trigger-token",
        "-t",
        required=True,
        help="Trigger token to use in captions (replaces TOK placeholder)",
    )

    args = parser.parse_args()

    if args.source:
        source_dir = Path(args.source)
        if not source_dir.exists():
            print(f"Error: Source directory not found: {source_dir}")
            sys.exit(1)
        images = load_images_from_directory(source_dir, args.trigger_token)
        print(f"Loaded {len(images)} images from {source_dir}")
    else:
        images = generate_synthetic_images(args.count, args.size)
        print(f"Generated {len(images)} synthetic images ({args.size}x{args.size})")

    if not images:
        print("No images to upload")
        sys.exit(1)

    await upload_images(args.dataset_name, images, args.server)


if __name__ == "__main__":
    asyncio.run(main())
