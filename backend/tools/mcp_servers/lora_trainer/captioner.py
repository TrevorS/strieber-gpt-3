# ABOUTME: Auto-captioning for LoRA training using Qwen-VL vision model.
# ABOUTME: Supports detailed descriptions, simple captions, and tag-based styles.

import base64
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Literal, Optional

import httpx

logger = logging.getLogger(__name__)

# Default Qwen-VL URL (internal Docker network port)
DEFAULT_QWEN_VL_URL = "http://llama-server-qwen-vl:8000"

# Caption prompts for different styles
PROMPTS = {
    "detailed": (
        "Describe this image in detail for training an AI model. "
        "Include subject appearance, clothing, pose, expression, background, and lighting."
    ),
    "simple": "Describe this image briefly in one short phrase.",
    "tags": (
        "List tags describing this image, separated by commas. "
        "Include subject, clothing, pose, setting, style."
    ),
}


class VisionCaptioner:
    """Generate image captions using Qwen-VL vision model."""

    def __init__(self, base_url: Optional[str] = None):
        """Initialize vision captioner.

        Args:
            base_url: Base URL for Qwen-VL vision model server.
                      Defaults to QWEN_VL_URL env var or internal Docker URL.
        """
        if base_url is None:
            base_url = os.environ.get("QWEN_VL_URL", DEFAULT_QWEN_VL_URL)
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=60.0)

    async def caption_image(
        self,
        image_bytes: bytes,
        style: Literal["detailed", "simple", "tags"] = "detailed",
        trigger_token: Optional[str] = None,
        custom_prompt: Optional[str] = None,
    ) -> str:
        """Generate caption for a single image.

        Args:
            image_bytes: Raw image bytes (PNG or JPEG).
            style: Caption style - detailed, simple, or tags.
            trigger_token: Optional token to prepend to caption.
            custom_prompt: Override style-based prompt with custom text.

        Returns:
            Generated caption.

        Raises:
            httpx.HTTPError: If API request fails.
            ValueError: If style is invalid.
        """
        if custom_prompt is None and style not in PROMPTS:
            raise ValueError(
                f"Invalid style '{style}'. Must be one of: {list(PROMPTS.keys())}"
            )

        # Encode image to base64
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Build OpenAI-compatible vision request
        prompt = custom_prompt if custom_prompt else PROMPTS[style]
        request = {
            "model": "qwen3-vl-2b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                        },
                    ],
                }
            ],
            "max_tokens": 200,
        }

        # Call vision model API
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=request,
            )
            response.raise_for_status()
            result = response.json()

            # Extract caption from response
            caption = result["choices"][0]["message"]["content"].strip()

            # Prepend trigger token if provided
            if trigger_token:
                caption = f"{trigger_token} {caption}"

            return caption

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Vision API error: {e.response.status_code} - {e.response.text}"
            )
            raise
        except KeyError as e:
            logger.error(f"Unexpected API response format: {e}")
            raise ValueError(f"Invalid API response format: missing {e}")

    async def caption_dataset(
        self,
        images_dir: Path,
        captions_dir: Path,
        trigger_token: str,
        style: Literal["detailed", "simple", "tags"] = "detailed",
        overwrite: bool = False,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, str]:
        """Caption all images in a directory.

        Args:
            images_dir: Directory containing images to caption.
            captions_dir: Directory to save caption files.
            trigger_token: Token to prepend to all captions.
            style: Caption style - detailed, simple, or tags.
            overwrite: If True, regenerate existing captions.
            on_progress: Optional callback(current, total, image_name).

        Returns:
            Dict mapping image names to generated captions.

        Raises:
            ValueError: If images_dir doesn't exist.
        """
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")

        # Ensure captions directory exists
        captions_dir.mkdir(parents=True, exist_ok=True)

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
            logger.warning(f"No images found in {images_dir}")
            return {}

        logger.info(
            f"Captioning {total} images with style='{style}', trigger='{trigger_token}'"
        )

        results = {}
        for idx, image_path in enumerate(image_files, start=1):
            image_name = image_path.name
            caption_name = f"{image_path.stem}.txt"
            caption_path = captions_dir / caption_name

            # Skip if caption already exists (unless overwrite=True)
            if caption_path.exists() and not overwrite:
                logger.debug(f"Skipping {image_name} (caption already exists)")
                with open(caption_path) as f:
                    results[image_name] = f.read()
                if on_progress:
                    on_progress(idx, total, image_name)
                continue

            # Generate caption
            try:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()

                caption = await self.caption_image(
                    image_bytes=image_bytes,
                    style=style,
                    trigger_token=trigger_token,
                )

                # Save caption to file
                with open(caption_path, "w") as f:
                    f.write(caption)

                results[image_name] = caption
                logger.info(f"[{idx}/{total}] {image_name}: {caption[:60]}...")

                if on_progress:
                    on_progress(idx, total, image_name)

            except Exception as e:
                logger.error(f"Failed to caption {image_name}: {e}")
                # Continue with other images even if one fails
                continue

        logger.info(f"Captioning complete: {len(results)}/{total} images")
        return results

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
