# ABOUTME: Image utilities for LoRA training datasets.
# ABOUTME: Provides URL fetching, smart cropping with face detection, and batch operations.

import asyncio
import io
import logging
from typing import Callable, List, Literal, Optional, Tuple, Union

import httpx
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


class ImageFetchError(Exception):
    """Raised when image fetching fails."""

    pass


class ImageProcessingError(Exception):
    """Raised when image processing fails."""

    pass


async def fetch_image(url: str, timeout: float = 30.0) -> bytes:
    """Fetch image from URL, validate it's actually an image.

    Args:
        url: URL to fetch image from.
        timeout: Request timeout in seconds.

    Returns:
        Raw image bytes.

    Raises:
        ImageFetchError: If fetch fails, timeout occurs, or content is not an image.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)

            # Check HTTP status
            if response.status_code == 404:
                raise ImageFetchError(f"Image not found (404): {url}")
            elif response.status_code != 200:
                raise ImageFetchError(
                    f"HTTP {response.status_code} fetching image: {url}"
                )

            # Validate Content-Type
            content_type = response.headers.get("content-type", "").lower()
            if not content_type.startswith("image/"):
                raise ImageFetchError(
                    f"URL does not return an image (Content-Type: {content_type}): {url}"
                )

            # Validate supported formats
            supported = ["image/png", "image/jpeg", "image/jpg", "image/webp"]
            if not any(content_type.startswith(fmt) for fmt in supported):
                raise ImageFetchError(
                    f"Unsupported image format (Content-Type: {content_type}): {url}. "
                    f"Supported: PNG, JPEG, WebP"
                )

            image_bytes = response.content

            # Verify we can actually open it as an image
            try:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    img.verify()
            except Exception as e:
                raise ImageFetchError(f"Invalid image data from {url}: {e}")

            logger.info(
                f"Fetched image from {url}: {len(image_bytes)} bytes, {content_type}"
            )
            return image_bytes

    except httpx.TimeoutException:
        raise ImageFetchError(f"Timeout fetching image from {url}")
    except httpx.RequestError as e:
        raise ImageFetchError(f"Request error fetching image from {url}: {e}")
    except ImageFetchError:
        raise
    except Exception as e:
        raise ImageFetchError(f"Unexpected error fetching image from {url}: {e}")


def _detect_face(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Detect face in image using mediapipe.

    Args:
        image: PIL Image to detect face in.

    Returns:
        Tuple of (x, y, width, height) for detected face bounding box, or None.
    """
    try:
        import mediapipe as mp

        # Convert PIL to RGB numpy array
        import numpy as np

        img_array = np.array(image.convert("RGB"))

        # Initialize face detection
        with mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # Full-range model (better for various distances)
            min_detection_confidence=0.5,
        ) as face_detection:
            results = face_detection.process(img_array)

            if not results.detections:
                return None

            # Get first (most prominent) detection
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            # Convert relative coords to absolute pixels
            h, w = img_array.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            logger.info(f"Detected face at ({x}, {y}, {width}, {height})")
            return (x, y, width, height)

    except ImportError:
        logger.warning("mediapipe not installed, face detection unavailable")
        return None
    except Exception as e:
        logger.warning(f"Face detection failed: {e}")
        return None


def smart_crop(
    image_bytes: bytes,
    target_size: int = 1024,
    crop_mode: Literal["center", "smart", "none"] = "smart",
) -> bytes:
    """Crop and resize image optimally for LoRA training.

    For character LoRAs, use smart mode to center-crop around detected faces.
    Falls back to center crop if no face detected or face detection unavailable.

    Args:
        image_bytes: Raw image bytes (PNG, JPEG, WebP).
        target_size: Output size (square). Default 1024.
        crop_mode: Cropping strategy:
            - "smart": Detect face and crop around it (fallback to center)
            - "center": Simple center crop
            - "none": No crop, just resize (may distort aspect ratio)

    Returns:
        Processed image as PNG bytes with EXIF removed.

    Raises:
        ImageProcessingError: If processing fails.
    """
    try:
        # Load image
        with Image.open(io.BytesIO(image_bytes)) as img:
            # Convert to RGB (handles RGBA, grayscale, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Remove EXIF data for privacy
            img = ImageOps.exif_transpose(img)
            img_data = img.getdata()
            img_without_exif = Image.new(img.mode, img.size)
            img_without_exif.putdata(img_data)
            img = img_without_exif

            original_width, original_height = img.size
            logger.info(
                f"Processing image: {original_width}x{original_height}, mode={crop_mode}"
            )

            if crop_mode == "none":
                # Just resize to target (may distort)
                img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            else:
                # Determine crop center
                if crop_mode == "smart":
                    face_bbox = _detect_face(img)
                    if face_bbox:
                        # Center crop around face
                        x, y, width, height = face_bbox
                        face_center_x = x + width // 2
                        face_center_y = y + height // 2

                        # Expand crop area around face center
                        # Use 2x face height or 1.5x face width, whichever is larger
                        crop_size = max(int(width * 1.5), int(height * 2))

                        # Use larger dimension as crop size (for square crop)
                        min_dim = min(original_width, original_height)
                        crop_size = min(
                            max(crop_size, min_dim // 2), min_dim
                        )  # Reasonable bounds

                        center_x = face_center_x
                        center_y = face_center_y
                        logger.info(
                            f"Smart crop centered on face: ({center_x}, {center_y})"
                        )
                    else:
                        # No face detected, fall back to center
                        center_x = original_width // 2
                        center_y = original_height // 2
                        crop_size = min(original_width, original_height)
                        logger.info("No face detected, using center crop")
                else:
                    # Simple center crop
                    center_x = original_width // 2
                    center_y = original_height // 2
                    crop_size = min(original_width, original_height)

                # Calculate crop box
                left = max(0, center_x - crop_size // 2)
                top = max(0, center_y - crop_size // 2)
                right = min(original_width, center_x + crop_size // 2)
                bottom = min(original_height, center_y + crop_size // 2)

                # Adjust if we hit edges
                if right - left < crop_size:
                    if left == 0:
                        right = min(original_width, left + crop_size)
                    else:
                        left = max(0, right - crop_size)

                if bottom - top < crop_size:
                    if top == 0:
                        bottom = min(original_height, top + crop_size)
                    else:
                        top = max(0, bottom - crop_size)

                # Crop and resize
                img = img.crop((left, top, right, bottom))
                img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

            # Convert to PNG bytes
            output = io.BytesIO()
            img.save(output, format="PNG", optimize=True)
            result_bytes = output.getvalue()

            logger.info(
                f"Processed image: {target_size}x{target_size}, {len(result_bytes)} bytes"
            )
            return result_bytes

    except Exception as e:
        raise ImageProcessingError(f"Failed to process image: {e}")


async def fetch_images_batch(
    urls: List[str],
    max_concurrent: int = 5,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> List[Tuple[str, Union[bytes, Exception]]]:
    """Fetch multiple images concurrently with rate limiting.

    Args:
        urls: List of image URLs to fetch.
        max_concurrent: Maximum number of concurrent requests.
        on_progress: Optional callback (completed_count, total_count, current_url).

    Returns:
        List of (url, result) tuples where result is bytes on success or Exception on failure.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results: List[Tuple[str, Union[bytes, Exception]]] = []
    completed = 0
    total = len(urls)

    async def fetch_one(url: str) -> Tuple[str, Union[bytes, Exception]]:
        nonlocal completed
        async with semaphore:
            try:
                image_bytes = await fetch_image(url)
                result = (url, image_bytes)
            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                result = (url, e)

            completed += 1
            if on_progress:
                on_progress(completed, total, url)

            return result

    # Fetch all concurrently
    tasks = [fetch_one(url) for url in urls]
    results = await asyncio.gather(*tasks)

    # Log summary
    success_count = sum(1 for _, result in results if isinstance(result, bytes))
    logger.info(f"Batch fetch complete: {success_count}/{total} succeeded")

    return results
