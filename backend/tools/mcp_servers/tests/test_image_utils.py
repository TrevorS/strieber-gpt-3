# ABOUTME: Unit tests for image_utils module.
# Tests URL fetching, smart cropping with face detection, and batch operations.

import io
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from lora_trainer.image_utils import (
    ImageFetchError,
    ImageProcessingError,
    fetch_image,
    fetch_images_batch,
    smart_crop,
)


def create_test_image(width: int = 100, height: int = 100, mode: str = "RGB") -> bytes:
    """Create a minimal valid image for testing."""
    img = Image.new(mode, (width, height), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestFetchImage:
    """Tests for fetch_image function."""

    @pytest.mark.asyncio
    async def test_fetch_image_success(self):
        """Test successful image fetch."""
        image_bytes = create_test_image()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/png"}
        mock_response.content = image_bytes

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await fetch_image("https://example.com/image.png")

        assert result == image_bytes

    @pytest.mark.asyncio
    async def test_fetch_image_404_error(self):
        """Test 404 response raises ImageFetchError."""
        mock_response = Mock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(ImageFetchError, match="not found"):
                await fetch_image("https://example.com/missing.png")

    @pytest.mark.asyncio
    async def test_fetch_image_wrong_content_type(self):
        """Test non-image content type raises ImageFetchError."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(ImageFetchError, match="does not return an image"):
                await fetch_image("https://example.com/page.html")

    @pytest.mark.asyncio
    async def test_fetch_image_unsupported_format(self):
        """Test unsupported image format raises ImageFetchError."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/gif"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(ImageFetchError, match="Unsupported image format"):
                await fetch_image("https://example.com/animation.gif")

    @pytest.mark.asyncio
    async def test_fetch_image_timeout(self):
        """Test timeout raises ImageFetchError."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(ImageFetchError, match="Timeout"):
                await fetch_image("https://example.com/slow.png")

    @pytest.mark.asyncio
    async def test_fetch_image_request_error(self):
        """Test request error raises ImageFetchError."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(ImageFetchError, match="Request error"):
                await fetch_image("https://example.com/image.png")

    @pytest.mark.asyncio
    async def test_fetch_image_invalid_data(self):
        """Test invalid image data raises ImageFetchError."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/png"}
        mock_response.content = b"not an image"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(ImageFetchError, match="Invalid image data"):
                await fetch_image("https://example.com/corrupt.png")

    @pytest.mark.asyncio
    async def test_fetch_image_jpeg_content_type(self):
        """Test JPEG content type is accepted."""
        # Create a JPEG image
        img = Image.new("RGB", (100, 100), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.content = image_bytes

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await fetch_image("https://example.com/photo.jpg")

        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_fetch_image_webp_content_type(self):
        """Test WebP content type is accepted."""
        # Create a WebP image
        img = Image.new("RGB", (100, 100), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="WEBP")
        image_bytes = buffer.getvalue()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/webp"}
        mock_response.content = image_bytes

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await fetch_image("https://example.com/image.webp")

        assert len(result) > 0


class TestSmartCrop:
    """Tests for smart_crop function."""

    def test_smart_crop_center_mode(self):
        """Test center crop mode."""
        # Create a 200x100 image (wider than tall)
        image_bytes = create_test_image(200, 100)

        result = smart_crop(image_bytes, target_size=64, crop_mode="center")

        # Verify output is valid PNG
        img = Image.open(io.BytesIO(result))
        assert img.size == (64, 64)
        assert img.format == "PNG"

    def test_smart_crop_none_mode(self):
        """Test no-crop resize mode (may distort)."""
        # Create a 200x100 image
        image_bytes = create_test_image(200, 100)

        result = smart_crop(image_bytes, target_size=64, crop_mode="none")

        img = Image.open(io.BytesIO(result))
        assert img.size == (64, 64)

    def test_smart_crop_smart_mode_no_face(self):
        """Test smart mode falls back to center when no face detected."""
        # Create a simple colored image (no face)
        image_bytes = create_test_image(200, 200)

        result = smart_crop(image_bytes, target_size=64, crop_mode="smart")

        img = Image.open(io.BytesIO(result))
        assert img.size == (64, 64)

    def test_smart_crop_rgba_to_rgb(self):
        """Test RGBA images are converted to RGB."""
        # Create RGBA image
        image_bytes = create_test_image(100, 100, mode="RGBA")

        result = smart_crop(image_bytes, target_size=64, crop_mode="center")

        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGB"
        assert img.size == (64, 64)

    def test_smart_crop_grayscale_to_rgb(self):
        """Test grayscale images are converted to RGB."""
        # Create grayscale image
        image_bytes = create_test_image(100, 100, mode="L")

        result = smart_crop(image_bytes, target_size=64, crop_mode="center")

        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGB"
        assert img.size == (64, 64)

    def test_smart_crop_large_image(self):
        """Test cropping a large image."""
        # Create 2048x1024 image
        image_bytes = create_test_image(2048, 1024)

        result = smart_crop(image_bytes, target_size=512, crop_mode="center")

        img = Image.open(io.BytesIO(result))
        assert img.size == (512, 512)

    def test_smart_crop_small_image(self):
        """Test handling of smaller-than-target images."""
        # Create 50x50 image (smaller than typical target)
        image_bytes = create_test_image(50, 50)

        result = smart_crop(image_bytes, target_size=64, crop_mode="center")

        img = Image.open(io.BytesIO(result))
        assert img.size == (64, 64)

    def test_smart_crop_preserves_quality(self):
        """Test that crop preserves reasonable image quality."""
        # Create test image
        image_bytes = create_test_image(512, 512)

        result = smart_crop(image_bytes, target_size=256, crop_mode="center")

        # Result should be reasonably sized (not tiny)
        assert len(result) > 100

    def test_smart_crop_invalid_image(self):
        """Test that invalid image data raises ImageProcessingError."""
        with pytest.raises(ImageProcessingError, match="Failed to process"):
            smart_crop(b"not an image", target_size=64, crop_mode="center")

    def test_smart_crop_default_size(self):
        """Test default target size of 1024."""
        image_bytes = create_test_image(2000, 2000)

        result = smart_crop(image_bytes, crop_mode="center")

        img = Image.open(io.BytesIO(result))
        assert img.size == (1024, 1024)

    def test_smart_crop_jpeg_input(self):
        """Test processing JPEG input."""
        # Create JPEG image
        img = Image.new("RGB", (200, 200), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        result = smart_crop(image_bytes, target_size=64, crop_mode="center")

        output_img = Image.open(io.BytesIO(result))
        assert output_img.size == (64, 64)
        assert output_img.format == "PNG"  # Output is always PNG


class TestSmartCropWithFaceDetection:
    """Tests for smart crop face detection (requires mediapipe)."""

    def test_smart_crop_with_mock_face_detection(self):
        """Test smart crop when face is detected (mocked)."""
        image_bytes = create_test_image(400, 400)

        # Mock face detection to return a face bbox
        with patch("lora_trainer.image_utils._detect_face") as mock_detect:
            # Return face at center-right of image
            mock_detect.return_value = (250, 150, 100, 100)

            result = smart_crop(image_bytes, target_size=128, crop_mode="smart")

            img = Image.open(io.BytesIO(result))
            assert img.size == (128, 128)
            mock_detect.assert_called_once()

    def test_smart_crop_face_detection_failure_fallback(self):
        """Test smart crop falls back to center when face detection fails."""
        image_bytes = create_test_image(400, 400)

        # Mock face detection to return None (no face found)
        with patch("lora_trainer.image_utils._detect_face") as mock_detect:
            mock_detect.return_value = None

            result = smart_crop(image_bytes, target_size=128, crop_mode="smart")

            img = Image.open(io.BytesIO(result))
            assert img.size == (128, 128)


class TestFetchImagesBatch:
    """Tests for fetch_images_batch function."""

    @pytest.mark.asyncio
    async def test_fetch_batch_all_success(self):
        """Test batch fetch with all successful URLs."""
        image_bytes = create_test_image()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/png"}
        mock_response.content = image_bytes

        with patch("lora_trainer.image_utils.fetch_image") as mock_fetch:
            mock_fetch.return_value = image_bytes

            urls = [
                "https://example.com/1.png",
                "https://example.com/2.png",
                "https://example.com/3.png",
            ]
            results = await fetch_images_batch(urls)

        assert len(results) == 3
        for url, result in results:
            assert isinstance(result, bytes)

    @pytest.mark.asyncio
    async def test_fetch_batch_partial_failure(self):
        """Test batch fetch with some failures."""
        image_bytes = create_test_image()

        call_count = 0

        async def mock_fetch(url: str, timeout: float = 30.0):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ImageFetchError("Network error")
            return image_bytes

        with patch("lora_trainer.image_utils.fetch_image", mock_fetch):
            urls = [
                "https://example.com/1.png",
                "https://example.com/2.png",
                "https://example.com/3.png",
            ]
            results = await fetch_images_batch(urls)

        assert len(results) == 3

        # First and third should succeed
        assert isinstance(results[0][1], bytes)
        assert isinstance(results[2][1], bytes)

        # Second should be an exception
        assert isinstance(results[1][1], Exception)

    @pytest.mark.asyncio
    async def test_fetch_batch_empty_list(self):
        """Test batch fetch with empty URL list."""
        results = await fetch_images_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_batch_progress_callback(self):
        """Test progress callback is called for each URL."""
        image_bytes = create_test_image()

        with patch("lora_trainer.image_utils.fetch_image") as mock_fetch:
            mock_fetch.return_value = image_bytes

            progress_calls = []

            def on_progress(completed: int, total: int, url: str):
                progress_calls.append((completed, total, url))

            urls = [
                "https://example.com/1.png",
                "https://example.com/2.png",
            ]
            await fetch_images_batch(urls, on_progress=on_progress)

        assert len(progress_calls) == 2
        # Order may vary due to concurrency, but all should be reported
        assert any(call[2] == "https://example.com/1.png" for call in progress_calls)
        assert any(call[2] == "https://example.com/2.png" for call in progress_calls)

    @pytest.mark.asyncio
    async def test_fetch_batch_respects_concurrency_limit(self):
        """Test that max_concurrent is respected."""
        image_bytes = create_test_image()
        concurrent_calls = 0
        max_concurrent_observed = 0

        async def mock_fetch(url: str, timeout: float = 30.0):
            nonlocal concurrent_calls, max_concurrent_observed
            concurrent_calls += 1
            max_concurrent_observed = max(max_concurrent_observed, concurrent_calls)
            # Simulate some work
            import asyncio

            await asyncio.sleep(0.01)
            concurrent_calls -= 1
            return image_bytes

        with patch("lora_trainer.image_utils.fetch_image", mock_fetch):
            urls = [f"https://example.com/{i}.png" for i in range(10)]
            await fetch_images_batch(urls, max_concurrent=3)

        # Should never exceed max_concurrent
        assert max_concurrent_observed <= 3

    @pytest.mark.asyncio
    async def test_fetch_batch_all_failures(self):
        """Test batch fetch when all URLs fail."""

        async def mock_fetch(url: str, timeout: float = 30.0):
            raise ImageFetchError("All fail")

        with patch("lora_trainer.image_utils.fetch_image", mock_fetch):
            urls = [
                "https://example.com/1.png",
                "https://example.com/2.png",
            ]
            results = await fetch_images_batch(urls)

        assert len(results) == 2
        for url, result in results:
            assert isinstance(result, Exception)
