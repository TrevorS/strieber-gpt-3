# ABOUTME: Unit tests for VisionCaptioner module.
# Tests caption generation and batch dataset captioning.

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from lora_trainer.captioner import VisionCaptioner, PROMPTS


class TestVisionCaptioner:
    """Tests for VisionCaptioner."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create a mock httpx AsyncClient."""
        mock = AsyncMock()
        mocker.patch("httpx.AsyncClient", return_value=mock)
        return mock

    @pytest.fixture
    def captioner(self, mock_client):
        """Create a VisionCaptioner instance."""
        return VisionCaptioner(base_url="http://test-server:9020")

    @pytest.mark.asyncio
    async def test_caption_image_detailed(self, captioner, mock_client):
        """Test generating a detailed caption."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "A person standing in a studio with soft lighting"
                    }
                }
            ]
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        image_bytes = b"\x89PNG\r\n\x1a\n"  # Minimal PNG header
        caption = await captioner.caption_image(image_bytes, style="detailed")

        assert caption == "A person standing in a studio with soft lighting"

        # Verify API call
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://test-server:9020/v1/chat/completions"
        request_data = call_args[1]["json"]
        assert request_data["model"] == "qwen3-vl-2b"
        assert request_data["max_tokens"] == 200
        assert len(request_data["messages"]) == 1
        message = request_data["messages"][0]
        assert message["role"] == "user"
        assert len(message["content"]) == 2
        assert message["content"][0]["type"] == "text"
        assert message["content"][0]["text"] == PROMPTS["detailed"]
        assert message["content"][1]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_caption_image_with_trigger_token(self, captioner, mock_client):
        """Test caption generation with trigger token prepended."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "portrait, smiling"}}]
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        image_bytes = b"\x89PNG\r\n\x1a\n"
        caption = await captioner.caption_image(
            image_bytes, style="simple", trigger_token="ohwx"
        )

        assert caption == "ohwx portrait, smiling"

    @pytest.mark.asyncio
    async def test_caption_image_simple_style(self, captioner, mock_client):
        """Test simple caption style."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "portrait photo"}}]
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        image_bytes = b"\x89PNG\r\n\x1a\n"
        caption = await captioner.caption_image(image_bytes, style="simple")

        assert caption == "portrait photo"

        # Verify correct prompt was used
        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["messages"][0]["content"][0]["text"] == PROMPTS["simple"]

    @pytest.mark.asyncio
    async def test_caption_image_tags_style(self, captioner, mock_client):
        """Test tags caption style."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "person, portrait, studio, professional lighting"
                    }
                }
            ]
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        image_bytes = b"\x89PNG\r\n\x1a\n"
        caption = await captioner.caption_image(image_bytes, style="tags")

        assert caption == "person, portrait, studio, professional lighting"

        # Verify correct prompt was used
        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["messages"][0]["content"][0]["text"] == PROMPTS["tags"]

    @pytest.mark.asyncio
    async def test_caption_image_invalid_style(self, captioner, mock_client):
        """Test that invalid style raises ValueError."""
        image_bytes = b"\x89PNG\r\n\x1a\n"
        with pytest.raises(ValueError, match="Invalid style"):
            await captioner.caption_image(image_bytes, style="invalid")

    @pytest.mark.asyncio
    async def test_caption_image_api_error(self, captioner, mock_client):
        """Test handling of API errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Server Error", request=Mock(), response=mock_response
        )
        mock_client.post = AsyncMock(return_value=mock_response)

        image_bytes = b"\x89PNG\r\n\x1a\n"
        with pytest.raises(httpx.HTTPStatusError):
            await captioner.caption_image(image_bytes)

    @pytest.mark.asyncio
    async def test_caption_image_invalid_response(self, captioner, mock_client):
        """Test handling of invalid API response format."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "response"}
        mock_client.post = AsyncMock(return_value=mock_response)

        image_bytes = b"\x89PNG\r\n\x1a\n"
        with pytest.raises(ValueError, match="Invalid API response format"):
            await captioner.caption_image(image_bytes)


class TestBatchCaptioning:
    """Tests for batch dataset captioning."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create a mock httpx AsyncClient."""
        mock = AsyncMock()
        mocker.patch("httpx.AsyncClient", return_value=mock)
        return mock

    @pytest.fixture
    def captioner(self, mock_client):
        """Create a VisionCaptioner instance."""
        return VisionCaptioner(base_url="http://test-server:9020")

    @pytest.fixture
    def sample_images_dir(self, tmp_path: Path):
        """Create a directory with sample images."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # Create minimal PNG files
        png_bytes = b"\x89PNG\r\n\x1a\n"
        (images_dir / "001.png").write_bytes(png_bytes)
        (images_dir / "002.jpg").write_bytes(b"\xff\xd8")  # JPEG header

        return images_dir

    @pytest.mark.asyncio
    async def test_caption_dataset_basic(
        self, captioner, mock_client, sample_images_dir, tmp_path
    ):
        """Test basic dataset captioning."""
        captions_dir = tmp_path / "captions"

        # Mock API responses
        mock_response = Mock()
        mock_response.status_code = 200

        call_count = 0

        def mock_json():
            nonlocal call_count
            call_count += 1
            return {"choices": [{"message": {"content": f"Caption {call_count}"}}]}

        mock_response.json = mock_json
        mock_client.post = AsyncMock(return_value=mock_response)

        results = await captioner.caption_dataset(
            images_dir=sample_images_dir,
            captions_dir=captions_dir,
            trigger_token="ohwx",
            style="detailed",
        )

        assert len(results) == 2
        assert "001.png" in results
        assert "002.jpg" in results
        assert results["001.png"] == "ohwx Caption 1"
        assert results["002.jpg"] == "ohwx Caption 2"

        # Verify captions were saved
        assert (captions_dir / "001.txt").exists()
        assert (captions_dir / "002.txt").exists()
        assert (captions_dir / "001.txt").read_text() == "ohwx Caption 1"
        assert (captions_dir / "002.txt").read_text() == "ohwx Caption 2"

    @pytest.mark.asyncio
    async def test_caption_dataset_skip_existing(
        self, captioner, mock_client, sample_images_dir, tmp_path
    ):
        """Test that existing captions are skipped unless overwrite=True."""
        captions_dir = tmp_path / "captions"
        captions_dir.mkdir()

        # Create existing caption for first image
        (captions_dir / "001.txt").write_text("ohwx Existing caption")

        # Mock API response (should only be called once)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "New caption"}}]
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        results = await captioner.caption_dataset(
            images_dir=sample_images_dir,
            captions_dir=captions_dir,
            trigger_token="ohwx",
            style="detailed",
            overwrite=False,
        )

        # Should have both captions, but only one was generated
        assert len(results) == 2
        assert results["001.png"] == "ohwx Existing caption"  # Kept existing
        assert results["002.jpg"] == "ohwx New caption"  # Generated new

        # API should only be called once (for 002.jpg)
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_caption_dataset_overwrite_existing(
        self, captioner, mock_client, sample_images_dir, tmp_path
    ):
        """Test that overwrite=True regenerates all captions."""
        captions_dir = tmp_path / "captions"
        captions_dir.mkdir()

        # Create existing caption
        (captions_dir / "001.txt").write_text("ohwx Old caption")

        # Mock API responses
        mock_response = Mock()
        mock_response.status_code = 200

        call_count = 0

        def mock_json():
            nonlocal call_count
            call_count += 1
            return {"choices": [{"message": {"content": f"New caption {call_count}"}}]}

        mock_response.json = mock_json
        mock_client.post = AsyncMock(return_value=mock_response)

        results = await captioner.caption_dataset(
            images_dir=sample_images_dir,
            captions_dir=captions_dir,
            trigger_token="ohwx",
            style="detailed",
            overwrite=True,
        )

        assert len(results) == 2
        assert results["001.png"] == "ohwx New caption 1"  # Regenerated
        assert results["002.jpg"] == "ohwx New caption 2"  # Generated

        # API should be called twice
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_caption_dataset_progress_callback(
        self, captioner, mock_client, sample_images_dir, tmp_path
    ):
        """Test progress callback is called during batch captioning."""
        captions_dir = tmp_path / "captions"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Caption"}}]
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        progress_calls = []

        def on_progress(current, total, image_name):
            progress_calls.append((current, total, image_name))

        await captioner.caption_dataset(
            images_dir=sample_images_dir,
            captions_dir=captions_dir,
            trigger_token="ohwx",
            on_progress=on_progress,
        )

        assert len(progress_calls) == 2
        assert progress_calls[0] == (1, 2, "001.png")
        assert progress_calls[1] == (2, 2, "002.jpg")

    @pytest.mark.asyncio
    async def test_caption_dataset_missing_directory(self, captioner, tmp_path):
        """Test that missing images directory raises ValueError."""
        with pytest.raises(ValueError, match="Images directory not found"):
            await captioner.caption_dataset(
                images_dir=tmp_path / "nonexistent",
                captions_dir=tmp_path / "captions",
                trigger_token="ohwx",
            )

    @pytest.mark.asyncio
    async def test_caption_dataset_empty_directory(
        self, captioner, mock_client, tmp_path
    ):
        """Test handling of empty images directory."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        captions_dir = tmp_path / "captions"

        results = await captioner.caption_dataset(
            images_dir=images_dir,
            captions_dir=captions_dir,
            trigger_token="ohwx",
        )

        assert results == {}
        assert mock_client.post.call_count == 0

    @pytest.mark.asyncio
    async def test_caption_dataset_partial_failure(
        self, captioner, mock_client, sample_images_dir, tmp_path
    ):
        """Test that failures for individual images don't stop batch processing."""
        captions_dir = tmp_path / "captions"

        # First call succeeds, second fails
        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Success caption"}}]
                }
                return mock_response
            else:
                mock_response = Mock()
                mock_response.status_code = 500
                mock_response.text = "Server error"
                mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "500", request=Mock(), response=mock_response
                )
                return mock_response

        mock_client.post = mock_post

        results = await captioner.caption_dataset(
            images_dir=sample_images_dir,
            captions_dir=captions_dir,
            trigger_token="ohwx",
        )

        # Should have one successful caption despite one failure
        assert len(results) == 1
        assert "001.png" in results
        assert results["001.png"] == "ohwx Success caption"

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_client):
        """Test async context manager usage."""
        async with VisionCaptioner() as captioner:
            assert captioner is not None

        # Client should be closed
        mock_client.aclose.assert_called_once()
