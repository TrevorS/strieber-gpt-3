"""
title: Image to File Router Filter
author: Claude Code
version: 0.1.0
description: Pre-model filter that converts inline images to Open WebUI File references for non-vision models
"""

import base64
import logging
import os
from typing import Optional
from urllib.parse import urlparse

import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Filter:
    """Open WebUI Filter that converts inline images to File API references.

    For non-vision models, this filter:
    1. Extracts image_url blocks from user messages
    2. Uploads images to OWUI Files API
    3. Replaces image blocks with text references to canonical URLs
    4. Adds hints for using image editing tools

    This ensures non-vision models can still trigger image editing workflows
    via MCP tools while avoiding vision-related errors.
    """

    class Valves(BaseModel):
        """Configuration values for the filter."""
        priority: int = 0
        owui_base_url: str = os.getenv("OWUI_BASE_URL", "http://localhost:3000")
        owui_api_token: str = os.getenv("OWUI_API_TOKEN", "")
        enable_for_non_vision_only: bool = True
        max_file_size_mb: int = 30

    def __init__(self):
        """Initialize the filter."""
        self.valves = self.Valves()
        logger.info("Image to File Router filter initialized")

    def _is_non_vision_model(self, __model__: dict) -> bool:
        """Check if the model supports vision/multimodal input.

        Args:
            __model__: Model metadata dictionary

        Returns:
            True if model is non-vision, False if it supports vision
        """
        # Check model info for vision capability
        info = __model__.get("info", {})

        # Check if model explicitly lists vision capability
        capabilities = info.get("capabilities", {})
        if isinstance(capabilities, dict):
            has_vision = capabilities.get("vision", False)
            if has_vision:
                return False

        # Check model ID/name patterns for known vision models
        model_id = __model__.get("id", "").lower()
        model_name = __model__.get("name", "").lower()

        vision_indicators = [
            "vision", "gpt-4-turbo", "gpt-4o", "claude-3", "gemini-pro-vision",
            "llava", "qwen-vl", "multimodal"
        ]

        for indicator in vision_indicators:
            if indicator in model_id or indicator in model_name:
                return False

        # Default: assume non-vision for safety
        return True

    def _download_image_data(self, url: str) -> Optional[bytes]:
        """Download image data from URL.

        Args:
            url: Data URL or HTTP(S) URL

        Returns:
            Image bytes or None if download fails
        """
        try:
            # Handle data URLs
            if url.startswith("data:"):
                # Extract base64 data
                if ";base64," in url:
                    _, b64_data = url.split(";base64,", 1)
                    return base64.b64decode(b64_data)
                else:
                    logger.warning("Unsupported data URL format (not base64)")
                    return None

            # Handle HTTP(S) URLs
            elif url.startswith(("http://", "https://")):
                # Enforce size limit
                max_size = self.valves.max_file_size_mb * 1024 * 1024
                response = requests.get(
                    url,
                    timeout=30,
                    stream=True,
                    headers={"User-Agent": "OpenWebUI-ImageFilter/1.0"}
                )
                response.raise_for_status()

                # Check content length
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > max_size:
                    logger.warning(f"Image too large: {content_length} bytes")
                    return None

                # Download with size checking
                chunks = []
                total_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    total_size += len(chunk)
                    if total_size > max_size:
                        logger.warning(f"Image download exceeded size limit: {total_size}")
                        return None
                    chunks.append(chunk)

                return b"".join(chunks)

            else:
                logger.warning(f"Unsupported URL scheme: {url}")
                return None

        except Exception as e:
            logger.error(f"Failed to download image from {url[:100]}: {e}")
            return None

    def _upload_to_owui(self, image_bytes: bytes, filename: str) -> Optional[dict]:
        """Upload image to Open WebUI Files API.

        Args:
            image_bytes: Raw image bytes
            filename: Filename for the upload

        Returns:
            Dict with 'id' and 'url' keys, or None if upload fails
        """
        if not self.valves.owui_api_token:
            logger.warning("OWUI API token not configured, skipping upload")
            return None

        try:
            url = f"{self.valves.owui_base_url.rstrip('/')}/api/v1/files/"

            files = {
                "file": (filename, image_bytes, "image/png")
            }

            headers = {
                "Authorization": f"Bearer {self.valves.owui_api_token}"
            }

            response = requests.post(url, files=files, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()
            file_id = data.get("id")

            if not file_id:
                logger.error(f"Upload response missing 'id': {data}")
                return None

            # Build content URL
            content_url = f"{self.valves.owui_base_url.rstrip('/')}/api/v1/files/{file_id}/content"

            logger.info(f"Uploaded image to OWUI: {file_id}")
            return {"id": file_id, "url": content_url}

        except Exception as e:
            logger.error(f"Failed to upload to OWUI: {e}")
            return None

    def inlet(self, body: dict, __model__: dict, __user__: dict) -> dict:
        """Process incoming request before it reaches the model.

        Args:
            body: Request body with messages
            __model__: Model metadata
            __user__: User metadata

        Returns:
            Modified body (or original if no changes needed)
        """
        # Check if filter should be enabled
        if self.valves.enable_for_non_vision_only:
            if not self._is_non_vision_model(__model__):
                logger.info(f"Skipping filter for vision-capable model: {__model__.get('name')}")
                return body

        messages = body.get("messages", [])
        if not messages:
            return body

        # Process the last user message
        last_message = None
        last_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                last_message = messages[idx]
                last_idx = idx
                break

        if not last_message:
            return body

        content = last_message.get("content", "")

        # Handle string content (no images)
        if isinstance(content, str):
            return body

        # Handle list content (may contain images)
        if not isinstance(content, list):
            return body

        # Extract and process image_url blocks
        new_content = []
        uploaded_urls = []
        image_count = 0

        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")

                if block_type == "image_url":
                    image_url_data = block.get("image_url", {})
                    if isinstance(image_url_data, dict):
                        url = image_url_data.get("url", "")
                    elif isinstance(image_url_data, str):
                        url = image_url_data
                    else:
                        url = ""

                    if url:
                        # Download image
                        image_bytes = self._download_image_data(url)

                        if image_bytes:
                            # Upload to OWUI
                            image_count += 1
                            filename = f"user_image_{image_count}.png"
                            result = self._upload_to_owui(image_bytes, filename)

                            if result:
                                uploaded_urls.append(result["url"])
                                logger.info(f"Converted image to OWUI file: {result['url']}")
                            else:
                                logger.warning(f"Failed to upload image {image_count}")
                        else:
                            logger.warning(f"Failed to download image {image_count}")

                    # Don't add the image_url block to new content (remove it)
                    continue

                elif block_type == "text":
                    # Keep text blocks as-is
                    new_content.append(block)
                else:
                    # Keep other block types
                    new_content.append(block)
            else:
                # Keep non-dict items
                new_content.append(block)

        # If we processed images, add text references
        if uploaded_urls:
            # Add text block with file references
            file_list = "\n".join(f"- {url}" for url in uploaded_urls)
            reference_text = f"\n\nImages uploaded to Open WebUI Files:\n{file_list}"

            # Add hint for image editing
            hint_text = (
                "\n\nNote: If you want to edit these images, use the qwen_image_edit tool "
                "with init_image_url set to one of the URLs above."
            )

            # Append to existing text block or create new one
            if new_content and isinstance(new_content[-1], dict) and new_content[-1].get("type") == "text":
                new_content[-1]["text"] = new_content[-1].get("text", "") + reference_text + hint_text
            else:
                new_content.append({
                    "type": "text",
                    "text": reference_text + hint_text
                })

            # Update message content
            messages[last_idx]["content"] = new_content

            logger.info(f"Processed {len(uploaded_urls)} images for non-vision model")

        return body


# Pydantic import (required by Open WebUI filters)
try:
    from pydantic import BaseModel
except ImportError:
    # Fallback for environments without pydantic
    class BaseModel:
        pass
