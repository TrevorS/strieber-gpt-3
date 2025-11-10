"""
title: Image to File Router Filter
author: Anthropic (for ComfyUI Qwen MCP)
version: 1.0.0
license: MIT
description: Pre-model filter that converts inline images to OWUI File URLs for non-vision models
required_open_webui_version: 0.6.0
"""

import asyncio
import base64
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field


class Filter:
    """Open WebUI filter that converts inline images to File URLs.

    When a non-vision model receives images in the chat, this filter:
    1. Detects image_url content blocks in user messages
    2. Downloads/decodes the images
    3. Uploads them to OWUI Files API
    4. Removes inline images from the message
    5. Appends file URLs as text

    This enables non-vision models to still work with tools like qwen_image_edit
    that accept file URLs.
    """

    class Valves(BaseModel):
        """Configuration for the filter."""

        priority: int = Field(
            default=0,
            description="Filter priority (lower runs first)",
        )
        owui_base_url: str = Field(
            default="",
            description="Open WebUI base URL (leave empty to use WEBUI_URL env var)",
        )
        owui_api_token: str = Field(
            default="",
            description="Open WebUI API token for file uploads (leave empty to use request token)",
        )
        enabled_for_non_vision_only: bool = Field(
            default=True,
            description="Only activate for non-vision models",
        )
        add_tool_hint: bool = Field(
            default=True,
            description="Add a hint about using qwen_image_edit tool",
        )
        max_image_size_mb: int = Field(
            default=30,
            description="Maximum image size in MB",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.logger = logging.getLogger(__name__)

    def _is_vision_model(self, __model__: Dict[str, Any]) -> bool:
        """Check if the model supports vision.

        Args:
            __model__: Model metadata dictionary

        Returns:
            True if model supports vision/images
        """
        # Check common vision model indicators
        model_id = __model__.get("id", "").lower()
        model_name = __model__.get("name", "").lower()

        vision_keywords = ["vision", "vl", "llava", "gpt-4-vision", "claude-3", "gemini"]

        for keyword in vision_keywords:
            if keyword in model_id or keyword in model_name:
                return True

        # Check capabilities
        capabilities = __model__.get("capabilities", {})
        if capabilities.get("vision") or capabilities.get("image_input"):
            return True

        return False

    def _get_base_url(self) -> str:
        """Get OWUI base URL from valves or environment.

        Returns:
            Base URL string
        """
        if self.valves.owui_base_url:
            return self.valves.owui_base_url.rstrip("/")

        # Try common environment variables
        for env_var in ["WEBUI_URL", "OWUI_BASE_URL", "OPEN_WEBUI_URL"]:
            url = os.getenv(env_var)
            if url:
                return url.rstrip("/")

        # Default to localhost
        return "http://localhost:3000"

    def _get_api_token(self, __user__: Dict[str, Any]) -> str:
        """Get API token from valves or user metadata.

        Args:
            __user__: User metadata dictionary

        Returns:
            API token string
        """
        if self.valves.owui_api_token:
            return self.valves.owui_api_token

        # Extract from user metadata (Open WebUI passes this)
        return __user__.get("token", "")

    async def _upload_file(
        self,
        file_bytes: bytes,
        filename: str,
        base_url: str,
        api_token: str,
    ) -> Optional[Dict[str, str]]:
        """Upload file to OWUI Files API.

        Args:
            file_bytes: Raw file bytes
            filename: Filename for upload
            base_url: OWUI base URL
            api_token: Bearer token

        Returns:
            Dict with 'file_id' and 'url' keys, or None on failure
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {
                    "file": (filename, file_bytes, "image/png"),
                }
                headers = {
                    "Authorization": f"Bearer {api_token}",
                }

                response = await client.post(
                    f"{base_url}/api/v1/files",
                    files=files,
                    headers=headers,
                )
                response.raise_for_status()

                data = response.json()
                file_id = data.get("id")

                if not file_id:
                    self.logger.error(f"No file ID in upload response: {data}")
                    return None

                content_url = f"{base_url}/api/v1/files/{file_id}/content"

                self.logger.info(f"Uploaded file: {file_id}")
                return {
                    "file_id": file_id,
                    "url": content_url,
                }

        except Exception as e:
            self.logger.error(f"Failed to upload file: {e}")
            return None

    async def _process_image_url(
        self,
        image_url: str,
        base_url: str,
        api_token: str,
    ) -> Optional[Dict[str, str]]:
        """Process an image URL: download and upload to OWUI.

        Args:
            image_url: Data URL or HTTP(S) URL
            base_url: OWUI base URL
            api_token: Bearer token

        Returns:
            Dict with 'file_id' and 'url', or None on failure
        """
        try:
            # Handle data URLs
            if image_url.startswith("data:"):
                # Extract base64 data
                if ";base64," in image_url:
                    _, b64_data = image_url.split(";base64,", 1)
                    image_bytes = base64.b64decode(b64_data)
                else:
                    self.logger.error("Unsupported data URL format")
                    return None

            # Handle HTTP(S) URLs
            elif image_url.startswith("http://") or image_url.startswith("https://"):
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(image_url)
                    response.raise_for_status()

                    # Check size
                    max_size = self.valves.max_image_size_mb * 1024 * 1024
                    if len(response.content) > max_size:
                        self.logger.error(f"Image too large: {len(response.content)} bytes")
                        return None

                    # Verify content type
                    content_type = response.headers.get("content-type", "")
                    if not content_type.startswith("image/"):
                        self.logger.error(f"Not an image: {content_type}")
                        return None

                    image_bytes = response.content

            else:
                self.logger.error(f"Unsupported URL scheme: {image_url[:50]}")
                return None

            # Upload to OWUI
            return await self._upload_file(
                image_bytes,
                "chat_image.png",
                base_url,
                api_token,
            )

        except Exception as e:
            self.logger.error(f"Failed to process image URL: {e}")
            return None

    async def inlet(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __model__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process incoming chat messages before model.

        Args:
            body: Request body with messages
            __user__: User metadata
            __model__: Model metadata

        Returns:
            Modified body
        """
        self.logger.info("Image to File Router filter activated")

        # Check if we should process
        if self.valves.enabled_for_non_vision_only:
            if __model__ and self._is_vision_model(__model__):
                self.logger.info("Vision model detected, skipping filter")
                return body

        # Get configuration
        base_url = self._get_base_url()
        api_token = self._get_api_token(__user__ or {})

        if not api_token:
            self.logger.warning("No API token available, skipping filter")
            return body

        # Process messages
        messages = body.get("messages", [])

        for message in messages:
            # Only process user messages
            if message.get("role") != "user":
                continue

            content = message.get("content")

            # Handle list content (multi-modal format)
            if isinstance(content, list):
                uploaded_urls = []
                new_content = []

                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")

                        # Process image_url blocks
                        if block_type == "image_url":
                            image_url_data = block.get("image_url", {})
                            url = image_url_data.get("url", "")

                            if url:
                                self.logger.info(f"Processing image: {url[:50]}...")

                                # Download and upload
                                result = asyncio.run(
                                    self._process_image_url(url, base_url, api_token)
                                )

                                if result:
                                    uploaded_urls.append(result["url"])
                                else:
                                    # Keep original if upload failed
                                    new_content.append(block)

                        # Keep other content types
                        else:
                            new_content.append(block)

                    else:
                        # Keep non-dict content
                        new_content.append(block)

                # Add uploaded file URLs as text
                if uploaded_urls:
                    self.logger.info(f"Uploaded {len(uploaded_urls)} image(s)")

                    urls_text = "Images uploaded:\n" + "\n".join(uploaded_urls)

                    if self.valves.add_tool_hint:
                        urls_text += (
                            "\n\nNote: If you need to edit these images, "
                            "use the qwen_image_edit tool with init_image_url "
                            "set to one of the URLs above."
                        )

                    new_content.append({
                        "type": "text",
                        "text": urls_text,
                    })

                # Update message content
                if new_content != content:
                    message["content"] = new_content

        self.logger.info("Filter processing complete")
        return body
