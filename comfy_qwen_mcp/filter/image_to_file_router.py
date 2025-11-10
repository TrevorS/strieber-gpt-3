"""
Open WebUI pre-model filter: Image to File Router

Converts inline images in messages to Open WebUI file uploads.
This allows non-vision models to still work with image-based tool calls.

Installation:
1. In Open WebUI, go to Admin Panel → Functions
2. Create a new function
3. Paste this entire file
4. Set as "Inlet Filter" (pre-model)
5. Configure OWUI_API_TOKEN in the function settings

How it works:
- Scans user messages for image_url blocks
- Uploads images to OWUI Files API
- Removes inline images from message
- Appends text with canonical file URLs
- Non-vision models see only text, but tools can access the URLs
"""

import base64
import json
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests

# For Open WebUI filter compatibility
try:
    from pydantic import BaseModel
except ImportError:
    # Fallback for environments without pydantic
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


class Filter:
    """Open WebUI pre-model filter for image routing."""

    class Valves(BaseModel):
        """Configuration for the filter."""

        priority: int = 0
        owui_base_url: str = "http://localhost:8080"
        owui_api_token: str = ""
        convert_for_nonvision_models: bool = True
        max_image_size_mb: int = 30

    def __init__(self):
        """Initialize the filter with configuration."""
        self.valves = self.Valves()

    def inlet(self, body: Dict[str, Any], __user__: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process incoming request before it reaches the model.

        Args:
            body: Request body with messages
            __user__: User context (optional)

        Returns:
            Modified request body
        """
        # Check if we should process this request
        if not self.valves.convert_for_nonvision_models:
            return body

        # Check if model is non-vision (heuristic)
        model_id = body.get("model", "")
        is_vision_model = self._is_vision_model(model_id)

        if is_vision_model:
            # Vision models can handle images directly
            print(f"[ImageToFileRouter] Model {model_id} supports vision, skipping conversion")
            return body

        print(f"[ImageToFileRouter] Non-vision model {model_id}, converting images to files")

        # Process messages
        messages = body.get("messages", [])
        modified = False

        for message in messages:
            if message.get("role") != "user":
                continue

            content = message.get("content")

            # Handle structured content (list of blocks)
            if isinstance(content, list):
                modified_content, has_images = self._process_content_blocks(content)
                if has_images:
                    message["content"] = modified_content
                    modified = True

            # Handle string content with potential data URLs
            elif isinstance(content, str):
                modified_content, has_images = self._process_text_content(content)
                if has_images:
                    message["content"] = modified_content
                    modified = True

        if modified:
            print("[ImageToFileRouter] Successfully converted images to file URLs")

        return body

    def _is_vision_model(self, model_id: str) -> bool:
        """
        Heuristic to detect if a model supports vision.

        Args:
            model_id: Model identifier

        Returns:
            True if model likely supports vision
        """
        # Common vision model patterns
        vision_patterns = [
            "vision",
            "gpt-4-turbo",
            "gpt-4o",
            "claude-3",
            "gemini-pro-vision",
            "gemini-1.5",
            "llava",
            "bakllava",
        ]

        model_lower = model_id.lower()
        return any(pattern in model_lower for pattern in vision_patterns)

    def _process_content_blocks(
        self, content_blocks: List[Dict]
    ) -> tuple[List[Dict], bool]:
        """
        Process structured content blocks and convert images.

        Args:
            content_blocks: List of content blocks (text, image_url, etc.)

        Returns:
            Tuple of (modified_blocks, has_images)
        """
        uploaded_urls = []
        new_blocks = []
        has_images = False

        for block in content_blocks:
            if block.get("type") == "image_url":
                # Extract image URL/data
                image_url_obj = block.get("image_url", {})
                image_url = image_url_obj.get("url", "")

                if image_url:
                    has_images = True
                    # Upload and get canonical URL
                    file_url = self._upload_image_to_owui(image_url)
                    if file_url:
                        uploaded_urls.append(file_url)
                    # Don't add this block to new_blocks (remove from message)

            else:
                # Keep non-image blocks
                new_blocks.append(block)

        # Add summary text block if we uploaded any images
        if uploaded_urls:
            summary_text = "\n\n[Images uploaded to Open WebUI Files]\n"
            for idx, url in enumerate(uploaded_urls, 1):
                summary_text += f"{idx}. {url}\n"

            summary_text += (
                "\nTo edit these images, use the qwen_image_edit tool "
                "with init_image_url set to the desired image URL."
            )

            new_blocks.append({"type": "text", "text": summary_text})

        return new_blocks, has_images

    def _process_text_content(self, content: str) -> tuple[str, bool]:
        """
        Process text content and extract data URLs.

        Args:
            content: Text content potentially containing data URLs

        Returns:
            Tuple of (modified_content, has_images)
        """
        # Pattern to match data URLs
        data_url_pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+'

        data_urls = re.findall(data_url_pattern, content)

        if not data_urls:
            return content, False

        uploaded_urls = []

        for data_url in data_urls:
            file_url = self._upload_image_to_owui(data_url)
            if file_url:
                uploaded_urls.append(file_url)

        # Remove data URLs from content
        modified_content = re.sub(data_url_pattern, '[image removed, see below]', content)

        # Add uploaded URLs summary
        if uploaded_urls:
            modified_content += "\n\n[Images uploaded to Open WebUI Files]\n"
            for idx, url in enumerate(uploaded_urls, 1):
                modified_content += f"{idx}. {url}\n"

            modified_content += (
                "\nTo edit these images, use the qwen_image_edit tool "
                "with init_image_url set to the desired image URL."
            )

        return modified_content, True

    def _upload_image_to_owui(self, image_source: str) -> Optional[str]:
        """
        Upload an image to Open WebUI Files API.

        Args:
            image_source: Data URL or HTTP(S) URL

        Returns:
            Canonical file content URL or None on failure
        """
        try:
            # Get image bytes
            if image_source.startswith("data:"):
                # Data URL
                image_bytes = self._decode_data_url(image_source)
            elif image_source.startswith(("http://", "https://")):
                # HTTP URL
                image_bytes = self._fetch_url(image_source)
            else:
                print(f"[ImageToFileRouter] Unsupported image source: {image_source[:50]}")
                return None

            if not image_bytes:
                return None

            # Check size limit
            max_bytes = self.valves.max_image_size_mb * 1024 * 1024
            if len(image_bytes) > max_bytes:
                print(
                    f"[ImageToFileRouter] Image too large: {len(image_bytes)} bytes "
                    f"(max {max_bytes})"
                )
                return None

            # Upload to OWUI Files API
            file_id = self._upload_to_files_api(image_bytes)

            if file_id:
                # Build canonical content URL
                content_url = urljoin(
                    self.valves.owui_base_url,
                    f"/api/v1/files/{file_id}/content"
                )
                print(f"[ImageToFileRouter] Uploaded → {content_url}")
                return content_url

        except Exception as e:
            print(f"[ImageToFileRouter] Upload failed: {e}")

        return None

    def _decode_data_url(self, data_url: str) -> Optional[bytes]:
        """Decode a data URL to bytes."""
        try:
            # Extract base64 part
            if ";base64," in data_url:
                b64_data = data_url.split(";base64,", 1)[1]
                return base64.b64decode(b64_data)
        except Exception as e:
            print(f"[ImageToFileRouter] Data URL decode failed: {e}")

        return None

    def _fetch_url(self, url: str) -> Optional[bytes]:
        """Fetch content from HTTP(S) URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Verify it's an image
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                print(f"[ImageToFileRouter] Non-image content type: {content_type}")
                return None

            return response.content

        except Exception as e:
            print(f"[ImageToFileRouter] URL fetch failed: {e}")

        return None

    def _upload_to_files_api(self, file_bytes: bytes) -> Optional[str]:
        """
        Upload file to Open WebUI Files API.

        Args:
            file_bytes: File content

        Returns:
            File ID or None on failure
        """
        url = urljoin(self.valves.owui_base_url, "/api/v1/files/")

        headers = {}
        if self.valves.owui_api_token:
            headers["Authorization"] = f"Bearer {self.valves.owui_api_token}"

        files = {
            "file": ("image.png", file_bytes, "image/png")
        }

        try:
            response = requests.post(url, files=files, headers=headers, timeout=60)
            response.raise_for_status()

            data = response.json()
            file_id = data.get("id")

            if file_id:
                print(f"[ImageToFileRouter] Uploaded file ID: {file_id}")
                return file_id

        except Exception as e:
            print(f"[ImageToFileRouter] Files API upload failed: {e}")

        return None


# For testing
if __name__ == "__main__":
    # Example usage
    filter_instance = Filter()

    # Example request body
    test_body = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                        },
                    },
                ],
            }
        ],
    }

    result = filter_instance.inlet(test_body)
    print(json.dumps(result, indent=2))
