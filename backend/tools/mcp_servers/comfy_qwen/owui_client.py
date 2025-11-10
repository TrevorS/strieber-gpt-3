"""ABOUTME: Open WebUI Files API client for uploading and downloading images.

Provides functionality to:
- Upload generated images to Open WebUI Files storage
- Download images by file ID
- Construct canonical URLs for file access
- Handle authentication with Bearer tokens
"""

import logging
import os
from typing import Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx


logger = logging.getLogger(__name__)


class OpenWebUIClient:
    """Client for interacting with Open WebUI Files API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize Open WebUI client.

        Args:
            base_url: Base URL for Open WebUI (e.g., https://webui.example.com)
            api_token: Bearer token for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = (base_url or os.getenv("OWUI_BASE_URL", "")).rstrip("/")
        self.api_token = api_token or os.getenv("OWUI_API_TOKEN", "")
        self.timeout = timeout

        if not self.base_url:
            logger.warning("OWUI_BASE_URL not set - file uploads will be disabled")

        if not self.api_token:
            logger.warning("OWUI_API_TOKEN not set - authentication may fail")

        self.files_endpoint = f"{self.base_url}/api/v1/files"

    def _get_headers(self) -> dict:
        """Get HTTP headers including authentication.

        Returns:
            Dictionary of headers with Bearer token
        """
        return {
            "Authorization": f"Bearer {self.api_token}",
        }

    def _is_owui_url(self, url: str) -> bool:
        """Check if URL belongs to Open WebUI domain.

        Args:
            url: URL to check

        Returns:
            True if URL is from OWUI domain
        """
        if not self.base_url:
            return False
        parsed = urlparse(url)
        base_parsed = urlparse(self.base_url)
        return parsed.netloc == base_parsed.netloc

    async def upload_file(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str = "image/png",
    ) -> Tuple[str, str]:
        """Upload file to Open WebUI Files storage.

        Args:
            file_bytes: Raw file bytes
            filename: Name for the file
            mime_type: MIME type (default: image/png)

        Returns:
            Tuple of (file_id, content_url)

        Raises:
            ValueError: If upload fails or response is invalid
            httpx.HTTPError: On network errors
        """
        if not self.base_url or not self.api_token:
            raise ValueError(
                "Open WebUI base URL and API token must be configured. "
                "Set OWUI_BASE_URL and OWUI_API_TOKEN environment variables."
            )

        logger.info(f"Uploading file to OWUI: {filename} ({len(file_bytes)} bytes)")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                files = {
                    "file": (filename, file_bytes, mime_type),
                }
                response = await client.post(
                    self.files_endpoint,
                    files=files,
                    headers=self._get_headers(),
                )
                response.raise_for_status()

                data = response.json()
                file_id = data.get("id")

                if not file_id:
                    raise ValueError(f"No file ID in upload response: {data}")

                # Construct canonical content URL
                content_url = f"{self.base_url}/api/v1/files/{file_id}/content"

                logger.info(f"File uploaded successfully: {file_id}")
                return file_id, content_url

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error uploading file: {e.response.status_code} - {e.response.text}")
            raise ValueError(
                f"Failed to upload file to Open WebUI: {e.response.status_code} - {e.response.text}"
            ) from e
        except httpx.HTTPError as e:
            logger.error(f"Network error uploading file: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading file: {e}")
            raise ValueError(f"Failed to upload file: {e}") from e

    async def download_file_content(self, file_id: str) -> bytes:
        """Download file content by file ID.

        Args:
            file_id: Open WebUI file ID

        Returns:
            Raw file bytes

        Raises:
            ValueError: If download fails
            httpx.HTTPError: On network errors
        """
        if not self.base_url or not self.api_token:
            raise ValueError(
                "Open WebUI base URL and API token must be configured. "
                "Set OWUI_BASE_URL and OWUI_API_TOKEN environment variables."
            )

        content_url = f"{self.base_url}/api/v1/files/{file_id}/content"
        logger.info(f"Downloading file from OWUI: {file_id}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    content_url,
                    headers=self._get_headers(),
                )
                response.raise_for_status()

                logger.info(f"File downloaded successfully: {file_id} ({len(response.content)} bytes)")
                return response.content

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading file: {e.response.status_code}")
            raise ValueError(
                f"Failed to download file from Open WebUI: {e.response.status_code}"
            ) from e
        except httpx.HTTPError as e:
            logger.error(f"Network error downloading file: {e}")
            raise

    async def download_url(self, url: str) -> bytes:
        """Download content from URL.

        Adds Bearer authentication if URL is from OWUI domain.

        Args:
            url: URL to download from

        Returns:
            Raw file bytes

        Raises:
            ValueError: If download fails or URL is not allowed
            httpx.HTTPError: On network errors
        """
        logger.info(f"Downloading from URL: {url}")

        # SSRF guard: only allow OWUI URLs
        if not self._is_owui_url(url):
            raise ValueError(
                f"URL not from Open WebUI domain: {url}. "
                f"Only URLs from {self.base_url} are allowed for security reasons."
            )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = self._get_headers() if self._is_owui_url(url) else {}
                response = await client.get(url, headers=headers)
                response.raise_for_status()

                # Verify it's an image
                content_type = response.headers.get("content-type", "")
                if not content_type.startswith("image/"):
                    raise ValueError(f"URL does not point to an image: {content_type}")

                # Size limit: 30 MB
                max_size = 30 * 1024 * 1024
                if len(response.content) > max_size:
                    raise ValueError(f"File too large: {len(response.content)} bytes (max {max_size})")

                logger.info(f"URL downloaded successfully ({len(response.content)} bytes)")
                return response.content

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading URL: {e.response.status_code}")
            raise ValueError(f"Failed to download URL: {e.response.status_code}") from e
        except httpx.HTTPError as e:
            logger.error(f"Network error downloading URL: {e}")
            raise
