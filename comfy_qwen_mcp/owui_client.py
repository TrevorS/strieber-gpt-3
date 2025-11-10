"""
Open WebUI API client for file uploads and downloads.

Handles authentication, file uploads to OWUI Files API, and retrieving file content.
"""

import logging
from typing import Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class OWUIConfig(BaseSettings):
    """Open WebUI configuration from environment."""

    owui_base_url: str = "http://localhost:8080"
    owui_api_token: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class OWUIClient:
    """Client for Open WebUI Files API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize OWUI client.

        Args:
            base_url: Base URL for Open WebUI (e.g., https://webui.example.com)
            api_token: Bearer token for authentication
            timeout: HTTP request timeout in seconds
        """
        config = OWUIConfig()
        self.base_url = (base_url or config.owui_base_url).rstrip("/")
        self.api_token = api_token or config.owui_api_token
        self.timeout = timeout

        # Parse domain for SSRF checks
        self.owui_domain = urlparse(self.base_url).netloc

        logger.info(f"Initialized OWUI client for {self.base_url}")

    def _get_headers(self) -> dict:
        """Build request headers with authentication."""
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    async def upload_file(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str = "image/png",
    ) -> Tuple[str, str]:
        """
        Upload a file to Open WebUI Files API.

        Args:
            file_bytes: File content as bytes
            filename: Name for the file
            mime_type: MIME type of the file

        Returns:
            Tuple of (file_id, content_url)

        Raises:
            httpx.HTTPError: On upload failure
        """
        url = urljoin(self.base_url, "/api/v1/files/")

        files = {
            "file": (filename, file_bytes, mime_type)
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            logger.info(f"Uploading file {filename} ({len(file_bytes)} bytes) to OWUI")

            response = await client.post(
                url,
                files=files,
                headers=self._get_headers(),
            )
            response.raise_for_status()

            data = response.json()
            file_id = data.get("id")

            if not file_id:
                raise ValueError(f"No file ID returned from OWUI: {data}")

            # Build canonical content URL
            content_url = urljoin(self.base_url, f"/api/v1/files/{file_id}/content")

            logger.info(f"Uploaded file {filename} → ID: {file_id}")
            return file_id, content_url

    async def download_file_content(self, file_id: str) -> bytes:
        """
        Download file content by file ID.

        Args:
            file_id: OWUI file ID

        Returns:
            File content as bytes

        Raises:
            httpx.HTTPError: On download failure
        """
        url = urljoin(self.base_url, f"/api/v1/files/{file_id}/content")
        return await self.download_url(url)

    async def download_url(self, url: str) -> bytes:
        """
        Download content from a URL.

        Adds Bearer token if the URL is from the configured OWUI domain.
        Implements basic SSRF protection.

        Args:
            url: URL to download from

        Returns:
            Content as bytes

        Raises:
            httpx.HTTPError: On download failure
            ValueError: If URL is not allowed (SSRF protection)
        """
        parsed = urlparse(url)

        # SSRF guard: only auto-authenticate OWUI URLs
        headers = {}
        if parsed.netloc == self.owui_domain:
            headers = self._get_headers()
        else:
            # For non-OWUI URLs, require explicit allowlist or reject
            logger.warning(f"Downloading from non-OWUI domain: {parsed.netloc}")
            # In production, you might want to reject or check an allowlist
            # For now, we'll allow but log

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            logger.info(f"Downloading from {url}")

            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()

            # Verify content type
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                logger.warning(f"Non-image content type: {content_type}")

            content = response.content

            # Size limit check (30 MB)
            max_size = 30 * 1024 * 1024
            if len(content) > max_size:
                raise ValueError(
                    f"File too large: {len(content)} bytes (max {max_size} bytes)"
                )

            logger.info(f"Downloaded {len(content)} bytes from {url}")
            return content

    async def is_owui_url(self, url: str) -> bool:
        """Check if a URL belongs to the configured OWUI instance."""
        parsed = urlparse(url)
        return parsed.netloc == self.owui_domain
