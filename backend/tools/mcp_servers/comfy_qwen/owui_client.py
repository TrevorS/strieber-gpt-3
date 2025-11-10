"""ABOUTME: Open WebUI client for file upload/download operations.

Handles interaction with Open WebUI Files API for storing and retrieving generated images.
Implements SSRF protection and proper authentication.
"""

import logging
from typing import Optional, Tuple
from urllib.parse import urljoin, urlparse
import httpx

logger = logging.getLogger(__name__)

# Size limit for file downloads (30 MB)
MAX_FILE_SIZE = 30 * 1024 * 1024


class OpenWebUIClient:
    """Client for Open WebUI Files API.

    Provides methods to upload generated images to OWUI and download
    image files by ID or URL for use in ComfyUI workflows.
    """

    def __init__(
        self,
        base_url: str,
        api_token: str,
        timeout: float = 60.0,
        max_file_size: int = MAX_FILE_SIZE
    ):
        """Initialize Open WebUI client.

        Args:
            base_url: Base URL of Open WebUI instance (e.g., "https://webui.example.com")
            api_token: Bearer token for API authentication
            timeout: HTTP request timeout in seconds
            max_file_size: Maximum file size to download in bytes
        """
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.timeout = timeout
        self.max_file_size = max_file_size

        # Parse domain for SSRF protection
        parsed = urlparse(self.base_url)
        self.owui_domain = parsed.netloc

        logger.info(f"OpenWebUI client initialized for {self.base_url}")

    def _get_headers(self) -> dict:
        """Get HTTP headers with authentication.

        Returns:
            Dictionary of HTTP headers including Bearer token
        """
        return {
            "Authorization": f"Bearer {self.api_token}",
        }

    async def upload_file(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str = "image/png"
    ) -> Tuple[str, str]:
        """Upload a file to Open WebUI Files API.

        Args:
            file_bytes: Raw bytes of the file to upload
            filename: Name for the uploaded file
            mime_type: MIME type of the file

        Returns:
            Tuple of (file_id, content_url)

        Raises:
            httpx.HTTPStatusError: If upload fails
            ValueError: If file is too large or invalid
        """
        if len(file_bytes) > self.max_file_size:
            raise ValueError(
                f"File size {len(file_bytes)} exceeds limit {self.max_file_size}"
            )

        url = urljoin(self.base_url, "/api/v1/files/")

        files = {
            "file": (filename, file_bytes, mime_type)
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            logger.info(f"Uploading file to OWUI: {filename} ({len(file_bytes)} bytes)")
            response = await client.post(
                url,
                headers=self._get_headers(),
                files=files
            )
            response.raise_for_status()

            data = response.json()
            file_id = data.get("id")

            if not file_id:
                raise ValueError(f"Upload response missing 'id' field: {data}")

            # Build content URL
            content_url = f"{self.base_url}/api/v1/files/{file_id}/content"

            logger.info(f"File uploaded successfully: {file_id}")
            return file_id, content_url

    async def download_file_content(self, file_id: str) -> bytes:
        """Download file content by file ID.

        Args:
            file_id: The OWUI file ID to download

        Returns:
            Raw bytes of the file

        Raises:
            httpx.HTTPStatusError: If download fails
            ValueError: If file is too large or not an image
        """
        url = f"{self.base_url}/api/v1/files/{file_id}/content"
        return await self.download_url(url)

    async def download_url(self, url: str) -> bytes:
        """Download content from a URL with SSRF protection.

        Only allows downloads from the configured OWUI domain for security.

        Args:
            url: URL to download from

        Returns:
            Raw bytes of the downloaded content

        Raises:
            ValueError: If URL is not from OWUI domain or file is invalid
            httpx.HTTPStatusError: If download fails
        """
        parsed = urlparse(url)

        # SSRF protection: only allow OWUI domain
        if parsed.netloc and parsed.netloc != self.owui_domain:
            raise ValueError(
                f"SSRF protection: URL domain {parsed.netloc} does not match "
                f"OWUI domain {self.owui_domain}. Only OWUI-hosted files are allowed."
            )

        # Build full URL if relative
        if not parsed.scheme:
            url = urljoin(self.base_url, url)

        headers = self._get_headers()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            logger.info(f"Downloading from URL: {url}")

            # Stream download to check size before reading all content
            async with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()

                # Verify content type
                content_type = response.headers.get("content-type", "")
                if not content_type.startswith("image/"):
                    raise ValueError(
                        f"Content-Type must be image/*, got: {content_type}"
                    )

                # Check content length if available
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > self.max_file_size:
                    raise ValueError(
                        f"File size {content_length} exceeds limit {self.max_file_size}"
                    )

                # Read content with size checking
                chunks = []
                total_size = 0
                async for chunk in response.aiter_bytes():
                    total_size += len(chunk)
                    if total_size > self.max_file_size:
                        raise ValueError(
                            f"File size exceeds limit {self.max_file_size}"
                        )
                    chunks.append(chunk)

                content = b"".join(chunks)
                logger.info(f"Downloaded {len(content)} bytes from {url}")
                return content


async def create_owui_client(
    base_url: Optional[str] = None,
    api_token: Optional[str] = None
) -> OpenWebUIClient:
    """Create OpenWebUI client from environment or parameters.

    Args:
        base_url: OWUI base URL (defaults to OWUI_BASE_URL env var)
        api_token: API token (defaults to OWUI_API_TOKEN env var)

    Returns:
        Configured OpenWebUIClient instance

    Raises:
        ValueError: If required configuration is missing
    """
    import os

    base_url = base_url or os.getenv("OWUI_BASE_URL")
    api_token = api_token or os.getenv("OWUI_API_TOKEN")

    if not base_url:
        raise ValueError(
            "OWUI base URL not configured. Set OWUI_BASE_URL environment variable "
            "or pass base_url parameter."
        )

    if not api_token:
        raise ValueError(
            "OWUI API token not configured. Set OWUI_API_TOKEN environment variable "
            "or pass api_token parameter."
        )

    return OpenWebUIClient(base_url, api_token)
