"""ABOUTME: ComfyUI workflow client for queuing, progress tracking, and output collection.

Provides functionality to:
- Queue workflows with custom parameters
- Track progress via WebSocket or polling
- Upload input images and masks
- Collect generated output images
"""

import asyncio
import logging
import os
import uuid
from typing import AsyncIterator, Dict, List, Optional, Tuple, Any

import httpx


logger = logging.getLogger(__name__)


class ComfyUIClient:
    """Client for interacting with ComfyUI API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 300.0,
    ):
        """Initialize ComfyUI client.

        Args:
            base_url: Base URL for ComfyUI (e.g., http://127.0.0.1:8188)
            timeout: Request timeout in seconds
        """
        self.base_url = (
            base_url or os.getenv("COMFY_URL", "http://127.0.0.1:8188")
        ).rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

        logger.info(f"ComfyUI client initialized: {self.base_url}")

    async def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """Queue a workflow for execution.

        Args:
            workflow: ComfyUI workflow/prompt dictionary

        Returns:
            Prompt ID for tracking

        Raises:
            ValueError: If queueing fails
            httpx.HTTPError: On network errors
        """
        logger.info("Queuing workflow to ComfyUI")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Clean up workflow: remove comment keys (start with #) that ComfyUI doesn't understand
                clean_workflow = {
                    k: v for k, v in workflow.items() if not k.startswith("#")
                }

                payload = {
                    "prompt": clean_workflow,
                    "client_id": self.client_id,
                }
                response = await client.post(
                    f"{self.base_url}/prompt",
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()
                prompt_id = data.get("prompt_id")

                if not prompt_id:
                    raise ValueError(f"No prompt_id in response: {data}")

                logger.info(f"Workflow queued successfully: {prompt_id}")
                return prompt_id

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error queueing workflow: {e.response.status_code} - {e.response.text}"
            )
            raise ValueError(
                f"Failed to queue workflow: {e.response.status_code} - {e.response.text}"
            ) from e
        except httpx.HTTPError as e:
            logger.error(f"Network error queueing workflow: {e}")
            raise ValueError("Failed to queue workflow: network error") from e

    async def progress(
        self, prompt_id: str, poll_interval: float = 1.0
    ) -> AsyncIterator[int]:
        """Track workflow progress.

        Yields progress values from 0-100. Uses polling fallback.

        Args:
            prompt_id: Prompt ID to track
            poll_interval: Seconds between polls

        Yields:
            Progress percentage (0-100)
        """
        logger.info(f"Tracking progress for prompt: {prompt_id}")

        # Use polling approach (WebSocket can be added later if needed)
        last_progress = 0
        yield last_progress

        while True:
            await asyncio.sleep(poll_interval)

            try:
                history = await self._get_history(prompt_id)

                # Check if completed
                if history:
                    # Check for errors
                    if "error" in history:
                        logger.error(f"Workflow failed: {history['error']}")
                        raise ValueError(f"Workflow failed: {history['error']}")

                    # Check if outputs are ready
                    outputs = history.get("outputs", {})
                    if outputs:
                        logger.info("Workflow completed")
                        yield 100
                        return

                # Emit estimated progress markers
                if last_progress < 90:
                    # Progress: 0 -> 25 -> 50 -> 75 -> 90 (while waiting)
                    progress_steps = [25, 50, 75, 90]
                    for step in progress_steps:
                        if last_progress < step:
                            last_progress = step
                            yield last_progress
                            await asyncio.sleep(poll_interval)
                            break

            except httpx.HTTPError as e:
                logger.warning(f"Error checking progress: {e}")
                await asyncio.sleep(poll_interval)
                continue

    async def _get_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow history by prompt ID.

        Args:
            prompt_id: Prompt ID to query

        Returns:
            History data or None if not found
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/history/{prompt_id}")
                response.raise_for_status()

                data = response.json()
                return data.get(prompt_id)

        except httpx.HTTPError:
            return None

    async def collect_output_files(self, prompt_id: str) -> List[Tuple[str, bytes]]:
        """Collect output files from completed workflow.

        Args:
            prompt_id: Completed prompt ID

        Returns:
            List of (filename, bytes) tuples

        Raises:
            ValueError: If collection fails or no outputs found
        """
        logger.info(f"Collecting outputs for prompt: {prompt_id}")

        try:
            history = await self._get_history(prompt_id)

            if not history:
                raise ValueError(f"No history found for prompt: {prompt_id}")

            # Check for errors
            if "error" in history:
                error_msg = history["error"]
                raise ValueError(f"Workflow failed: {error_msg}")

            outputs = history.get("outputs", {})
            if not outputs:
                raise ValueError("No outputs found in workflow history")

            # Collect all output images
            results = []

            for node_id, node_output in outputs.items():
                images = node_output.get("images", [])

                for img_info in images:
                    filename = img_info.get("filename")
                    subfolder = img_info.get("subfolder", "")
                    file_type = img_info.get("type", "output")

                    if not filename:
                        logger.warning(f"No filename in output: {img_info}")
                        continue

                    # Download the image
                    img_bytes = await self._download_output(
                        filename, subfolder, file_type
                    )
                    results.append((filename, img_bytes))
                    logger.info(
                        f"Collected output: {filename} ({len(img_bytes)} bytes)"
                    )

            if not results:
                raise ValueError("No output images found")

            logger.info(f"Collected {len(results)} output file(s)")
            return results

        except httpx.HTTPError as e:
            logger.error(f"Network error collecting outputs: {e}")
            raise ValueError("Failed to collect outputs: network error") from e

    async def _download_output(
        self,
        filename: str,
        subfolder: str = "",
        file_type: str = "output",
    ) -> bytes:
        """Download output file from ComfyUI.

        Args:
            filename: Output filename
            subfolder: Subfolder path
            file_type: File type (output, input, temp)

        Returns:
            Raw file bytes
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {
                "filename": filename,
                "type": file_type,
            }
            if subfolder:
                params["subfolder"] = subfolder

            response = await client.get(
                f"{self.base_url}/view",
                params=params,
            )
            response.raise_for_status()
            return response.content

    async def upload_image(
        self,
        image_bytes: bytes,
        filename: str,
        kind: str = "image",
        overwrite: bool = True,
    ) -> str:
        """Upload image to ComfyUI input directory.

        Args:
            image_bytes: Raw image bytes
            filename: Filename for the upload
            kind: Upload kind ("image" or "mask")
            overwrite: Whether to overwrite existing file

        Returns:
            Uploaded filename (may differ from input if not overwrite)

        Raises:
            ValueError: If upload fails
        """
        logger.info(
            f"Uploading {kind} to ComfyUI: {filename} ({len(image_bytes)} bytes)"
        )

        endpoint = f"/upload/{kind}" if kind == "mask" else "/upload/image"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {
                    "image": (filename, image_bytes, "image/png"),
                }
                data = {
                    "type": "input",
                    "overwrite": str(overwrite).lower(),
                }

                response = await client.post(
                    f"{self.base_url}{endpoint}",
                    files=files,
                    data=data,
                )
                response.raise_for_status()

                result = response.json()
                uploaded_name = result.get("name", filename)

                logger.info(f"Image uploaded successfully: {uploaded_name}")
                return uploaded_name

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error uploading image: {e.response.status_code} - {e.response.text}"
            )
            raise ValueError(
                f"Failed to upload {kind}: {e.response.status_code}"
            ) from e
        except httpx.HTTPError as e:
            logger.error(f"Network error uploading image: {e}")
            raise ValueError(f"Failed to upload {kind}: network error") from e
