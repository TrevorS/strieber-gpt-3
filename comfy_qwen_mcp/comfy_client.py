"""
ComfyUI API client for queuing workflows and retrieving results.

Handles prompt queuing, progress monitoring (WebSocket + polling fallback),
image uploads, and output file collection.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
import websockets
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class ComfyConfig(BaseSettings):
    """ComfyUI configuration from environment."""

    comfy_url: str = "http://127.0.0.1:8188"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class ComfyUIClient:
    """Client for ComfyUI API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 300.0,
    ):
        """
        Initialize ComfyUI client.

        Args:
            base_url: Base URL for ComfyUI (e.g., http://127.0.0.1:8188)
            timeout: HTTP request timeout in seconds
        """
        config = ComfyConfig()
        self.base_url = (base_url or config.comfy_url).rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

        logger.info(f"Initialized ComfyUI client for {self.base_url}")

    async def upload_image(
        self,
        image_bytes: bytes,
        filename: str,
        kind: str = "input",
    ) -> str:
        """
        Upload an image to ComfyUI.

        Args:
            image_bytes: Image content
            filename: Name for the uploaded file
            kind: Upload type - "input" for init images, "mask" for masks

        Returns:
            Uploaded filename (may be modified by ComfyUI)

        Raises:
            httpx.HTTPError: On upload failure
        """
        url = f"{self.base_url}/upload/image"

        files = {
            "image": (filename, image_bytes, "image/png")
        }
        data = {
            "type": kind,
            "overwrite": "true",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            logger.info(f"Uploading {kind} image {filename} ({len(image_bytes)} bytes)")

            response = await client.post(url, files=files, data=data)
            response.raise_for_status()

            result = response.json()
            uploaded_name = result.get("name", filename)

            logger.info(f"Uploaded image → {uploaded_name}")
            return uploaded_name

    async def queue_prompt(
        self,
        workflow: Dict[str, Any],
        client_id: Optional[str] = None,
    ) -> str:
        """
        Queue a workflow for execution.

        Args:
            workflow: ComfyUI workflow JSON (API format)
            client_id: Optional client ID for WebSocket progress tracking

        Returns:
            Prompt ID for tracking execution

        Raises:
            httpx.HTTPError: On queue failure
        """
        url = f"{self.base_url}/prompt"

        payload = {
            "prompt": workflow,
            "client_id": client_id or self.client_id,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            logger.info("Queueing workflow for execution")

            response = await client.post(url, json=payload)
            response.raise_for_status()

            result = response.json()
            prompt_id = result.get("prompt_id")

            if not prompt_id:
                raise ValueError(f"No prompt_id returned from ComfyUI: {result}")

            logger.info(f"Queued workflow → prompt_id: {prompt_id}")
            return prompt_id

    async def get_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """
        Get execution history for a prompt.

        Args:
            prompt_id: Prompt ID to query

        Returns:
            History data or None if not found
        """
        url = f"{self.base_url}/history/{prompt_id}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            response.raise_for_status()

            data = response.json()
            return data.get(prompt_id)

    async def progress_ws(
        self,
        prompt_id: str,
        client_id: Optional[str] = None,
    ) -> AsyncIterator[int]:
        """
        Monitor execution progress via WebSocket.

        Yields progress percentages (0-100) as the workflow executes.

        Args:
            prompt_id: Prompt ID to monitor
            client_id: Client ID used when queuing

        Yields:
            Progress percentage (0-100)
        """
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws?clientId={client_id or self.client_id}"

        try:
            async with websockets.connect(ws_url) as websocket:
                logger.info(f"Connected to ComfyUI WebSocket for prompt {prompt_id}")

                total_steps = None
                current_step = 0

                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)

                        msg_type = data.get("type")

                        # Track execution progress
                        if msg_type == "executing":
                            node = data.get("data", {}).get("node")
                            if node is None:
                                # Execution complete
                                yield 100
                                logger.info("Execution complete (WebSocket)")
                                return

                        elif msg_type == "progress":
                            progress_data = data.get("data", {})
                            current_step = progress_data.get("value", 0)
                            total_steps = progress_data.get("max", 100)

                            if total_steps > 0:
                                percent = int((current_step / total_steps) * 100)
                                yield min(percent, 99)

                        elif msg_type == "execution_error":
                            error = data.get("data", {})
                            logger.error(f"ComfyUI execution error: {error}")
                            raise RuntimeError(f"ComfyUI execution failed: {error}")

                    except asyncio.TimeoutError:
                        # Check if execution completed via history
                        history = await self.get_history(prompt_id)
                        if history and "outputs" in history:
                            yield 100
                            return

        except Exception as e:
            logger.warning(f"WebSocket progress failed: {e}, falling back to polling")
            # Fall back to polling
            async for progress in self.progress_poll(prompt_id):
                yield progress

    async def progress_poll(
        self,
        prompt_id: str,
        poll_interval: float = 0.5,
    ) -> AsyncIterator[int]:
        """
        Monitor execution progress via polling (fallback from WebSocket).

        Yields approximate progress percentages based on completion status.

        Args:
            prompt_id: Prompt ID to monitor
            poll_interval: Seconds between polls

        Yields:
            Progress percentage (0-100)
        """
        logger.info(f"Polling progress for prompt {prompt_id}")

        # Emit approximate progress markers
        markers = [25, 50, 75, 90]
        marker_idx = 0

        while True:
            await asyncio.sleep(poll_interval)

            history = await self.get_history(prompt_id)

            if history:
                if "outputs" in history:
                    # Execution complete
                    yield 100
                    logger.info("Execution complete (polling)")
                    return

                elif marker_idx < len(markers):
                    # Emit next marker
                    yield markers[marker_idx]
                    marker_idx += 1

    async def progress(
        self,
        prompt_id: str,
        use_websocket: bool = True,
    ) -> AsyncIterator[int]:
        """
        Monitor execution progress (WebSocket with polling fallback).

        Args:
            prompt_id: Prompt ID to monitor
            use_websocket: Try WebSocket first if True

        Yields:
            Progress percentage (0-100)
        """
        if use_websocket:
            async for p in self.progress_ws(prompt_id):
                yield p
        else:
            async for p in self.progress_poll(prompt_id):
                yield p

    async def wait_for_completion(
        self,
        prompt_id: str,
        max_wait: float = 600.0,
    ) -> Dict[str, Any]:
        """
        Wait for workflow execution to complete.

        Args:
            prompt_id: Prompt ID to wait for
            max_wait: Maximum time to wait in seconds

        Returns:
            Final history data

        Raises:
            TimeoutError: If execution doesn't complete in time
            RuntimeError: If execution fails
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            history = await self.get_history(prompt_id)

            if history and "outputs" in history:
                logger.info(f"Workflow {prompt_id} completed successfully")
                return history

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_wait:
                raise TimeoutError(
                    f"Workflow execution timed out after {max_wait}s"
                )

            await asyncio.sleep(0.5)

    async def collect_output_files(
        self,
        prompt_id: str,
    ) -> List[Tuple[str, bytes]]:
        """
        Collect output files from a completed workflow.

        Args:
            prompt_id: Prompt ID of completed workflow

        Returns:
            List of (filename, file_bytes) tuples

        Raises:
            ValueError: If workflow not complete or no outputs found
        """
        history = await self.get_history(prompt_id)

        if not history:
            raise ValueError(f"No history found for prompt {prompt_id}")

        outputs = history.get("outputs", {})
        if not outputs:
            raise ValueError(f"No outputs found for prompt {prompt_id}")

        files = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for node_id, node_output in outputs.items():
                images = node_output.get("images", [])

                for img_info in images:
                    filename = img_info.get("filename")
                    subfolder = img_info.get("subfolder", "")
                    file_type = img_info.get("type", "output")

                    if not filename:
                        continue

                    # Build download URL
                    params = {
                        "filename": filename,
                        "type": file_type,
                    }
                    if subfolder:
                        params["subfolder"] = subfolder

                    url = f"{self.base_url}/view"

                    logger.info(f"Downloading output: {filename}")

                    response = await client.get(url, params=params)
                    response.raise_for_status()

                    files.append((filename, response.content))

        logger.info(f"Collected {len(files)} output files")
        return files

    async def execute_workflow(
        self,
        workflow: Dict[str, Any],
        progress_callback: Optional[callable] = None,
    ) -> List[Tuple[str, bytes]]:
        """
        Execute a workflow and return output files.

        Convenience method that queues, monitors, and collects results.

        Args:
            workflow: ComfyUI workflow JSON
            progress_callback: Optional callback for progress updates (0-100)

        Returns:
            List of (filename, file_bytes) tuples
        """
        # Queue workflow
        prompt_id = await self.queue_prompt(workflow)

        # Monitor progress
        async for progress_pct in self.progress(prompt_id):
            if progress_callback:
                await progress_callback(progress_pct)

        # Collect outputs
        return await self.collect_output_files(prompt_id)
