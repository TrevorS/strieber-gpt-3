"""ABOUTME: ComfyUI client for workflow queuing, progress tracking, and output retrieval.

Handles all interactions with ComfyUI API including:
- Uploading images for img2img/inpaint workflows
- Queueing workflow execution
- Tracking progress via WebSocket or polling
- Retrieving generated output images
"""

import asyncio
import json
import logging
import uuid
from typing import AsyncIterator, Dict, List, Optional, Tuple, Any
from io import BytesIO

import httpx
import websockets

logger = logging.getLogger(__name__)


class ComfyUIClient:
    """Client for ComfyUI API operations.

    Provides methods to queue workflows, track progress, and retrieve outputs.
    Supports both WebSocket and polling-based progress tracking.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        timeout: float = 300.0,
        poll_interval: float = 2.0
    ):
        """Initialize ComfyUI client.

        Args:
            base_url: Base URL of ComfyUI instance
            timeout: HTTP request timeout in seconds
            poll_interval: Polling interval for progress updates in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval

        # Build WebSocket URL from HTTP URL
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.ws_url = f"{ws_url}/ws"

        logger.info(f"ComfyUI client initialized for {self.base_url}")

    async def upload_image(
        self,
        image_bytes: bytes,
        filename: str,
        image_type: str = "input",
        subfolder: str = ""
    ) -> str:
        """Upload an image to ComfyUI.

        Args:
            image_bytes: Raw bytes of the image
            filename: Name for the uploaded image
            image_type: Type of image ("input" or "temp")
            subfolder: Optional subfolder for organization

        Returns:
            Name of the uploaded image as stored by ComfyUI

        Raises:
            httpx.HTTPStatusError: If upload fails
        """
        url = f"{self.base_url}/upload/image"

        files = {
            "image": (filename, BytesIO(image_bytes), "image/png")
        }

        data = {
            "type": image_type,
            "subfolder": subfolder
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            logger.info(f"Uploading image to ComfyUI: {filename} ({len(image_bytes)} bytes)")
            response = await client.post(url, files=files, data=data)
            response.raise_for_status()

            result = response.json()
            uploaded_name = result.get("name", filename)
            logger.info(f"Image uploaded successfully: {uploaded_name}")
            return uploaded_name

    async def upload_mask(
        self,
        mask_bytes: bytes,
        filename: str,
        original_ref: Optional[Dict[str, str]] = None
    ) -> str:
        """Upload a mask image to ComfyUI.

        Args:
            mask_bytes: Raw bytes of the mask image
            filename: Name for the uploaded mask
            original_ref: Optional reference to original image

        Returns:
            Name of the uploaded mask as stored by ComfyUI

        Raises:
            httpx.HTTPStatusError: If upload fails
        """
        url = f"{self.base_url}/upload/mask"

        files = {
            "image": (filename, BytesIO(mask_bytes), "image/png")
        }

        data = {"type": "input"}
        if original_ref:
            data["original_ref"] = json.dumps(original_ref)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            logger.info(f"Uploading mask to ComfyUI: {filename}")
            response = await client.post(url, files=files, data=data)
            response.raise_for_status()

            result = response.json()
            uploaded_name = result.get("name", filename)
            logger.info(f"Mask uploaded successfully: {uploaded_name}")
            return uploaded_name

    async def queue_prompt(
        self,
        workflow: Dict[str, Any],
        client_id: Optional[str] = None
    ) -> str:
        """Queue a workflow for execution.

        Args:
            workflow: ComfyUI workflow JSON (API format)
            client_id: Optional client ID for WebSocket tracking

        Returns:
            Prompt ID for tracking execution

        Raises:
            httpx.HTTPStatusError: If queueing fails
        """
        if client_id is None:
            client_id = str(uuid.uuid4())

        url = f"{self.base_url}/prompt"

        payload = {
            "prompt": workflow,
            "client_id": client_id
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            logger.info(f"Queueing workflow with client_id: {client_id}")
            response = await client.post(url, json=payload)
            response.raise_for_status()

            result = response.json()
            prompt_id = result.get("prompt_id")

            if not prompt_id:
                raise ValueError(f"Queue response missing 'prompt_id': {result}")

            logger.info(f"Workflow queued successfully: {prompt_id}")
            return prompt_id

    async def get_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get execution history for a prompt.

        Args:
            prompt_id: The prompt ID to query

        Returns:
            History data if available, None if not found

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        url = f"{self.base_url}/history/{prompt_id}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            response.raise_for_status()

            data = response.json()
            return data.get(prompt_id)

    async def track_progress_ws(
        self,
        prompt_id: str,
        client_id: str
    ) -> AsyncIterator[Tuple[int, str]]:
        """Track workflow progress via WebSocket.

        Args:
            prompt_id: The prompt ID to track
            client_id: Client ID used when queueing

        Yields:
            Tuples of (progress_percent, message)

        Note:
            Falls back to polling if WebSocket connection fails
        """
        try:
            async with websockets.connect(
                f"{self.ws_url}?clientId={client_id}",
                ping_interval=20,
                ping_timeout=10
            ) as ws:
                logger.info(f"WebSocket connected for prompt {prompt_id}")

                # Track execution progress
                current_node = 0
                total_nodes = 0
                completed = False

                while not completed:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)

                        msg_type = data.get("type")

                        if msg_type == "execution_start":
                            yield (5, "Execution started")

                        elif msg_type == "executing":
                            node = data.get("data", {}).get("node")
                            if node is None:
                                # Execution finished
                                completed = True
                                yield (100, "Execution completed")
                            else:
                                current_node += 1
                                # Estimate progress (reserve last 10% for output retrieval)
                                progress = min(90, int((current_node / max(total_nodes, 1)) * 90))
                                yield (progress, f"Processing node {node}")

                        elif msg_type == "progress":
                            value = data.get("data", {}).get("value", 0)
                            max_val = data.get("data", {}).get("max", 100)
                            progress = int((value / max_val) * 90)
                            yield (progress, f"Progress: {value}/{max_val}")

                        elif msg_type == "execution_error":
                            error_msg = data.get("data", {}).get("exception_message", "Unknown error")
                            raise RuntimeError(f"ComfyUI execution error: {error_msg}")

                    except asyncio.TimeoutError:
                        # Check if prompt is still in queue
                        history = await self.get_history(prompt_id)
                        if history:
                            completed = True
                            yield (100, "Execution completed")
                        else:
                            yield (current_node * 10, "Processing...")

        except (websockets.WebSocketException, OSError) as e:
            logger.warning(f"WebSocket connection failed: {e}, falling back to polling")
            async for progress, msg in self.track_progress_poll(prompt_id):
                yield progress, msg

    async def track_progress_poll(self, prompt_id: str) -> AsyncIterator[Tuple[int, str]]:
        """Track workflow progress via polling.

        Args:
            prompt_id: The prompt ID to track

        Yields:
            Tuples of (progress_percent, message)
        """
        logger.info(f"Polling for progress on prompt {prompt_id}")

        milestones = [(25, "Processing..."), (50, "Generating..."), (75, "Finalizing...")]
        milestone_idx = 0

        while True:
            history = await self.get_history(prompt_id)

            if history:
                status = history.get("status", {})
                if status.get("status_str") == "success":
                    yield (100, "Execution completed")
                    break
                elif status.get("status_str") == "error":
                    error_msg = status.get("messages", ["Unknown error"])[0]
                    raise RuntimeError(f"ComfyUI execution error: {error_msg}")

            # Emit milestone progress
            if milestone_idx < len(milestones):
                progress, msg = milestones[milestone_idx]
                yield (progress, msg)
                milestone_idx += 1

            await asyncio.sleep(self.poll_interval)

    async def get_output_images(self, prompt_id: str) -> List[Tuple[str, bytes]]:
        """Retrieve output images from a completed workflow.

        Args:
            prompt_id: The prompt ID to retrieve outputs for

        Returns:
            List of tuples (filename, image_bytes)

        Raises:
            RuntimeError: If execution failed or outputs not found
        """
        history = await self.get_history(prompt_id)

        if not history:
            raise RuntimeError(f"No history found for prompt {prompt_id}")

        status = history.get("status", {})
        if status.get("status_str") == "error":
            error_msg = status.get("messages", ["Unknown error"])[0]
            raise RuntimeError(f"ComfyUI execution failed: {error_msg}")

        # Extract output images from history
        outputs = history.get("outputs", {})
        images = []

        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    filename = img_info.get("filename")
                    subfolder = img_info.get("subfolder", "")
                    img_type = img_info.get("type", "output")

                    if filename:
                        # Download image
                        image_bytes = await self.download_output_image(
                            filename, subfolder, img_type
                        )
                        images.append((filename, image_bytes))

        if not images:
            raise RuntimeError(f"No output images found for prompt {prompt_id}")

        logger.info(f"Retrieved {len(images)} output images")
        return images

    async def download_output_image(
        self,
        filename: str,
        subfolder: str = "",
        image_type: str = "output"
    ) -> bytes:
        """Download an output image from ComfyUI.

        Args:
            filename: Name of the image file
            subfolder: Subfolder where image is stored
            image_type: Type of image ("output", "input", "temp")

        Returns:
            Raw bytes of the image

        Raises:
            httpx.HTTPStatusError: If download fails
        """
        url = f"{self.base_url}/view"
        params = {
            "filename": filename,
            "type": image_type
        }
        if subfolder:
            params["subfolder"] = subfolder

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            logger.info(f"Downloading output image: {filename}")
            response = await client.get(url, params=params)
            response.raise_for_status()

            image_bytes = response.content
            logger.info(f"Downloaded {len(image_bytes)} bytes")
            return image_bytes


async def create_comfy_client(base_url: Optional[str] = None) -> ComfyUIClient:
    """Create ComfyUI client from environment or parameters.

    Args:
        base_url: ComfyUI base URL (defaults to COMFY_URL env var or http://127.0.0.1:8188)

    Returns:
        Configured ComfyUIClient instance
    """
    import os

    base_url = base_url or os.getenv("COMFY_URL", "http://127.0.0.1:8188")
    return ComfyUIClient(base_url)
