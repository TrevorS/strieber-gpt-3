"""ABOUTME: HTTP client for llama-server ReaderLM-v2 inference.

Communicates with the dedicated llama-server instance running
ReaderLM-v2 model for HTML-to-Markdown conversion with optional
instruction-based extraction.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class LlamaReaderClient:
    """Client for communicating with llama-server-readerlm inference service."""

    def __init__(self, endpoint: str = "http://llama-server-readerlm:8000"):
        """
        Initialize llama-server client.

        Args:
            endpoint: URL to the llama-server-readerlm service (internal Docker port 8000)
        """
        self.endpoint = endpoint
        self.model = "ReaderLM-v2"
        self.client = httpx.AsyncClient(timeout=120.0)  # Long timeout for inference

    async def html_to_markdown(
        self,
        html_content: str,
        instruction: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.1
    ) -> tuple[str, bool]:
        """
        Convert HTML to Markdown using ReaderLM-v2 with optional instruction.

        Args:
            html_content: Raw HTML to convert
            instruction: Optional instruction for extraction (e.g., "Extract the price")
                        If None, uses default: "Extract the main content from the given HTML and convert it to Markdown format."
            max_tokens: Maximum tokens in output (default 8192)
            temperature: Temperature for generation (default 0.1, near-deterministic)

        Returns:
            (markdown_content, success)
        """
        try:
            # Use default instruction if none provided
            if instruction is None:
                instruction = "Extract the main content from the given HTML and convert it to Markdown format."

            # Use chat completions endpoint with proper message format for ReaderLM-v2
            response = await self.client.post(
                f"{self.endpoint}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": instruction
                        },
                        {
                            "role": "user",
                            "content": html_content
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.95
                }
            )
            response.raise_for_status()

            result = response.json()
            if result.get("choices") and len(result["choices"]) > 0:
                # Extract message content from chat completion response
                message = result["choices"][0].get("message", {})
                markdown = message.get("content", "")
                return markdown, True
            else:
                logger.error("No choices in response from llama-server")
                return "", False

        except httpx.TimeoutException:
            logger.error("Llama-server inference timeout")
            return "", False
        except httpx.HTTPStatusError as e:
            logger.error(f"Llama-server HTTP error {e.response.status_code}: {e}")
            return "", False
        except Exception as e:
            logger.error(f"Llama-server error: {e}")
            return "", False

    async def health_check(self) -> bool:
        """Check if llama-server is healthy."""
        try:
            response = await self.client.get(
                f"{self.endpoint}/health",
                timeout=5.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Llama-server health check failed: {e}")
            return False

    async def get_info(self) -> Optional[dict]:
        """Get model information from llama-server."""
        try:
            response = await self.client.get(
                f"{self.endpoint}/props",
                timeout=5.0
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get llama-server info: {e}")
        return None

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
