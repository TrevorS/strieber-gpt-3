"""ABOUTME: HTTP client for llama-server ReaderLM-v2 inference.

Communicates with the dedicated llama-server instance running
ReaderLM-v2 model for HTML-to-Markdown conversion with optional
instruction-based extraction.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Context window limits
MAX_CONTEXT_TOKENS = 131072  # llama-server-readerlm context size
SYSTEM_PROMPT_TOKENS = 500   # Approximate tokens for system prompt
OUTPUT_TOKENS = 8000         # Budget for output generation
SAFE_INPUT_TOKENS = MAX_CONTEXT_TOKENS - SYSTEM_PROMPT_TOKENS - OUTPUT_TOKENS
BYTES_PER_TOKEN = 3.5        # Conservative estimate: 3.5 bytes = 1 token


def estimate_tokens(text: str) -> int:
    """Estimate token count from text (conservative estimate).

    Args:
        text: Text to estimate tokens for

    Returns:
        Approximate token count
    """
    return max(1, len(text.encode('utf-8')) // int(BYTES_PER_TOKEN))


def truncate_html(html: str, max_bytes: int = 300000) -> tuple[str, bool]:
    """Truncate HTML if it exceeds safe size for ReaderLM-v2.

    Strategy: Keep first N% of HTML (main content usually at top)

    Args:
        html: HTML content to potentially truncate
        max_bytes: Maximum HTML size in bytes before truncation (default 300KB for safety)

    Returns:
        (truncated_html, was_truncated) tuple
    """
    html_bytes = len(html.encode('utf-8'))

    if html_bytes <= max_bytes:
        return html, False

    # Calculate safe truncation point
    # Aim for ~90% to ensure we stay under limit with safety margin
    target_bytes = int(max_bytes * 0.85)

    # Find safe truncation point (avoid breaking in middle of tag)
    html_truncated = html[:target_bytes]

    # Try to truncate at a tag boundary
    last_close_tag = html_truncated.rfind('>')
    if last_close_tag > target_bytes * 0.75:  # Must be reasonably close to target
        html_truncated = html_truncated[:last_close_tag + 1]

    html_truncated += "\n<!-- [Content truncated due to size] -->\n"

    truncated_tokens = estimate_tokens(html_truncated)
    original_tokens = estimate_tokens(html)

    logger.warning(
        f"HTML truncated: {html_bytes} bytes ({original_tokens} tokens) "
        f"→ {len(html_truncated.encode('utf-8'))} bytes ({truncated_tokens} tokens)"
    )

    return html_truncated, True


class LlamaReaderClient:
    """Client for communicating with llama-server-readerlm inference service."""

    def __init__(self, endpoint: str = "http://llama-server-readerlm:8000", default_timeout: float = 180.0):
        """
        Initialize llama-server client.

        Args:
            endpoint: URL to the llama-server-readerlm service (internal Docker port 8000)
            default_timeout: Default timeout for inference requests in seconds (default 180s)
        """
        self.endpoint = endpoint
        self.model = "ReaderLM-v2"
        self.default_timeout = default_timeout
        self.client = httpx.AsyncClient(timeout=default_timeout)

    async def html_to_markdown(
        self,
        html_content: str,
        instruction: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.1,
        timeout: Optional[float] = None
    ) -> tuple[str, bool]:
        """
        Convert HTML to Markdown using ReaderLM-v2 with optimal default extraction.

        Args:
            html_content: Raw HTML to convert
            instruction: Ignored - always uses default extraction for optimal quality
                        ReaderLM-v2 performs 24.6% better than GPT-4o with default extraction
            max_tokens: Maximum tokens in output (default 8192)
            temperature: Temperature for generation (default 0.1, near-deterministic)
            timeout: Optional timeout override in seconds (uses default_timeout if None)

        Returns:
            (markdown_content, success)
        """
        try:
            # Validate HTML size before sending to ReaderLM
            html_truncated, was_truncated = truncate_html(html_content)
            tokens_in = estimate_tokens(html_truncated)

            if was_truncated or tokens_in > SAFE_INPUT_TOKENS:
                logger.warning(
                    f"HTML input size: {tokens_in} tokens "
                    f"(safe limit: {SAFE_INPUT_TOKENS} tokens, "
                    f"max context: {MAX_CONTEXT_TOKENS} tokens)"
                )

            # Always use default instruction - research shows custom instructions reduce quality
            # ReaderLM-v2 is optimized for main content extraction
            instruction = "Extract the main content from the given HTML and convert it to Markdown format."

            # Use the provided timeout or fall back to default
            request_timeout = timeout if timeout is not None else self.default_timeout

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
                            "content": html_truncated
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.95
                },
                timeout=request_timeout
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
