"""ABOUTME: Code Interpreter MCP Server - Sandboxed Python code execution.

Provides sandboxed Python code execution via Docker containers.
Enforces security constraints: network disabled, resource limits, timeout.
Supports matplotlib figure capture and serialization.
"""

import asyncio
import ast
import base64
import docker
import io
import json
import logging
import os
import sys
from typing import Dict, Any, List, Optional
from mcp.server.fastmcp import Context
from mcp.types import TextContent, ImageContent, CallToolResult

from common.mcp_base import MCPServerBase
from common.error_handling import (
    ERROR_INVALID_INPUT,
    ERROR_TIMEOUT,
    create_error_result,
)

# ============================================================================
# CONSTANTS
# ============================================================================

# Docker configuration
DOCKER_IMAGE_NAME = "code-executor:latest"
DOCKER_USER = "executor"
DOCKER_MEMORY_LIMIT = "1g"
DOCKER_CPU_PERIOD = 100000
DOCKER_CPU_QUOTA = 100000
DOCKER_NETWORK_DISABLED = True
DOCKER_REMOVE_CONTAINER = True

# Code validation
MAX_CODE_LENGTH_BYTES = 51200  # 50KB (per MCP best practices)
MIN_CODE_LENGTH_BYTES = 1

# Execution configuration
EXECUTION_TIMEOUT_SECONDS = 30.0

# Figure capture
FIGURE_DPI = 100
FIGURE_FORMAT = "png"
FIGURE_MIME_TYPE = "image/png"
EXECUTION_RESULT_MARKER = "__EXECUTION_RESULT__"

# Error codes (tool-specific, not in shared module)
ERROR_CODE_CONTAINER_ERROR = "container_error"
ERROR_CODE_EXECUTION_FAILED = "execution_failed"
ERROR_CODE_DOCKER_NOT_AVAILABLE = "docker_not_available"
ERROR_CODE_IMAGE_NOT_FOUND = "image_not_found"
ERROR_CODE_API_ERROR = "api_error"

# ============================================================================
# INITIALIZATION
# ============================================================================

# Initialize MCP server with base class
server = MCPServerBase("code-interpreter")
mcp = server.get_mcp()
logger = server.get_logger()

# Initialize Docker client
_docker_client: Optional[docker.DockerClient] = None
try:
    _docker_client = docker.from_env()
    logger.info("Docker client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Docker client: {e}")
    _docker_client = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _validate_python_code(code: str) -> tuple[bool, Optional[str]]:
    """Validate Python code syntax without executing it.

    Args:
        code: Python code string to validate

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Parse error: {str(e)}"


def _should_wrap_matplotlib(code: str) -> bool:
    """Check if code actually uses matplotlib or seaborn (not just imports it).

    Args:
        code: User's Python code

    Returns:
        True if matplotlib or seaborn is actually used in the code
    """
    # Check for actual matplotlib and seaborn usage patterns, not just imports
    # Look for plt./sns. or pyplot/seaborn usage, which indicates real visualization code
    visualization_usage_patterns = [
        'plt.',              # matplotlib.pyplot alias
        'pyplot.',           # Direct pyplot module usage
        'matplotlib.pyplot', # Full module path
        'sns.',              # seaborn alias
        'seaborn.',          # seaborn module
        'plt.show()',        # Explicit show call (even though we don't need it)
        'plt.plot(',         # matplotlib plotting functions
        'plt.scatter(',
        'plt.bar(',
        'plt.hist(',
        'plt.imshow(',
        'sns.histplot(',     # seaborn plotting functions
        'sns.scatterplot(',
        'sns.lineplot(',
        'sns.barplot(',
        'sns.boxplot(',
        'sns.violinplot(',
        'sns.heatmap(',
        'sns.pairplot(',
    ]

    return any(pattern in code for pattern in visualization_usage_patterns)


def _wrap_code_for_figure_capture(code: str) -> str:
    """Wrap user code to capture matplotlib/seaborn figures if visualization code is used.

    Only adds wrapper if code actually uses matplotlib or seaborn.
    Always returns code that uses our figure capture mechanism.

    Args:
        code: User's Python code

    Returns:
        Wrapped code with figure capture (if matplotlib/seaborn is used)
    """
    if not _should_wrap_matplotlib(code):
        # Code doesn't use matplotlib, execute as-is
        return code

    # Code uses matplotlib, wrap it with capture code
    # Set up matplotlib with Agg backend and capture code at the end
    wrapper_start = f'''import sys
import io
import base64
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

_CAPTURED_FIGURES = []

def _capture_figures():
    """Capture all matplotlib figures as base64 PNG strings."""
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='{FIGURE_FORMAT}', dpi={FIGURE_DPI}, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            _CAPTURED_FIGURES.append(img_base64)
        except Exception as e:
            print(f"Error capturing figure: {{e}}", file=sys.stderr)
    plt.close('all')
    return _CAPTURED_FIGURES

# User code starts here
'''

    wrapper_end = f'''
# Figure capture happens here
_figures = _capture_figures()
if _figures:
    print("{EXECUTION_RESULT_MARKER}" + json.dumps({{"images": _figures}}))
'''

    return wrapper_start + code + wrapper_end


# ============================================================================
# MCP TOOLS
# ============================================================================

@mcp.tool()
async def execute_python(code: str, ctx: Context = None) -> CallToolResult:
    """Execute Python code in a sandboxed Docker container.

    **Security Features**:
    - Runs as non-root user
    - Network disabled
    - Resource limits: 1GB RAM, 1 CPU
    - Container removed after execution
    - Max code size: 50KB

    **Available Libraries**:
    numpy, pandas, matplotlib, seaborn, requests, scipy, sympy

    **Visualization Support**:
    - Matplotlib and seaborn figures are AUTOMATICALLY CAPTURED and DISPLAYED
    - Any figures you create via plt.plot(), sns.histplot(), etc. will be automatically
      shown to the user - you don't need to do anything special
    - Do NOT try to manually capture or encode figures (the system handles this)

    Args:
        code: Python code to execute (max 50KB)

    Returns:
        CallToolResult with TextContent/ImageContent and metadata:
        - execution_time_ms: Time spent executing in milliseconds
        - code_length: Length of submitted code in bytes
        - output_length: Length of stdout output in bytes
        - image_count: Number of matplotlib figures captured
        - wrapped: Boolean indicating if matplotlib wrapper was applied

    Examples:
        execute_python("print('Hello, world!')")
        execute_python("import numpy as np; print(np.array([1,2,3]).sum())")
        execute_python("import matplotlib.pyplot as plt\\nplt.plot([1,2,3])\\nplt.title('My Plot')")
        execute_python("import seaborn as sns\\ndata = [1,2,2,3,3,3]\\nsns.histplot(data)")
    """
    # ============================================================================
    # INPUT VALIDATION
    # ============================================================================

    # Validate: Docker availability
    if _docker_client is None:
        logger.error("Docker client not available")
        if ctx:
            await ctx.error("Docker client not available")
        return create_error_result(
            error_message="Docker client not available. Is Docker running?",
            error_code=ERROR_CODE_DOCKER_NOT_AVAILABLE,
            error_type="docker_error"
        )

    # Validate: Code is not empty
    if not code or len(code.strip()) < MIN_CODE_LENGTH_BYTES:
        logger.warning("Empty or whitespace-only code submitted")
        if ctx:
            await ctx.error("Code cannot be empty")
        return create_error_result(
            error_message="Code cannot be empty or whitespace-only",
            error_code=ERROR_INVALID_INPUT,
            error_type="validation_error"
        )

    # Validate: Code length limit (50KB per MCP best practices)
    code_bytes = len(code.encode('utf-8'))
    if code_bytes > MAX_CODE_LENGTH_BYTES:
        logger.warning(f"Code too large: {code_bytes} bytes (max: {MAX_CODE_LENGTH_BYTES})")
        if ctx:
            await ctx.error(f"Code exceeds {MAX_CODE_LENGTH_BYTES} byte limit")
        return create_error_result(
            error_message=f"Code exceeds maximum length of {MAX_CODE_LENGTH_BYTES} bytes ({code_bytes} bytes submitted)",
            error_code=ERROR_INVALID_INPUT,
            error_type="validation_error",
            additional_metadata={
                "code_length": code_bytes,
                "max_length": MAX_CODE_LENGTH_BYTES,
            }
        )

    # Validate: Python syntax
    is_valid, syntax_error = _validate_python_code(code)
    if not is_valid:
        logger.warning(f"Invalid Python syntax: {syntax_error}")
        if ctx:
            await ctx.error(f"Invalid syntax: {syntax_error}")
        return create_error_result(
            error_message=f"Invalid Python syntax: {syntax_error}",
            error_code=ERROR_INVALID_INPUT,
            error_type="validation_error",
            additional_metadata={"syntax_error": syntax_error}
        )

    # ============================================================================
    # EXECUTION
    # ============================================================================

    try:
        import time
        start_time = time.time()

        logger.info(f"Executing Python code (length: {code_bytes} bytes)")
        if ctx:
            await ctx.report_progress(1, 4, "Validating and preparing code...")

        # Wrap code to capture matplotlib figures (only if matplotlib is used)
        wrapped_code = _wrap_code_for_figure_capture(code)
        is_wrapped = wrapped_code != code

        logger.debug(f"Code wrapped for matplotlib: {is_wrapped}")
        if ctx:
            await ctx.report_progress(2, 4, f"Executing {code_bytes} bytes of Python code...")

        # Execute in Docker container
        result = await asyncio.to_thread(
            _docker_client.containers.run,
            DOCKER_IMAGE_NAME,
            command=["python3", "-c", wrapped_code],
            remove=DOCKER_REMOVE_CONTAINER,
            mem_limit=DOCKER_MEMORY_LIMIT,
            cpu_period=DOCKER_CPU_PERIOD,
            cpu_quota=DOCKER_CPU_QUOTA,
            network_disabled=DOCKER_NETWORK_DISABLED,
            stdout=True,
            stderr=True,
            user=DOCKER_USER
        )

        execution_time_ms = int((time.time() - start_time) * 1000)
        output = result.decode('utf-8')

        logger.info(f"Code execution successful (output: {len(output)} chars, time: {execution_time_ms}ms)")
        if ctx:
            await ctx.report_progress(3, 4, f"Parsing output ({len(output)} chars)...")

        # Parse output for execution result and matplotlib figures
        stdout = output
        images: List[str] = []

        if EXECUTION_RESULT_MARKER in output:
            # Split output into stdout and execution result
            parts = output.split(EXECUTION_RESULT_MARKER, 1)
            stdout = parts[0].strip()

            # Parse execution result JSON (contains images)
            try:
                result_json = json.loads(parts[1])
                images = result_json.get("images", [])
                logger.debug(f"Captured {len(images)} matplotlib figures")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse execution result JSON: {e}")

        if ctx:
            await ctx.report_progress(4, 4, f"Complete: {len(images)} figures")

        # Build MCP content array
        content: List[Any] = []

        # Add stdout if present
        if stdout:
            content.append(TextContent(
                type="text",
                text=stdout
            ))

        # Add images as ImageContent blocks
        for img_base64 in images:
            content.append(ImageContent(
                type="image",
                data=img_base64,
                mimeType=FIGURE_MIME_TYPE
            ))

        # Return empty text if no output
        if not content:
            content = [TextContent(type="text", text="")]

        return CallToolResult(
            content=content,
            isError=False,
            metadata={
                "execution_time_ms": execution_time_ms,
                "code_length": code_bytes,
                "output_length": len(stdout),
                "image_count": len(images),
                "wrapped": is_wrapped,
            }
        )

    except docker.errors.ContainerError as e:
        # Container exited with non-zero code (execution error)
        execution_time_ms = int((time.time() - start_time) * 1000)
        stderr = e.stderr.decode('utf-8') if e.stderr else str(e)
        stdout = e.stdout.decode('utf-8') if e.stdout else ""

        logger.warning(f"Container execution error (exit code {e.exit_status}): {stderr[:200]}")
        if ctx:
            await ctx.error(f"Execution error: {stderr[:100]}")

        # Build error message with stdout/stderr for content
        error_details = []
        if stdout:
            error_details.append(stdout)
        if stderr:
            error_details.append(f"Error:\n{stderr}")

        error_message = "\n".join(error_details) if error_details else "Execution error (no output)"

        return create_error_result(
            error_message=error_message,
            error_code=ERROR_CODE_CONTAINER_ERROR,
            error_type="execution_error",
            additional_metadata={
                "exit_status": e.exit_status,
                "execution_time_ms": execution_time_ms,
            }
        )

    except docker.errors.ImageNotFound:
        logger.error(f"Docker image '{DOCKER_IMAGE_NAME}' not found")
        error_msg = f"Docker image '{DOCKER_IMAGE_NAME}' not found. Please build it first: docker build -t {DOCKER_IMAGE_NAME} ./code-executor"
        if ctx:
            await ctx.error("Docker image not found")

        return create_error_result(
            error_message=error_msg,
            error_code=ERROR_CODE_IMAGE_NOT_FOUND,
            error_type="docker_error",
            additional_metadata={"image_name": DOCKER_IMAGE_NAME}
        )

    except docker.errors.APIError as e:
        logger.error(f"Docker API error: {e}")
        error_msg = f"Docker API Error: {str(e)}"
        if ctx:
            await ctx.error(error_msg)

        return create_error_result(
            error_message=error_msg,
            error_code=ERROR_CODE_API_ERROR,
            error_type="docker_error",
            additional_metadata={"api_error": str(e)}
        )

    except Exception as e:
        logger.error(f"Unexpected error during code execution: {e}", exc_info=True)
        error_msg = f"Unexpected error: {str(e)}"
        if ctx:
            await ctx.error(error_msg)

        return create_error_result(
            error_message=error_msg,
            error_code=ERROR_CODE_EXECUTION_FAILED,
            error_type="execution_error",
            additional_metadata={"exception": str(e)}
        )


# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Code Interpreter MCP server (Streamable HTTP)...")
    logger.info(f"Docker image: {DOCKER_IMAGE_NAME}")
    logger.info(f"Max code length: {MAX_CODE_LENGTH_BYTES} bytes")
    logger.info(f"Memory limit: {DOCKER_MEMORY_LIMIT}")
    server.run(transport="streamable-http")
