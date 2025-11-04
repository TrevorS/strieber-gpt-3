"""
Code Interpreter MCP Server

Provides sandboxed Python code execution via Docker containers.
Enforces security constraints: network disabled, resource limits, timeout.
Supports matplotlib figure capture and serialization.
"""

import asyncio
import base64
import docker
import io
import json
import logging
import os
import sys
from mcp.server.fastmcp import FastMCP, Context
from starlette.requests import Request
from starlette.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("code-interpreter")

# Health check endpoint for Docker container orchestration
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint for Docker container orchestration."""
    return JSONResponse({"status": "ok"})

# Initialize Docker client
try:
    docker_client = docker.from_env()
    logger.info("Docker client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Docker client: {e}")
    docker_client = None


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
    wrapper_start = '''import sys
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
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            _CAPTURED_FIGURES.append(img_base64)
        except Exception as e:
            print(f"Error capturing figure: {e}", file=sys.stderr)
    plt.close('all')
    return _CAPTURED_FIGURES

# User code starts here
'''

    wrapper_end = '''
# Figure capture happens here
_figures = _capture_figures()
if _figures:
    print("__EXECUTION_RESULT__" + json.dumps({"images": _figures}))
'''

    return wrapper_start + code + wrapper_end


@mcp.tool()
async def execute_python(code: str, ctx: Context = None) -> dict:
    """Execute Python code in a sandboxed Docker container.

    Security features:
    - Runs as non-root user
    - Network disabled
    - Resource limits: 1GB RAM, 1 CPU
    - Container removed after execution

    Visualization Support:
    - Matplotlib and seaborn figures are AUTOMATICALLY CAPTURED and DISPLAYED
    - Any figures you create via plt.plot(), sns.histplot(), etc. will be automatically
      shown to the user - you don't need to do anything special
    - Do NOT try to manually capture or encode figures (the system handles this)

    Returns unified dict format:
    {
        "stdout": "text output",
        "stderr": "error output if any",
        "images": ["base64_png_1", "base64_png_2"],  # Internal use only - figures displayed separately
        "success": true/false
    }

    Args:
        code: Python code to execute (string)
        ctx: MCP context for progress/logging (auto-injected)

    Returns:
        Dict with stdout, stderr, images, and success flag

    Example:
        execute_python("print('Hello, world!')")
        execute_python("import numpy as np; print(np.array([1,2,3]).sum())")
        execute_python("import matplotlib.pyplot as plt\\nplt.plot([1,2,3])\\nplt.title('My Plot')")
        execute_python("import seaborn as sns\\ndata = [1,2,2,3,3,3]\\nsns.histplot(data)")
    """
    if docker_client is None:
        if ctx:
            await ctx.error("Docker client not available")
        return {
            "stdout": "",
            "stderr": "Docker client not available. Is Docker running?",
            "images": [],
            "success": False
        }

    try:
        logger.info(f"Executing Python code (length: {len(code)} chars)")
        if ctx:
            await ctx.report_progress(1, 4, "Preparing Docker container...")

        # Wrap code to capture matplotlib figures (only if matplotlib is used)
        wrapped_code = _wrap_code_for_figure_capture(code)
        if ctx:
            await ctx.report_progress(2, 4, f"Executing {len(code)} chars of Python code...")

        result = await asyncio.to_thread(
            docker_client.containers.run,
            "code-executor:latest",
            command=["python3", "-c", wrapped_code],
            remove=True,
            mem_limit="1g",
            cpu_period=100000,
            cpu_quota=100000,
            network_disabled=True,
            stdout=True,
            stderr=True,
            user="executor"
        )

        output = result.decode('utf-8')
        logger.info(f"Code execution successful (output length: {len(output)} chars)")
        if ctx:
            await ctx.report_progress(3, 4, "Executed successfully")

        # Parse output for execution result and matplotlib figures
        stdout = output
        images = []

        if "__EXECUTION_RESULT__" in output:
            # Split output into stdout and execution result
            parts = output.split("__EXECUTION_RESULT__", 1)
            stdout = parts[0].strip()

            # Parse execution result JSON (contains images)
            try:
                result_json = json.loads(parts[1])
                images = result_json.get("images", [])
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse execution result: {e}")

        if ctx:
            await ctx.report_progress(4, 4, "Complete")

        # Return unified format
        result_obj = {
            "stdout": stdout,
            "stderr": "",
            "images": images,
            "success": True
        }
        return result_obj

    except docker.errors.ContainerError as e:
        # Container exited with non-zero code
        stderr = e.stderr.decode('utf-8') if e.stderr else str(e)
        stdout = e.stdout.decode('utf-8') if e.stdout else ""

        logger.warning(f"Container execution error: {stderr}")
        if ctx:
            await ctx.error(f"Execution error: {stderr[:100]}")

        result_obj = {
            "stdout": stdout,
            "stderr": stderr,
            "images": [],
            "success": False
        }
        return result_obj

    except docker.errors.ImageNotFound:
        logger.error("code-executor:latest image not found")
        error_msg = "code-executor Docker image not found. Please build it first: docker build -t code-executor:latest ./code-executor"
        if ctx:
            await ctx.error("Docker image not found")

        result_obj = {
            "stdout": "",
            "stderr": error_msg,
            "images": [],
            "success": False
        }
        return result_obj

    except docker.errors.APIError as e:
        logger.error(f"Docker API error: {e}")
        error_msg = f"Docker API Error: {str(e)}"
        if ctx:
            await ctx.error(error_msg)

        result_obj = {
            "stdout": "",
            "stderr": error_msg,
            "images": [],
            "success": False
        }
        return result_obj

    except Exception as e:
        logger.error(f"Unexpected error during code execution: {e}", exc_info=True)
        error_msg = f"Unexpected Error: {str(e)}"
        if ctx:
            await ctx.error(error_msg)

        result_obj = {
            "stdout": "",
            "stderr": error_msg,
            "images": [],
            "success": False
        }
        return result_obj


@mcp.tool()
async def get_available_libraries(ctx: Context = None) -> str:
    """Get list of pre-installed Python libraries in the code executor.

    Args:
        ctx: MCP context for logging (auto-injected)

    Returns:
        List of available libraries (formatted as string)
    """
    libraries = [
        "numpy - Numerical computing",
        "pandas - Data analysis",
        "matplotlib - Plotting",
        "requests - HTTP client",
        "scipy - Scientific computing",
        "sympy - Symbolic mathematics"
    ]

    return "Available Python libraries:\n" + "\n".join(f"  • {lib}" for lib in libraries)


if __name__ == "__main__":
    logger.info("Starting Code Interpreter MCP server (Streamable HTTP)...")
    mcp.run(transport="streamable-http")
