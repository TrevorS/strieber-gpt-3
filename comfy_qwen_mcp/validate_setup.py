#!/usr/bin/env python3
"""
Validation script to check that all dependencies and services are accessible.

Run this before starting the MCP server to ensure everything is configured correctly.
"""

import asyncio
import sys
from pathlib import Path

import httpx
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Load configuration from environment."""

    comfy_url: str = "http://127.0.0.1:8188"
    owui_base_url: str = "http://localhost:8080"
    owui_api_token: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


async def check_comfyui(url: str) -> bool:
    """Check if ComfyUI is accessible."""
    print(f"\n[1/4] Checking ComfyUI at {url}...", end=" ")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            if response.status_code == 200:
                print("✓ OK")
                return True
            else:
                print(f"✗ FAILED (HTTP {response.status_code})")
                return False
    except Exception as e:
        print(f"✗ FAILED ({type(e).__name__}: {e})")
        return False


async def check_owui(url: str, token: str) -> bool:
    """Check if Open WebUI is accessible."""
    print(f"[2/4] Checking Open WebUI at {url}...", end=" ")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            if response.status_code == 200:
                print("✓ OK")

                # Check API authentication if token provided
                if token:
                    print("     Checking API authentication...", end=" ")
                    headers = {"Authorization": f"Bearer {token}"}
                    api_response = await client.get(
                        f"{url}/api/v1/files/",
                        headers=headers
                    )
                    if api_response.status_code in (200, 401, 403):
                        # 401/403 means endpoint exists but auth may be wrong
                        if api_response.status_code == 200:
                            print("✓ Authenticated")
                        else:
                            print(f"⚠ Auth issue (HTTP {api_response.status_code})")
                            print(f"     Check OWUI_API_TOKEN in .env")
                    else:
                        print(f"✗ Unexpected response (HTTP {api_response.status_code})")
                else:
                    print("     ⚠ No API token configured (set OWUI_API_TOKEN in .env)")

                return True
            else:
                print(f"✗ FAILED (HTTP {response.status_code})")
                return False
    except Exception as e:
        print(f"✗ FAILED ({type(e).__name__}: {e})")
        return False


def check_workflows() -> bool:
    """Check if workflow files exist."""
    print("[3/4] Checking workflow files...", end=" ")

    workflows_dir = Path(__file__).parent / "workflows"
    required_files = ["qwen_image_api.json", "qwen_edit_api.json"]

    missing = []
    for filename in required_files:
        if not (workflows_dir / filename).exists():
            missing.append(filename)

    if missing:
        print(f"✗ MISSING: {', '.join(missing)}")
        return False
    else:
        print("✓ OK")
        return True


def check_dependencies() -> bool:
    """Check if required Python packages are installed."""
    print("[4/4] Checking Python dependencies...", end=" ")

    required = [
        "fastmcp",
        "httpx",
        "pydantic",
        "pillow",
        "uvicorn",
        "websockets",
    ]

    missing = []
    for package in required:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)

    if missing:
        print(f"✗ MISSING: {', '.join(missing)}")
        print(f"\n   Install with: pip install {' '.join(missing)}")
        return False
    else:
        print("✓ OK")
        return True


async def main():
    """Run all validation checks."""
    print("=" * 60)
    print("ComfyUI Qwen MCP Server - Setup Validation")
    print("=" * 60)

    # Load config
    try:
        config = Config()
    except Exception as e:
        print(f"\n✗ Failed to load configuration: {e}")
        print("\nCreate a .env file based on .env.example")
        return False

    # Run checks
    checks = []

    checks.append(await check_comfyui(config.comfy_url))
    checks.append(await check_owui(config.owui_base_url, config.owui_api_token))
    checks.append(check_workflows())
    checks.append(check_dependencies())

    # Summary
    print("\n" + "=" * 60)
    passed = sum(checks)
    total = len(checks)

    if passed == total:
        print(f"✓ All checks passed ({passed}/{total})")
        print("\nYou can now start the server with:")
        print("  python server.py")
        print("  # or")
        print("  uvicorn server:mcp.get_asgi_app --host 0.0.0.0 --port 8000")
        return True
    else:
        print(f"✗ Some checks failed ({passed}/{total} passed)")
        print("\nPlease resolve the issues above before starting the server.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
