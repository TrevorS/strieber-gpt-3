"""Example usage of ComfyUI Qwen MCP Server.

This script demonstrates how to test the MCP server tools directly
without going through Open WebUI.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from comfy_qwen.comfy_client import ComfyUIClient
from comfy_qwen.owui_client import OpenWebUIClient


async def test_comfy_connection():
    """Test ComfyUI connection."""
    print("Testing ComfyUI connection...")

    client = ComfyUIClient()

    try:
        # Try to access the root endpoint
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as http:
            response = await http.get(f"{client.base_url}/")
            print(f"✓ ComfyUI is accessible at {client.base_url}")
            print(f"  Status: {response.status_code}")
            return True
    except Exception as e:
        print(f"✗ ComfyUI connection failed: {e}")
        print(f"  URL: {client.base_url}")
        print(f"  Make sure ComfyUI is running!")
        return False


async def test_owui_connection():
    """Test Open WebUI connection."""
    print("\nTesting Open WebUI connection...")

    client = OpenWebUIClient()

    if not client.base_url:
        print("✗ OWUI_BASE_URL not set")
        return False

    if not client.api_token:
        print("✗ OWUI_API_TOKEN not set")
        return False

    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as http:
            headers = {"Authorization": f"Bearer {client.api_token}"}
            response = await http.get(f"{client.base_url}/api/v1/files", headers=headers)

            if response.status_code in [200, 401, 403]:
                print(f"✓ Open WebUI is accessible at {client.base_url}")
                if response.status_code == 200:
                    print(f"  ✓ API token is valid")
                    return True
                else:
                    print(f"  ✗ API token may be invalid (status {response.status_code})")
                    return False
            else:
                print(f"✗ Unexpected status: {response.status_code}")
                return False
    except Exception as e:
        print(f"✗ Open WebUI connection failed: {e}")
        print(f"  URL: {client.base_url}")
        return False


async def test_workflow_loading():
    """Test workflow JSON loading."""
    print("\nTesting workflow loading...")

    workflows_dir = Path(__file__).parent / "workflows"

    # Check qwen_image workflow
    qwen_image_path = workflows_dir / "qwen_image_api.json"
    if qwen_image_path.exists():
        try:
            with open(qwen_image_path) as f:
                workflow = json.load(f)
            print(f"✓ qwen_image_api.json loaded ({len(workflow)} nodes)")
        except Exception as e:
            print(f"✗ Failed to load qwen_image_api.json: {e}")
            return False
    else:
        print(f"✗ qwen_image_api.json not found at {qwen_image_path}")
        return False

    # Check qwen_edit workflow
    qwen_edit_path = workflows_dir / "qwen_edit_api.json"
    if qwen_edit_path.exists():
        try:
            with open(qwen_edit_path) as f:
                workflow = json.load(f)
            print(f"✓ qwen_edit_api.json loaded ({len(workflow)} nodes)")
        except Exception as e:
            print(f"✗ Failed to load qwen_edit_api.json: {e}")
            return False
    else:
        print(f"✗ qwen_edit_api.json not found at {qwen_edit_path}")
        return False

    return True


async def example_txt2img():
    """Example: Text-to-image generation.

    This is a simplified example showing the workflow.
    In production, use the MCP server tools.
    """
    print("\n" + "="*60)
    print("Example: Text-to-Image Generation")
    print("="*60)

    print("""
To use the qwen_image tool via Open WebUI:

1. Make sure the MCP server is running
2. In Open WebUI chat, say:
   "Generate a photo of a sunset over mountains"

3. The AI will call the tool with parameters like:
   {
     "prompt": "photo of a sunset over mountains",
     "width": 1024,
     "height": 1024,
     "steps": 20,
     "guidance": 5.0
   }

4. The tool will:
   - Queue the workflow in ComfyUI
   - Stream progress updates (0% → 100%)
   - Collect the generated images
   - Upload them to OWUI Files
   - Return resource links

5. You'll receive:
   - Text summary of the generation
   - Links to view/download the full images
   - (Optional) Small thumbnail preview
""")


async def example_img2img():
    """Example: Image editing.

    This is a simplified example showing the workflow.
    In production, use the MCP server tools.
    """
    print("\n" + "="*60)
    print("Example: Image Editing")
    print("="*60)

    print("""
To use the qwen_image_edit tool via Open WebUI:

1. Upload an image in the chat
   (The filter will automatically upload it to OWUI Files)

2. Say something like:
   "Make the sky more dramatic and add storm clouds"

3. The AI will call the tool with parameters like:
   {
     "prompt": "dramatic sky with storm clouds",
     "init_image_url": "https://webui.example.com/api/v1/files/abc123/content",
     "strength": 0.7,
     "steps": 30,
     "guidance": 5.0
   }

4. The tool will:
   - Download the init image from OWUI
   - Upload it to ComfyUI
   - Queue the editing workflow
   - Stream progress updates
   - Collect edited images
   - Upload them back to OWUI Files
   - Return resource links

5. You'll receive edited images as resource links
""")


async def check_environment():
    """Check environment configuration."""
    print("\n" + "="*60)
    print("Environment Configuration")
    print("="*60)

    env_vars = {
        "COMFY_URL": os.getenv("COMFY_URL", "http://127.0.0.1:8188"),
        "OWUI_BASE_URL": os.getenv("OWUI_BASE_URL", "(not set)"),
        "OWUI_API_TOKEN": "***" + os.getenv("OWUI_API_TOKEN", "")[-4:] if os.getenv("OWUI_API_TOKEN") else "(not set)",
        "HOST": os.getenv("HOST", "0.0.0.0"),
        "PORT": os.getenv("PORT", "8000"),
    }

    for key, value in env_vars.items():
        status = "✓" if value != "(not set)" else "✗"
        print(f"{status} {key}: {value}")

    print("\nIf any are missing, set them in your environment:")
    print("  export COMFY_URL='http://localhost:8188'")
    print("  export OWUI_BASE_URL='https://your-webui.com'")
    print("  export OWUI_API_TOKEN='your_token_here'")


async def main():
    """Run all examples and tests."""
    print("ComfyUI Qwen MCP Server - Example Usage")
    print("=" * 60)

    # Check environment
    await check_environment()

    # Test connections
    comfy_ok = await test_comfy_connection()
    owui_ok = await test_owui_connection()
    workflows_ok = await test_workflow_loading()

    # Show examples
    await example_txt2img()
    await example_img2img()

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    if comfy_ok and owui_ok and workflows_ok:
        print("✓ All checks passed!")
        print("\nYou can now:")
        print("1. Start the MCP server:")
        print("   cd backend/tools/mcp_servers")
        print("   python launcher.py")
        print("")
        print("2. Connect it to Open WebUI:")
        print("   Settings → Connections → MCP Servers")
        print("   Add Server: http://localhost:8000")
        print("")
        print("3. Start chatting and generating images!")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nCommon issues:")
        print("- ComfyUI not running → Start ComfyUI")
        print("- OWUI env vars not set → Export OWUI_BASE_URL and OWUI_API_TOKEN")
        print("- Workflows not configured → Export your workflows from ComfyUI")


if __name__ == "__main__":
    asyncio.run(main())
