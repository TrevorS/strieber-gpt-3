#!/usr/bin/env python3
"""
Example usage of the ComfyUI Qwen MCP client tools.

This demonstrates how to call the MCP tools programmatically.
For actual MCP usage, you'd typically use an MCP client library.
"""

import asyncio
import json

from comfy_client import ComfyUIClient
from owui_client import OWUIClient
from server import qwen_image, qwen_image_edit


async def example_txt2img():
    """Example: Generate an image from text prompt."""
    print("\n" + "=" * 60)
    print("Example 1: Text-to-Image Generation")
    print("=" * 60)

    result = await qwen_image(
        prompt="a serene mountain landscape at sunset, highly detailed, 8k",
        negative_prompt="blurry, low quality, distorted, ugly",
        width=768,
        height=512,
        steps=25,
        guidance=4.5,
        seed=42,
        batch_size=1,
        inline_preview=False,
        upload_results_to_openwebui=True,
    )

    print("\nResult content blocks:")
    for idx, block in enumerate(result, 1):
        print(f"\n[Block {idx}] Type: {block.get('type', block.__class__.__name__)}")
        if hasattr(block, 'text'):
            print(f"  Text: {block.text}")
        elif hasattr(block, 'resource'):
            print(f"  Resource URI: {block.resource.get('uri')}")
            print(f"  MIME Type: {block.resource.get('mimeType')}")


async def example_img2img():
    """Example: Edit an existing image."""
    print("\n" + "=" * 60)
    print("Example 2: Image-to-Image Editing")
    print("=" * 60)

    # Note: You need to provide a valid file_id or URL from OWUI
    # This is just a demonstration of the API
    print("\nNote: This example requires a valid init_image_file_id or init_image_url")
    print("Upload an image to Open WebUI first, then use its file ID here.")

    # Uncomment and modify with your actual file ID:
    """
    result = await qwen_image_edit(
        prompt="add dramatic storm clouds to the sky",
        init_image_file_id="your_file_id_here",
        strength=0.65,
        steps=30,
        guidance=5.0,
        seed=777,
        inline_preview=False,
        upload_results_to_openwebui=True,
    )

    print("\nResult content blocks:")
    for idx, block in enumerate(result, 1):
        print(f"\n[Block {idx}] Type: {block.get('type', block.__class__.__name__)}")
        if hasattr(block, 'text'):
            print(f"  Text: {block.text}")
        elif hasattr(block, 'resource'):
            print(f"  Resource URI: {block.resource.get('uri')}")
    """


async def example_client_usage():
    """Example: Direct client usage (without MCP tools)."""
    print("\n" + "=" * 60)
    print("Example 3: Direct ComfyUI Client Usage")
    print("=" * 60)

    # Initialize clients
    comfy = ComfyUIClient()
    owui = OWUIClient()

    print(f"\nComfyUI URL: {comfy.base_url}")
    print(f"OWUI URL: {owui.base_url}")

    # You could build and queue a workflow manually:
    """
    from pathlib import Path
    import json

    workflow_path = Path(__file__).parent / "workflows" / "qwen_image_api.json"
    with open(workflow_path) as f:
        workflow = json.load(f)

    # Patch workflow with your parameters
    workflow["1"]["inputs"]["text"] = "your prompt here"
    # ... etc

    # Queue and monitor
    prompt_id = await comfy.queue_prompt(workflow)
    print(f"Queued workflow: {prompt_id}")

    async for progress in comfy.progress(prompt_id):
        print(f"Progress: {progress}%")

    # Collect results
    outputs = await comfy.collect_output_files(prompt_id)
    print(f"Generated {len(outputs)} images")
    """

    print("\nSee server.py for full workflow patching examples.")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("ComfyUI Qwen MCP - Usage Examples")
    print("=" * 60)

    try:
        await example_txt2img()
    except Exception as e:
        print(f"\n✗ Example 1 failed: {e}")
        print("  Make sure ComfyUI and Open WebUI are running and configured.")

    await example_img2img()
    await example_client_usage()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nFor actual MCP usage, connect an MCP client to this server.")
    print("The server provides these tools:")
    print("  - qwen_image (text-to-image)")
    print("  - qwen_image_edit (image-to-image)")


if __name__ == "__main__":
    asyncio.run(main())
