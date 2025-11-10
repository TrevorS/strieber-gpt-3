#!/usr/bin/env bash
set -euo pipefail

# Get host and port from environment variables or use defaults
HOST="${COMFY_LISTEN_HOST:-0.0.0.0}"
PORT="${COMFY_PORT:-8188}"

echo "========================================="
echo "ComfyUI for NVIDIA DGX Spark"
echo "========================================="
echo "Host: ${HOST}"
echo "Port: ${PORT}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================="

# Optional: Auto-download Stable Diffusion 1.5 checkpoint on first run
if [[ "${DOWNLOAD_SD15:-0}" == "1" ]]; then
  mkdir -p /opt/ComfyUI/models/checkpoints
  CHECKPOINT_PATH="/opt/ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"

  if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Downloading Stable Diffusion 1.5 model (~2GB)..."
    echo "This may take a few minutes depending on your network speed."
    wget -O "${CHECKPOINT_PATH}" \
      "https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors"
    echo "Download complete!"
  else
    echo "SD1.5 checkpoint already present at ${CHECKPOINT_PATH}"
  fi
fi

echo "Starting ComfyUI..."
echo "========================================="

# Start ComfyUI with configurable host/port and pass through any additional arguments
exec python3 main.py --listen "${HOST}" --port "${PORT}" "$@"
