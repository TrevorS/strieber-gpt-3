# ComfyUI for NVIDIA DGX Spark
# Base: NGC PyTorch 25.09 with CUDA 13.0 for Blackwell GB10 GPU support
FROM nvcr.io/nvidia/pytorch:25.09-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # Enable expandable segments for better GPU memory management
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # ComfyUI server configuration
    COMFY_LISTEN_HOST=0.0.0.0 \
    COMFY_PORT=8188

# Build-time args for user mapping (avoid root-owned files on host)
ARG USERNAME=app
ARG UID=1000
ARG GID=1000

# Install system dependencies
# - git: Clone ComfyUI repo
# - wget/curl: Download models and healthcheck
# - ffmpeg: Video processing support
# - libgl1: OpenGL support for image operations
# - python3-dev: Python development headers for native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      wget \
      curl \
      ca-certificates \
      ffmpeg \
      libgl1 \
      python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user to match host UID/GID
# Handle case where GID/UID might already exist in base image
RUN (groupadd -g ${GID} ${USERNAME} 2>/dev/null || groupmod -n ${USERNAME} $(getent group ${GID} | cut -d: -f1)) && \
    (useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} 2>/dev/null || usermod -l ${USERNAME} -d /home/${USERNAME} -m $(getent passwd ${UID} | cut -d: -f1))

WORKDIR /opt

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

WORKDIR /opt/ComfyUI

# Install ComfyUI dependencies
# Note: PyTorch (torch, torchvision, torchaudio) is already in the NGC base image
# We must exclude ONLY these exact packages from requirements.txt to avoid installing CPU-only versions
RUN python -m pip install --upgrade pip && \
    grep -v -E '^(torch|torchvision|torchaudio)($|[^a-z])' requirements.txt > /tmp/requirements_filtered.txt && \
    python -m pip install -r /tmp/requirements_filtered.txt

# Create well-known directories for bind mounts
# These will be mounted from the host to persist data
RUN mkdir -p \
      /opt/ComfyUI/models/checkpoints \
      /opt/ComfyUI/models/clip \
      /opt/ComfyUI/models/vae \
      /opt/ComfyUI/models/loras \
      /opt/ComfyUI/models/controlnet \
      /opt/ComfyUI/models/upscale_models \
      /opt/ComfyUI/input \
      /opt/ComfyUI/output \
      /opt/ComfyUI/custom_nodes

# Healthcheck to verify ComfyUI web server is running
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=10 \
  CMD curl -sfI http://127.0.0.1:${COMFY_PORT} || exit 1

# Copy and set permissions on entrypoint script
COPY --chown=${UID}:${GID} docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Change ownership of ComfyUI directory to non-root user
# This allows the user to create necessary directories at runtime
RUN chown -R ${UID}:${GID} /opt/ComfyUI

# Switch to non-root user
USER ${USERNAME}

# Expose ComfyUI web interface port
EXPOSE 8188

ENTRYPOINT ["/entrypoint.sh"]
