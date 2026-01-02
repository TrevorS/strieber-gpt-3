# ABOUTME: AI-Toolkit training container for Z-Image Turbo LoRA training.
# Uses NGC PyTorch for native Blackwell GPU optimization.

# NGC PyTorch 25.12 - proven working pattern from comfyui/Dockerfile
FROM nvcr.io/nvidia/pytorch:25.12-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # Enable expandable segments for better GPU memory management
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # Persist torch.compile kernels across container restarts
    TORCHINDUCTOR_CACHE_DIR=/data/.torch_cache \
    TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
    # Triton has its own cache that must be set separately
    TRITON_CACHE_DIR=/data/.triton_cache

WORKDIR /app

# Clone ai-toolkit
RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git . && \
    git submodule update --init --recursive

# Unset PIP_CONSTRAINT to avoid NGC's constraint file interfering
# Install dependencies in stages to avoid resolution-too-deep error
RUN unset PIP_CONSTRAINT && python -m pip install --upgrade pip

# Stage 1: Install packages that don't depend on torch (safe batch install)
RUN unset PIP_CONSTRAINT && pip install \
    safetensors pyyaml oyaml flatten_json toml pydantic omegaconf \
    python-dotenv hf_transfer sentencepiece huggingface_hub python-slugify \
    tensorboard einops timm invisible-watermark

# Stage 2: Install packages that depend on torch with --no-deps
# The torch dependency is already satisfied by NGC base image
RUN unset PIP_CONSTRAINT && pip install --no-deps \
    lycoris-lora==1.8.3 \
    kornia \
    accelerate \
    bitsandbytes \
    lpips \
    pytorch_fid \
    optimum-quanto==0.2.4 \
    peft \
    k-diffusion \
    open_clip_torch \
    controlnet_aux==0.0.10 \
    prodigyopt \
    pytorch-wavelets==1.3.0

# Stage 3: Install albumentations with fixed versions (has complex deps)
RUN unset PIP_CONSTRAINT && pip install \
    albucore==0.0.16 \
    albumentations==1.4.15

# Stage 4: Install transformers and diffusers (huggingface stack)
RUN unset PIP_CONSTRAINT && pip install \
    transformers==4.57.3 \
    diffusers

# Stage 5: Install missing deps for k-diffusion and open-clip-torch
RUN unset PIP_CONSTRAINT && pip install \
    ftfy \
    clean-fid \
    clip-anytorch \
    dctorch \
    jsonmerge \
    torchdiffeq \
    torchsde \
    wandb

# Stage 6: Install gradio and remaining deps
RUN unset PIP_CONSTRAINT && pip install \
    gradio \
    opencv-python \
    matplotlib==3.10.1

# Create stub torchaudio module (ai-toolkit imports it but only uses for audio export)
# We don't need audio export for LoRA training
RUN mkdir -p /usr/local/lib/python3.12/dist-packages/torchaudio && \
    echo 'def save(*args, **kwargs): pass' > /usr/local/lib/python3.12/dist-packages/torchaudio/__init__.py

# Create directories for weights
RUN mkdir -p /weights/z-image-turbo

# Entrypoint runs training
RUN printf '#!/bin/bash\nset -e\nexec python run.py "$@"\n' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
