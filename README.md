# strieber-gpt-3: Clean llama.cpp Container Build

A clean, optimized rebuild of the llama.cpp inference server for DGX Spark GB10 Blackwell with **all CLI utilities included** (llama-bench, llama-cli, llama-quantize, and more).

**Status**: Phase 1 Complete ✅

---

## What's Included

### Core Infrastructure

- **Multi-stage Docker build** optimized for DGX Spark Blackwell (sm_121)
- **All llama.cpp binaries** - server + CLI tools in one container
- **CUDA 13.0.1** with Unified Memory Architecture support
- **Docker Compose** for easy deployment
- **Makefile** with convenient command wrappers

### CLI Utilities (All Included)

```
/app/bin/
├── llama-server              # OpenAI-compatible inference server
├── llama-bench              # Performance benchmarking
├── llama-cli                # Interactive CLI interface
├── llama-quantize           # Model quantization
├── llama-perplexity         # Perplexity calculation
├── llama-embedding          # Text embeddings
├── llama-tokenize           # Tokenizer testing
├── llama-imatrix            # Importance matrix generation
└── ... (other utilities)
```

---

## Quick Start

### Prerequisites

```bash
# Docker with NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu22.04 nvidia-smi

# CUDA 13.0+ drivers
nvidia-smi
```

### Build the Container

```bash
cd /home/trevor/Projects/strieber-gpt-3

# Build (10-20 minutes first time)
make build

# Verify GPU detection
docker run --rm --gpus all strieber-llama-server:latest --version
```

### List Available Utilities

```bash
make list-binaries
```

### Download a Model

```bash
# Testing model (16GB)
make download-20b

# Or production model (63GB, 3 parts)
make download-120b
```

### Start the Server

```bash
# Start llama-server only
make up

# Or start both llama-server and Open WebUI
make up-all

# Check health
make health

# View logs
make logs
```

### Access Open WebUI

Once `make up-all` is running:

```
Open WebUI: http://localhost:3000
llama-server API: http://localhost:8000
```

Configure Open WebUI to use the llama-server endpoint (it's pre-configured by default).

### Run Utilities

```bash
# Benchmarking
make llama-bench

# Interactive CLI
make llama-cli

# Other tools
make llama-quantize
make llama-perplexity
make llama-embedding

# Help for any utility
docker compose -f compose.yml exec llama-server /app/bin/llama-bench --help
```

---

## Configuration

### Environment Variables

Edit `.env` to configure:

```bash
# Model to load
MODEL_FILE=gpt-oss-20b-Q4_K_M.gguf

# Context window
CONTEXT_SIZE=8192

# GPU layers (999 = auto)
GPU_LAYERS=999

# Server port
LLAMA_PORT=8000
```

### Model Upgrade Path

**Start with gpt-oss-20b** (16GB, fast) for testing:
```bash
make download-20b
make up
```

**Upgrade to gpt-oss-120b** (63GB, production quality):
```bash
make download-120b
# Edit .env: MODEL_FILE=gpt-oss-120b-mxfp4-00001-of-00003.gguf
make restart
```

### Open WebUI Configuration

**Open WebUI** is a beautiful web interface for interacting with your llama-server.

#### Starting Open WebUI

```bash
# Start both services
make up-all

# Access at http://localhost:3000
```

#### OpenAI API Configuration

Open WebUI is pre-configured to connect to your llama-server via:
```
Base URL: http://llama-server:8000/v1
API Key: sk-open-webui-local (dummy key)
```

This configuration is set in `compose.yml` and `.env`. The URL uses the Docker container name `llama-server` for inter-container communication.

#### Customizing Open WebUI Port

Edit `.env`:
```bash
OPENWEBUI_PORT=3000    # Change to different port if needed
```

Then restart:
```bash
make down-all
make up-all
```

#### Persisting Data

Open WebUI data (chats, settings, files) is stored in a Docker named volume `open-webui-data` and persists across container restarts.

### ComfyUI Configuration

**ComfyUI** is an advanced node-based interface for image generation with Stable Diffusion and other models.

#### Starting ComfyUI

```bash
# Starts automatically with main stack
make up

# Or manage separately
make comfyui-restart
make comfyui-logs
```

#### Accessing ComfyUI

Once services are running:

```
ComfyUI Web UI: http://localhost:9040
```

#### GPU Sharing

ComfyUI shares GPU 0 with:
- Main llama-server (gpt-oss-120b)
- ReaderLM-v2 (1.5B)
- Qwen-VL-2B (vision model)

**Memory Usage:**
- **Idle:** ~2-4GB (ComfyUI) + ~83GB (other services) = ~85-87GB
- **Generating (SD1.5):** ~10-15GB (ComfyUI) + ~83GB (other services) = ~93-96GB
- **Headroom:** ~32-35GB free during generation ✅

#### Model Storage

Models stored in `/home/trevor/models/comfyui/`:

```
/home/trevor/models/comfyui/
├── checkpoints/          # SD1.5, SDXL, etc.
├── vae/                  # VAE models
├── loras/               # LoRA fine-tunes
├── controlnet/          # ControlNet models
├── upscale_models/      # Upscaling models
├── diffusion_models/    # Other diffusion models
├── clip/                # CLIP text encoders
└── text_encoders/       # Other text encoders
```

#### Makefile Commands for ComfyUI

```bash
make comfyui-build       # Build ComfyUI image
make comfyui-logs        # View logs
make comfyui-shell       # Open bash shell
make comfyui-restart     # Restart service
make comfyui-health      # Check health status
```

#### ComfyUI Documentation

See `comfyui/README.md` for detailed ComfyUI documentation and troubleshooting.

---

## Makefile Commands

### Core Operations

```bash
make build              # Build container (10-20 min)
make up                 # Start llama-server only
make up-all             # Start llama-server + Open WebUI
make down               # Stop all containers
make down-all           # Alias for 'down'
make restart            # Restart all containers
make logs               # Show logs from all services (follow)
make health             # Check health endpoint
make status             # Container status
make shell              # Bash shell in llama-server container
```

### CLI Utilities

```bash
make llama-bench        # Run benchmarks
make llama-cli          # Interactive CLI
make llama-quantize     # Quantize models
make llama-perplexity   # Perplexity calc
make llama-embedding    # Generate embeddings
make llama-tokenize     # Test tokenizer
make llama-imatrix      # Importance matrix
make list-binaries      # Show all utilities
```

### Model Management

```bash
make download-20b       # Download testing model
make download-120b      # Download production model
make models-info        # Show downloaded models
```

### Benchmarking

```bash
make bench-sequential   # Sequential benchmark (prefill/generation across context depths)
make bench-parallel     # Parallel benchmark (128k context, batch sizes 1-16)
make bench-extreme      # Extreme stress test (300k context, batch sizes 1-32)
make bench-all          # Run all benchmarks sequentially
make bench-list         # List previous benchmark results
```

**Results** are saved to `benchmarks/` directory as timestamped JSONL files for analysis.

### Container Management

```bash
make copy-binary BIN=llama-bench    # Copy binary to host
make container-stats                 # Real-time resource usage
make clean                           # Remove everything
```

---

## Directory Structure

```
strieber-gpt-3/
├── Dockerfile.llamacpp      # Multi-stage build (all utilities)
├── compose.yml              # Docker Compose config (llama-server + Open WebUI)
├── Makefile                 # Development commands
├── README.md               # This file
├── .env.example            # Configuration template
├── .env                    # Your configuration (gitignored)
├── models/                 # GGUF model files
│   └── (empty until downloaded)
└── docs/
    └── PHASE1-README.md    # Detailed build documentation
```

---

## Performance

### Build Time
- **First build**: 10-20 minutes (network dependent)
- **Cached builds**: 2-5 minutes
- **Image size**: ~500MB runtime

### Inference Performance (DGX Spark)

**With gpt-oss-20b** (16GB):
- Tokens/sec: 40-60
- First token: <1 second
- Memory: ~20-25GB

**With gpt-oss-120b** (63GB):
- Tokens/sec: 18-30
- First token: <2 seconds
- Memory: ~70-80GB

---

## GPU Optimization

### Build Configuration

The Dockerfile is optimized for **DGX Spark GB10 Blackwell**:

```dockerfile
# Blackwell architecture (sm_121)
-DCMAKE_CUDA_ARCHITECTURES=121

# 128GB Unified Memory
-DGGML_CUDA_ENABLE_UNIFIED_MEMORY=1

# ARM64 compatible
-DGGML_NATIVE=OFF
```

To change target GPU, edit `Dockerfile.llamacpp`:
- Change `CUDA_DOCKER_ARCH` build arg
- Change `CMAKE_CUDA_ARCHITECTURES` CMake flag

---

## Benchmarking

The strieber-gpt-3 project includes comprehensive benchmarking tools to measure model performance across different scenarios.

### Sequential Benchmarks (make bench-sequential)

Tests **prefill and generation performance** across varying context depths:

```bash
make bench-sequential  # ~5 minutes
```

**What it tests:**
- Prefill speed at 2048 tokens (pp2048)
- Generation speed for 32 tokens (tg32)
- Performance at context depths: 0, 4k, 8k, 16k, 32k tokens
- Impact of KV cache size on throughput

**Expected output (gpt-oss-20b on Blackwell):**
- Prefill: 2,000-3,000 tokens/sec
- Generation: 40-60 tokens/sec
- Degrades as context depth increases

### Parallel Benchmarks (make bench-parallel)

Tests **multi-user throughput** with concurrent requests:

```bash
make bench-parallel    # ~10 minutes
```

**What it tests:**
- 128k context window (realistic scenario)
- Prompt sizes: 4k and 8k tokens
- Batch sizes: 1, 2, 4, 8, 16 concurrent requests
- Throughput scaling with parallelism
- Per-request latency

**Key metrics:**
- Total throughput (tokens/second across all requests)
- Per-request latency
- Scaling efficiency

### Extreme Stress Test (make bench-extreme)

Tests **maximum capacity** with large context and high concurrency:

```bash
make bench-extreme     # ~20 minutes
```

**What it tests:**
- 300k context window (stress test)
- Batch sizes: 1, 2, 4, 8, 16, 32 concurrent requests
- Memory pressure and thermal behavior
- Maximum theoretical throughput

**⚠️ Warning**: This test pushes GPU memory to limits. Monitor with `nvidia-smi` in another terminal.

### Run All Benchmarks

```bash
make bench-all         # ~35 minutes total
```

Sequentially runs all three benchmark suites and lists results.

### Analyzing Results

Results are saved to `benchmarks/bench-[type]-[timestamp].log` in JSONL format:

```bash
# View results
cat benchmarks/bench-sequential-*.log | head -20

# Parse with jq
cat benchmarks/bench-sequential-*.log | jq '.name, .t_per_token_ms'

# Compare runs
diff <(cat benchmarks/bench-sequential-20251102-120000.log) \
     <(cat benchmarks/bench-sequential-20251102-130000.log)
```

### GPU Monitoring During Benchmarks

In another terminal, monitor resource usage:

```bash
watch -n 1 nvidia-smi

# Or specific GPU metrics
nvidia-smi dmon  # Real-time memory and power
```

Watch for:
- **GPU Memory**: Should stay under 128GB
- **GPU Utilization**: 90-100% during benchmarking
- **Temperature**: Normal operating range (under 80°C)
- **Power Usage**: Typically 200-300W for 20B model

---

## Next Steps

### Phase 2: Model Inference Testing
1. Download a model
2. Start the server
3. Test with inference requests
4. Run benchmarks

### Phase 3: Backend Integration
1. Add FastAPI backend
2. Implement agent orchestration
3. Add tool servers (MCP)

### Phase 4: Production Tools
1. Web search integration
2. Code execution sandbox
3. Streaming responses

---

## Troubleshooting

### Build fails with CUDA errors
- Check CUDA 13.0+: `nvidia-smi`
- Test CUDA base image: `docker run --rm nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu22.04 nvcc --version`
- Install NVIDIA Container Toolkit

### GPU not detected in container
- Test GPU access: `docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu22.04 nvidia-smi`
- Reinstall NVIDIA Container Toolkit

### Server exits immediately
- Check logs: `make logs`
- Verify model file exists: `make models-info`
- Ensure GPU is available

### Out of memory during build
- Reduce parallel jobs in Dockerfile
- Change `-j$(nproc)` to `-j4`

See `docs/PHASE1-README.md` for detailed troubleshooting.

---

## Architecture

### Why This Design?

1. **Separate server**: Inference isolation, easy scaling
2. **All utilities included**: Benchmarking, quantization, analysis
3. **Multi-stage build**: Small runtime image (~500MB)
4. **GPU optimized**: Target-specific architecture
5. **Clean foundation**: Ready for backend layers

### Based On

NVIDIA multi-agent-chatbot and llama.cpp official builds, adapted for:
- Single model focus
- DGX Spark optimization
- Comprehensive CLI access
- Educational clarity

---

## References

- **llama.cpp**: https://github.com/ggml-org/llama.cpp
- **NVIDIA CUDA**: https://docs.nvidia.com/cuda/
- **DGX Spark**: https://docs.nvidia.com/dgx/dgx-spark/
- **Models**: https://huggingface.co/ggml-org/gpt-oss-120b-GGUF

---

## Support

For issues or questions:

1. Check `make help` for all available commands
2. Review `docs/PHASE1-README.md` for detailed docs
3. Check Docker logs: `make logs`
4. Verify prerequisites: CUDA 13.0+, NVIDIA Container Toolkit
5. Test GPU: `docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu22.04 nvidia-smi`

---

**Ready to build!** 🚀

```bash
cd /home/trevor/Projects/strieber-gpt-3
make build
```
