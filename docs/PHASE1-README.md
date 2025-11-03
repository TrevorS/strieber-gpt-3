# Phase 1: llama.cpp Server Setup for DGX Spark (Clean Build)

**Status**: ✅ Infrastructure Complete
**Goal**: Build llama.cpp server optimized for DGX Spark GB10 Blackwell GPU
**New**: Includes all CLI utilities (llama-bench, llama-cli, llama-quantize, etc.)

---

## Overview

Phase 1 establishes the inference infrastructure for strieber-gpt. We build llama.cpp from source in Docker with optimizations specific to DGX Spark's:
- **GB10 Blackwell GPU** (compute capability 12.1)
- **128GB Unified Memory Architecture (UMA)**
- **ARM64 20-core CPU**
- **CUDA 13.0+**

**New in strieber-gpt-3**: This clean build now includes all llama.cpp CLI utilities for benchmarking, quantization, and analysis.

---

## 📦 Model Configuration

**Phase 1-2 (Testing)**: Uses **gpt-oss-20b-Q4_K_M** (~16GB)
- ✅ Faster download and testing (5-15 minutes vs 30-60 minutes)
- ✅ Lower memory requirements (~20-25GB vs ~70-80GB)
- ✅ Validates full infrastructure works
- ✅ Recommended for development and iteration

**Phase 3+ (Production)**: Upgrade to **gpt-oss-120b** (~63GB)
- ✅ Higher quality responses
- ✅ Larger context window (128k vs 8k tokens)
- ✅ Production-ready performance
- ✅ Easy upgrade when testing is complete

See "Next Steps" section for upgrade instructions.

---

## Quick Start

### Prerequisites

- Docker with NVIDIA Container Toolkit installed
- DGX Spark or compatible NVIDIA GPU system
- CUDA 13.0+ drivers
- Sufficient disk space (~10GB for Docker images)

### Build the Server

```bash
# 1. Navigate to project root
cd /home/trevor/Projects/strieber-gpt-3

# 2. Build the llama.cpp Docker image
make build
# Or: docker compose -f compose.llama.yml build

# Expected: Build completes in 10-20 minutes (first time)
# Output: "Successfully tagged strieber-llama-server:latest"
```

### Verify the Build

```bash
# Test 1: Check GPU detection
docker run --rm --gpus all strieber-llama-server:latest --version

# Expected output should include:
# - llama.cpp version number
# - "CUDA: 1" or similar (confirms CUDA enabled)
# - No errors about missing libraries

# Test 2: View available options
docker run --rm strieber-llama-server:latest --help | grep -E "(gpu-layers|cuda|jinja)"

# Expected: Should see all configured flags listed

# Test 3: List available utilities
make list-binaries
# Shows all utilities in /app/bin/
```

### Start the Server (Will Fail - Expected)

```bash
# Try to start without model (will fail - this is normal)
make up
# Or: docker compose -f compose.llama.yml up -d

# Expected error:
# "error loading model: failed to load model from /models/..."
#
# This is CORRECT behavior - we haven't downloaded the model yet.
# We're just verifying the build and configuration work.
```

---

## What Was Built

### 1. Multi-Stage Docker Image

**Build Stage** (`nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu22.04`):
- Installs build dependencies (cmake, ninja, gcc, etc.)
- Clones llama.cpp from GitHub
- Compiles with DGX Spark optimizations
- Extracts binaries and shared libraries

**Runtime Stage** (`nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu22.04`):
- Minimal runtime environment (~500MB vs ~5GB build stage)
- **Includes ALL llama.cpp binaries** (server + CLI utilities)
- CUDA runtime libraries
- curl for health checks

### 2. Included CLI Utilities

The container now includes all llama.cpp CLI tools:

- **llama-server** - OpenAI-compatible inference server
- **llama-bench** - Performance benchmarking tool
- **llama-cli** - Interactive CLI inference interface
- **llama-quantize** - Model quantization utility
- **llama-perplexity** - Perplexity calculation
- **llama-embedding** - Text embedding generation
- **llama-tokenize** - Tokenizer testing
- **llama-imatrix** - Importance matrix generation

Access them via:
```bash
# Interactive:
make llama-cli

# Benchmarks:
make llama-bench

# Other utilities:
make llama-quantize
make llama-perplexity
# etc...

# Or directly via docker exec:
docker compose -f compose.llama.yml exec llama-server /app/bin/llama-bench
```

### 3. Critical Build Flags

The following CMake flags are **essential** for DGX Spark:

```cmake
-DCMAKE_CUDA_ARCHITECTURES=121         # Blackwell GB10 (sm_121)
-DGGML_CUDA_ENABLE_UNIFIED_MEMORY=1    # Enable 128GB UMA
-DGGML_NATIVE=OFF                      # ARM64 compatibility
-DGGML_CUDA=ON                         # CUDA acceleration
-DGGML_BACKEND_DL=ON                   # Dynamic backends
-DLLAMA_BUILD_TESTS=OFF                # Skip tests
```

**Why these flags matter**:
- **sm_121**: Wrong architecture = slower inference or build failure
- **UMA**: Without this, can't utilize full 128GB shared memory
- **NATIVE=OFF**: x86 optimizations would crash on ARM64
- **CUDA=ON**: Core requirement for GPU acceleration

### 4. Docker Compose Configuration

**Service: llama-server**
- **GPU**: All available GPUs with compute capability
- **Port**: 8000 (configurable via LLAMA_PORT in .env)
- **Volumes**: `./models:/models` for GGUF files
- **Health**: HTTP endpoint at `/health`
- **Restart**: Unless manually stopped

**Optimized for gpt-oss-120b**:
- 64k context window (configurable to 128k)
- All 70 layers offloaded to GPU
- Jinja template support for chat formatting

---

## Configuration

### Environment Variables

Copy the example file and customize:

```bash
cp .env.example .env
nano .env  # or your preferred editor
```

**Key variables**:

```bash
# Model file (must exist in ./models/)
MODEL_FILE=gpt-oss-120b-mxfp4-00001-of-00003.gguf

# Context size: 8192 recommended for 20b, up to 131072 (128k) for 120b
CONTEXT_SIZE=8192

# GPU layers: auto-detect
GPU_LAYERS=999

# Server port
LLAMA_PORT=8000
```

### Directory Structure

```
strieber-gpt-3/
├── Dockerfile.llamacpp       # Multi-stage build (includes all utilities)
├── compose.llama.yml         # Service configuration
├── Makefile                  # Development tasks + utility wrappers
├── .env                      # Your configuration (not in git)
├── .env.example              # Configuration template
├── models/                   # GGUF model files (empty for now)
└── docs/                     # Documentation
    └── PHASE1-README.md
```

---

## Makefile Utilities

The Makefile provides convenient wrappers for all tools:

```bash
# Core commands
make build              # Build the container
make up                 # Start llama-server
make down               # Stop llama-server
make logs               # Show logs (follow mode)
make health             # Check health endpoint

# CLI Utilities
make llama-bench        # Run benchmarks
make llama-cli          # Interactive CLI
make llama-quantize     # Quantize models
make llama-perplexity   # Calculate perplexity
make llama-embedding    # Generate embeddings
make llama-tokenize     # Test tokenizer
make llama-imatrix      # Generate importance matrix

# Model Management
make download-20b       # Download testing model (16GB)
make download-120b      # Download production model (63GB)
make models-info        # Show available models

# Container info
make list-binaries      # Show all available utilities
make copy-binary BIN=llama-bench  # Copy binary to host

# Utilities
make shell              # Open shell in container
make status             # Check container status
make clean              # Remove everything
```

---

## Testing Without a Model

You can verify the build works without downloading the 63GB model:

### Test 1: GPU Detection

```bash
docker run --rm --gpus all strieber-llama-server:latest --version
```

**Expected output**:
```
llama.cpp version: <version>
CUDA: 1
...
```

If you see "CUDA: 0" or errors, check:
- NVIDIA Container Toolkit is installed: `docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu22.04 nvidia-smi`
- Drivers are CUDA 13.0+: `nvidia-smi`

### Test 2: Help Menu

```bash
docker run --rm strieber-llama-server:latest --help
```

**Expected**: Full help output with all flags listed

### Test 3: List Binaries

```bash
make list-binaries
```

**Expected**: Output showing all llama.cpp utilities in `/app/bin/`

### Test 4: Check Health Endpoint (After Model Download)

```bash
# Start server with model (Phase 2)
make up

# Wait 60 seconds for model to load
sleep 60

# Check health
make health
# Or: curl http://localhost:8000/health

# Expected: {"status": "ok"} or similar
```

---

## Troubleshooting

### Build Fails: "CUDA not found"

**Symptom**: CMake can't find CUDA during build

**Fix**:
```bash
# Verify CUDA base image works
docker run --rm nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu22.04 nvcc --version

# Should show: Cuda compilation tools, release 13.0
```

If this fails, you need CUDA 13.0+ drivers on the host.

### Build Fails: "Compute capability not supported"

**Symptom**: Error about sm_121 not being supported

**Fix**: Ensure you're using CUDA 13.0+. Older CUDA versions don't support Blackwell.

```bash
# Check your CUDA version
nvidia-smi

# Driver Version should be 570.xx.xx or higher
```

### GPU Not Detected in Container

**Symptom**: `--version` shows "CUDA: 0"

**Fix**: Install/configure NVIDIA Container Toolkit

```bash
# Test GPU passthrough
docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu22.04 nvidia-smi

# If this fails, install nvidia-container-toolkit:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Out of Memory During Build

**Symptom**: Build fails with memory errors

**Fix**: Reduce parallel jobs

```bash
# Edit Dockerfile.llamacpp, line with cmake --build
# Change: -j$(nproc)
# To: -j4

make build
```

### Container Exits Immediately

**Symptom**: `make up` exits with error

**Likely causes**:
1. **No model file** (expected in Phase 1) - error will say "failed to load model"
2. **Wrong model path** - check MODEL_FILE in .env matches actual file
3. **GPU not available** - verify `make health` works

**Check logs**:
```bash
make logs
```

---

## Performance Expectations

### Build Time

- **First build**: 10-20 minutes (network dependent)
- **Subsequent builds**: 2-5 minutes (cached layers)
- **Image size**: ~500MB runtime, ~5GB build stage (cached)

### Runtime Performance (with model - Phase 2)

**Expected performance on DGX Spark with gpt-oss-20b** (testing):

- **Token generation**: 40-60 tokens/second
- **First token latency**: <1 second
- **Memory usage**: ~20-25GB (including KV cache)
- **Context processing**: 200-400 tokens/second
- **Download time**: 5-15 minutes

**Expected performance on DGX Spark with gpt-oss-120b** (production):

- **Token generation**: 18-30 tokens/second
- **First token latency**: <2 seconds
- **Memory usage**: ~70-80GB (including KV cache)
- **Context processing**: ~100-200 tokens/second
- **Download time**: 30-60 minutes

*Use 20b for testing, upgrade to 120b when you need higher quality.*

---

## Next Steps (Phase 2)

Once the build is verified, Phase 2 will:

1. **Download the test model** (16GB for gpt-oss-20b):
   ```bash
   make download-20b
   ```

2. **Start the server**:
   ```bash
   make up
   ```

3. **Test inference**:
   ```bash
   curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "gpt-oss-20b",
       "messages": [{"role": "user", "content": "Hello!"}],
       "stream": false,
       "max_tokens": 100
     }'
   ```

4. **Run benchmarks**:
   ```bash
   make llama-bench
   ```

5. **Optimize parameters** if needed

### Upgrading to gpt-oss-120b (Phase 3+)

When ready for production quality:

1. **Download 120b model** (63.4GB):
   ```bash
   make download-120b
   ```

2. **Update .env**:
   ```bash
   MODEL_FILE=gpt-oss-120b-mxfp4-00001-of-00003.gguf
   CONTEXT_SIZE=65536
   GPU_LAYERS=70
   ```

3. **Restart server**:
   ```bash
   make restart
   ```

---

## Success Criteria

Phase 1 is complete when:

- ✅ Docker image builds successfully
- ✅ Build includes CUDA 13.0.1 and targets sm_121
- ✅ llama-server binary runs (help/version work)
- ✅ All CLI utilities are present (llama-bench, llama-cli, etc.)
- ✅ GPU passthrough confirmed via test commands
- ✅ docker-compose.yml validates without errors
- ✅ All configuration files are in place
- ✅ Makefile utilities work

**Current Status**: ✅ All criteria met - ready for Phase 2

---

## Architecture Notes

### Why Separate llama.cpp Server?

We use llama.cpp as a **separate service** (not embedded library) because:

1. **Process isolation**: Inference crash doesn't kill API
2. **Easy backend swapping**: Can replace with vLLM/TRT-LLM later
3. **Horizontal scaling**: Add more inference servers easily
4. **Hot reload**: Update API code without reloading 80GB model
5. **Multi-user support**: Server handles concurrent requests efficiently
6. **CLI access**: Can run benchmarks, analysis tools without API

---

## Support

If you encounter issues:

1. Check this troubleshooting section
2. Review logs: `make logs`
3. Check Docker and NVIDIA Container Toolkit installation
4. Verify CUDA 13.0+ drivers
5. Check DGX Spark hardware compatibility

---

**Phase 1 Complete!** 🎉

Ready to proceed to Phase 2: Model download and inference testing.
