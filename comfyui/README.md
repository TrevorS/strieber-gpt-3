# ComfyUI for Strieber-GPT-3

ComfyUI integrated into the strieber-gpt-3 multi-agent stack for advanced image generation workflows.

## Quick Start

```bash
# Start all services (includes ComfyUI)
make up

# Access ComfyUI Web UI
open http://localhost:9040
```

## Configuration

- **Port:** 9040 (configurable via `COMFYUI_PORT` in `.env`)
- **Models:** `/home/trevor/models/comfyui/`
- **Outputs:** `./comfyui-data/output/`
- **GPU:** Shared GPU 0 with main llama-server, ReaderLM-v2, and Qwen-VL

## Service Management

```bash
# View ComfyUI logs (follow mode)
make comfyui-logs

# Open bash shell in container
make comfyui-shell

# Restart service
make comfyui-restart

# Check health status
make comfyui-health
```

## Model Storage

Place models in `/home/trevor/models/comfyui/`:

- **Checkpoints:** `checkpoints/` (stable-diffusion, etc.)
- **LoRAs:** `loras/`
- **VAE:** `vae/`
- **ControlNet:** `controlnet/`
- **Text Encoders:** `text_encoders/`
- **Upscale Models:** `upscale_models/`
- **Diffusion Models:** `diffusion_models/`
- **CLIP:** `clip/`

## GPU Sharing

ComfyUI shares GPU 0 (128GB) with:
- Main llama-server (gpt-oss-120b)
- ReaderLM-v2 (1.5B)
- Qwen-VL-2B (vision model)

**Memory Usage:**
- **Idle:** ~2-4GB (ComfyUI) + ~83GB (other services) = ~85-87GB
- **Generating (SD1.5):** ~10-15GB (ComfyUI) + ~83GB (other services) = ~93-96GB
- **Headroom:** ~32-35GB free during generation ✅

## Workflow

1. Create/load a workflow in ComfyUI UI
2. Configure generation parameters
3. Queue nodes and generate images
4. Outputs saved to `./comfyui-data/output/`

## API Endpoints

ComfyUI API available at:
```
http://localhost:9040/api/
```

See [ComfyUI API docs](https://github.com/comfyanonymous/ComfyUI/blob/master/server.py) for available endpoints.

## Logs

Watch real-time logs:
```bash
make comfyui-logs

# Or directly with docker
docker compose logs -f comfyui
```

Look for:
- GPU detection on startup
- Model loading messages
- Generation progress during inference

## Troubleshooting

**Issue: Port 9040 already in use**
- Change `COMFYUI_PORT` in `.env`
- Or find process: `lsof -i :9040`

**Issue: GPU memory exhaustion**
- Check current usage: `nvidia-smi`
- Reduce batch size in workflows
- Use smaller models temporarily

**Issue: Models not found**
- Verify paths: `ls -la /home/trevor/models/comfyui/`
- Check UI console for exact errors
- Restart container: `make comfyui-restart`

**Issue: Web UI not responding**
- Check logs: `make comfyui-logs`
- Verify container running: `docker ps | grep comfyui`
- Allow 20-30 seconds for startup

## References

- [ComfyUI Official Repository](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI Documentation](https://github.com/comfyanonymous/ComfyUI/wiki)
- Optimized for: NVIDIA DGX Spark Blackwell GB10 (H200 GPU)
