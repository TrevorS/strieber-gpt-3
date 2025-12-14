# ABOUTME: Training job execution via Docker.
# Manages ai-toolkit container lifecycle and monitors training progress.

import asyncio
import logging
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import docker
import os
import yaml
from docker.errors import APIError, ContainerError, ImageNotFound

from lora_trainer.job_store import JobStore
from lora_trainer.models import TrainingConfig, TrainingJob, TrainingStatus


logger = logging.getLogger(__name__)

DOCKER_IMAGE = "strieber-ai-toolkit:latest"
LORAS_OUTPUT_PATH = Path("/output/loras")

# Host path for training data (passed via environment)
# This is needed because the MCP server runs in Docker but spawns sibling containers
HOST_DATA_PATH = os.environ.get("HOST_DATA_PATH", "/home/trevor/lora-training")
HOST_MODELS_PATH = os.environ.get("HOST_MODELS_PATH", "/home/trevor/models")


class TrainingRunner:
    """Manages ai-toolkit training jobs via Docker."""

    def __init__(
        self,
        datasets_path: Path,
        outputs_path: Path,
        configs_path: Path,
        job_store: JobStore,
        loras_path: Optional[Path] = None,
    ):
        """Initialize training runner.

        Args:
            datasets_path: Path to datasets directory.
            outputs_path: Path to training outputs.
            configs_path: Path for config files.
            job_store: Job persistence store.
            loras_path: Path to ComfyUI loras directory.
        """
        self.datasets_path = Path(datasets_path)
        self.outputs_path = Path(outputs_path)
        self.configs_path = Path(configs_path)
        self.job_store = job_store
        self.loras_path = Path(loras_path) if loras_path else LORAS_OUTPUT_PATH

        # Ensure directories exist
        self.outputs_path.mkdir(parents=True, exist_ok=True)
        self.configs_path.mkdir(parents=True, exist_ok=True)

        # Initialize Docker client
        try:
            self._docker = docker.from_env()
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
            self._docker = None

        self._active_containers: dict[str, str] = {}

    async def start_training(
        self,
        dataset_name: str,
        trigger_token: str,
        config: TrainingConfig,
    ) -> TrainingJob:
        """Start a training job in Docker container.

        Args:
            dataset_name: Name of the dataset.
            trigger_token: Trigger token for the LoRA.
            config: Training configuration.

        Returns:
            Created training job.
        """
        job_id = str(uuid.uuid4())[:8]

        # Create job
        job = TrainingJob(
            job_id=job_id,
            dataset_name=dataset_name,
            trigger_token=trigger_token,
            config=config,
            total_steps=config.steps,
        )
        self.job_store.save_job(job)

        # Create output directory
        job_output_path = self.outputs_path / job_id
        job_output_path.mkdir(parents=True, exist_ok=True)
        (job_output_path / "checkpoints").mkdir()
        (job_output_path / "samples").mkdir()

        # Generate config YAML
        config_path = self._write_config(job_id, dataset_name, trigger_token, config)

        # Start training in background
        asyncio.create_task(self._run_training(job_id, config_path))

        return job

    def _write_config(
        self,
        job_id: str,
        dataset_name: str,
        trigger_token: str,
        config: TrainingConfig,
    ) -> Path:
        """Generate ai-toolkit config YAML.

        ai-toolkit expects this structure:
        - job: extension
        - config:
            - name: string
            - process: array of trainer configs

        Args:
            job_id: Job identifier.
            dataset_name: Dataset name.
            trigger_token: LoRA trigger token.
            config: Training configuration.

        Returns:
            Path to generated config file.
        """
        # Build the process config (the actual trainer settings)
        process_config = {
            "type": "sd_trainer",
            "compile": True,  # torch.compile optimization for Blackwell GPU
            "training_folder": f"/data/outputs/{job_id}",
            "device": "cuda:0",
            "trigger_word": trigger_token,
            "network": {
                "type": "lora",
                "linear": config.lora_rank,
                "linear_alpha": config.lora_rank,
            },
            "save": {
                "dtype": "float16",
                "save_every": config.checkpoint_every,
                "max_step_saves_to_keep": 5,
            },
            "datasets": [
                {
                    "folder_path": f"/data/datasets/{dataset_name}/images",
                    "caption_ext": "txt",
                    "caption_dropout_rate": 0.05,
                    "shuffle_tokens": False,
                    "cache_latents_to_disk": True,
                    "resolution": [config.image_size, config.image_size],
                }
            ],
            "train": {
                "batch_size": config.batch_size,
                "steps": config.steps,
                "gradient_accumulation_steps": 1,
                "train_unet": True,
                "train_text_encoder": False,
                "gradient_checkpointing": False,  # Disabled: 128GB VRAM available
                "noise_scheduler": "flowmatch",
                "optimizer": "adamw8bit",
                "lr": config.lr,
                "ema_config": {
                    "use_ema": True,
                    "ema_decay": 0.99,
                },
                "dtype": "bf16",
            },
            "model": {
                # Specify Z-Image architecture for ai-toolkit
                "arch": "zimage",
                # Z-Image Turbo from HuggingFace (diffusers format)
                "name_or_path": "Tongyi-MAI/Z-Image-Turbo",
                # Training adapter for de-distillation (enables proper LoRA training)
                "assistant_lora_path": "/weights/z-image-turbo/zimage_turbo_training_adapter_v1.safetensors",
                # Disable quantization - we have 128GB VRAM and quanto has issues with Blackwell
                "quantize": False,
            },
            "sample": {
                "sampler": "flowmatch",
                "sample_every": config.sample_every,
                "width": config.image_size,
                "height": config.image_size,
                "prompts": config.sample_prompts,
                "neg": "",
                "seed": 42,
                "walk_seed": True,
                "guidance_scale": 4,
                "sample_steps": 20,
            },
        }

        # Wrap in the expected ai-toolkit structure
        config_content = {
            "job": "extension",
            "config": {
                "name": f"{dataset_name}_{job_id}",
                "process": [process_config],
                "meta": {
                    "name": f"[lora] {dataset_name}",
                    "version": "1.0",
                },
            },
        }

        config_path = self.configs_path / f"{job_id}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_content, f, default_flow_style=False)

        logger.info(f"Wrote config to {config_path}")
        return config_path

    async def _run_training(self, job_id: str, config_path: Path) -> None:
        """Run training in Docker container.

        Args:
            job_id: Job identifier.
            config_path: Path to config YAML.
        """
        job = self.job_store.get_job(job_id)
        if not job:
            return

        if not self._docker:
            job.status = TrainingStatus.FAILED
            job.error_message = "Docker not available"
            self.job_store.save_job(job)
            return

        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.utcnow()
        self.job_store.save_job(job)

        container = None
        try:
            # Start container
            # Entrypoint already runs "python run.py", just pass the config path
            container = await asyncio.to_thread(
                self._docker.containers.run,
                DOCKER_IMAGE,
                command=[f"/data/configs/{job_id}.yaml"],
                volumes={
                    # Mount data directory containing datasets, outputs, configs
                    # Use host paths since we're spawning sibling containers
                    HOST_DATA_PATH: {"bind": "/data", "mode": "rw"},
                    # Mount model weights (read-only)
                    HOST_MODELS_PATH: {"bind": "/weights", "mode": "ro"},
                },
                device_requests=[
                    docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                ],
                shm_size="16g",
                ipc_mode="host",
                # Allow network for HuggingFace model downloads
                network_mode="bridge",
                detach=True,
                remove=False,
                name=f"lora-training-{job_id}",
            )

            self._active_containers[job_id] = container.id
            job.container_id = container.id
            self.job_store.save_job(job)

            logger.info(f"Started training container {container.id} for job {job_id}")

            # Monitor progress
            await self._monitor_progress(job_id, container)

            # Wait for completion
            result = await asyncio.to_thread(container.wait)

            if result["StatusCode"] == 0:
                job.status = TrainingStatus.COMPLETED
                logger.info(f"Training job {job_id} completed successfully")
            else:
                job.status = TrainingStatus.FAILED
                logs = await asyncio.to_thread(container.logs, tail=100)
                job.error_message = logs.decode()[-1000:]
                logger.error(f"Training job {job_id} failed: {job.error_message[:200]}")

        except ContainerError as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            logger.error(f"Container error for job {job_id}: {e}")
        except ImageNotFound:
            job.status = TrainingStatus.FAILED
            job.error_message = f"Docker image not found: {DOCKER_IMAGE}"
            logger.error(f"Image not found: {DOCKER_IMAGE}")
        except APIError as e:
            job.status = TrainingStatus.FAILED
            job.error_message = f"Docker API error: {e}"
            logger.error(f"Docker API error for job {job_id}: {e}")
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            logger.error(f"Training error for job {job_id}: {e}", exc_info=True)
        finally:
            job.completed_at = datetime.utcnow()
            self.job_store.save_job(job)
            self._active_containers.pop(job_id, None)

            # Cleanup container
            if container:
                try:
                    await asyncio.to_thread(container.remove)
                    logger.debug(f"Removed container for job {job_id}")
                except Exception:
                    pass

    async def _monitor_progress(self, job_id: str, container) -> None:
        """Monitor training progress by parsing logs.

        Args:
            job_id: Job identifier.
            container: Docker container object.
        """
        while True:
            await asyncio.sleep(5)

            # Check if container still running
            try:
                await asyncio.to_thread(container.reload)
                if container.status != "running":
                    break
            except Exception:
                break

            # Parse logs for progress
            try:
                logs = await asyncio.to_thread(container.logs, tail=50)
                self._parse_progress(job_id, logs.decode())
            except Exception as e:
                logger.debug(f"Failed to parse logs: {e}")

            # Scan for new outputs
            self._scan_outputs(job_id)

    def _parse_progress(self, job_id: str, logs: str) -> None:
        """Parse ai-toolkit output for step/loss updates.

        Args:
            job_id: Job identifier.
            logs: Recent log output.
        """
        job = self.job_store.get_job(job_id)
        if not job:
            return

        updated = False

        # Parse patterns like "Step 500/3000", "step: 500", or tqdm "| 71/100 ["
        step_patterns = [
            r"\|\s*(\d+)/(\d+)\s+\[",  # tqdm: | 71/100 [
            r"[Ss]tep[:\s]+(\d+)[/\s]+(\d+)",
            r"(\d+)/(\d+)\s+steps?",
            r"step[:\s]+(\d+)",
        ]

        for pattern in step_patterns:
            match = re.search(pattern, logs)
            if match:
                job.current_step = int(match.group(1))
                if len(match.groups()) > 1:
                    job.total_steps = int(match.group(2))
                updated = True
                break

        # Parse loss patterns like "Loss: 0.0234", "loss=0.023", or "loss: 2.891e-01"
        loss_patterns = [
            r"loss[:\s]+([0-9.e+-]+)",  # tqdm: loss: 2.891e-01
            r"[Ll]oss[:\s=]+([0-9.]+)",
            r"train_loss[:\s=]+([0-9.]+)",
        ]

        for pattern in loss_patterns:
            match = re.search(pattern, logs)
            if match:
                try:
                    job.latest_loss = float(match.group(1))
                    updated = True
                except ValueError:
                    pass
                break

        if updated:
            self.job_store.save_job(job)

    def _scan_outputs(self, job_id: str) -> None:
        """Scan for new checkpoints and sample images.

        Args:
            job_id: Job identifier.
        """
        job = self.job_store.get_job(job_id)
        if not job:
            return

        output_path = self.outputs_path / job_id
        updated = False

        # Scan checkpoints
        checkpoints_dir = output_path / "checkpoints"
        if checkpoints_dir.exists():
            checkpoints = sorted(checkpoints_dir.glob("*.safetensors"))
            new_checkpoints = [c.name for c in checkpoints]
            if new_checkpoints != job.checkpoints:
                job.checkpoints = new_checkpoints
                updated = True

        # Scan samples
        samples_dir = output_path / "samples"
        if samples_dir.exists():
            samples = sorted(samples_dir.glob("*.png"))
            new_samples = [str(s) for s in samples]
            if new_samples != job.sample_images:
                job.sample_images = new_samples
                updated = True

        if updated:
            self.job_store.save_job(job)

    async def stop_training(self, job_id: str) -> TrainingJob:
        """Stop a running training job.

        Args:
            job_id: Job identifier.

        Returns:
            Updated job.

        Raises:
            ValueError: If job not found.
        """
        job = self.job_store.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        if job.container_id and self._docker:
            try:
                container = self._docker.containers.get(job.container_id)
                await asyncio.to_thread(container.stop, timeout=10)
                logger.info(f"Stopped container for job {job_id}")
            except Exception as e:
                logger.warning(f"Failed to stop container: {e}")

        job.status = TrainingStatus.STOPPED
        job.completed_at = datetime.utcnow()
        self.job_store.save_job(job)

        return job

    def promote_checkpoint(
        self,
        job_id: str,
        checkpoint_name: str,
        output_name: str,
    ) -> str:
        """Copy checkpoint to loras directory.

        Args:
            job_id: Job identifier.
            checkpoint_name: Checkpoint filename.
            output_name: Output LoRA name (without extension).

        Returns:
            Path to promoted LoRA file.

        Raises:
            ValueError: If job or checkpoint not found.
        """
        job = self.job_store.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        src = self.outputs_path / job_id / "checkpoints" / checkpoint_name
        if not src.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_name}")

        # Ensure output name has extension
        if not output_name.endswith(".safetensors"):
            output_name = f"{output_name}.safetensors"

        dst = self.loras_path / output_name
        self.loras_path.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src, dst)
        logger.info(f"Promoted checkpoint to {dst}")

        return str(dst)

    def get_active_jobs(self) -> list[str]:
        """Get list of actively running job IDs.

        Returns:
            List of job IDs with running containers.
        """
        return list(self._active_containers.keys())
