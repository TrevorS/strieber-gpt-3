# ABOUTME: Training job execution via Docker.
# Manages ai-toolkit container lifecycle and monitors training progress.

import asyncio
import logging
import re
import shutil
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import docker
import os
import yaml
from docker.errors import APIError, ContainerError, ImageNotFound

from lora_trainer.job_store_base import JobStoreBase
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
        job_store: JobStoreBase,
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

        # Create output directory (ai-toolkit creates its own structure inside)
        job_output_path = self.outputs_path / job_id
        job_output_path.mkdir(parents=True, exist_ok=True)

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
                "gradient_checkpointing": True,  # Enabled: other services use ~91GB VRAM
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
            # Use SQLite logging for reliable progress monitoring
            # Writes to {training_folder}/{name}/loss_log.db
            "logging": {
                "log_every": 1,  # Log every step for real-time updates
                "use_ui_logger": True,
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
        job.started_at = datetime.now(UTC)
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
            # Re-fetch job to get latest state (samples, checkpoints, progress)
            # that was updated during monitoring
            latest_job = self.job_store.get_job(job_id)
            if latest_job:
                latest_job.status = job.status
                latest_job.completed_at = datetime.now(UTC)
                if job.error_message:
                    latest_job.error_message = job.error_message
                # Do one final scan for outputs
                self._scan_outputs(job_id)
                # Re-fetch again after scan
                latest_job = self.job_store.get_job(job_id)
                if latest_job:
                    latest_job.status = job.status
                    latest_job.completed_at = datetime.now(UTC)
                    if job.error_message:
                        latest_job.error_message = job.error_message
                    self.job_store.save_job(latest_job)
            else:
                # Fallback if job somehow doesn't exist
                job.completed_at = datetime.now(UTC)
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

            # Read progress: prefer SQLite, fallback to log parsing
            try:
                # Try SQLite first (reliable, structured data)
                sqlite_success = self._read_progress_from_sqlite(job_id)

                # Fallback to log parsing if SQLite not available yet
                if not sqlite_success:
                    logs = await asyncio.to_thread(container.logs, tail=50)
                    self._parse_progress(job_id, logs.decode())
            except Exception as e:
                logger.debug(f"Failed to read progress: {e}")

            # Scan for new outputs
            self._scan_outputs(job_id)

    def _read_progress_from_sqlite(self, job_id: str) -> bool:
        """Read training progress from ai-toolkit's SQLite log.

        The UILogger writes to loss_log.db with tables:
        - steps: (step INTEGER PRIMARY KEY, wall_time REAL)
        - metrics: (step INTEGER, key TEXT, value_real REAL, value_text TEXT)

        Args:
            job_id: Job identifier.

        Returns:
            True if progress was updated from SQLite, False otherwise.
        """
        job = self.job_store.get_job(job_id)
        if not job:
            logger.info(f"SQLite progress: job {job_id} not found in store")
            return False

        # ai-toolkit writes to: {training_folder}/{name}/loss_log.db
        db_path = (
            self.outputs_path
            / job_id
            / f"{job.dataset_name}_{job_id}"
            / "loss_log.db"
        )

        if not db_path.exists():
            logger.debug(f"SQLite progress: {db_path} not found")
            return False

        logger.info(f"SQLite progress: reading from {db_path}")

        try:
            # Use timeout and read-only mode for safe concurrent access
            conn = sqlite3.connect(
                f"file:{db_path}?mode=ro",
                uri=True,
                timeout=5.0,
            )
            cursor = conn.cursor()

            # Get the latest step
            cursor.execute("SELECT MAX(step) FROM steps")
            result = cursor.fetchone()
            if not result or result[0] is None:
                conn.close()
                return False

            # ai-toolkit uses 0-indexed steps, convert to 1-indexed for display
            current_step = result[0] + 1

            # Get the latest loss value (ai-toolkit uses 'loss/loss' key)
            cursor.execute(
                "SELECT value_real FROM metrics WHERE key IN ('loss', 'loss/loss') "
                "ORDER BY step DESC LIMIT 1"
            )
            result = cursor.fetchone()
            latest_loss = result[0] if result else None

            conn.close()

            # Update job
            updated = False
            if current_step != job.current_step:
                job.current_step = current_step
                updated = True

            if latest_loss is not None and latest_loss != job.latest_loss:
                job.latest_loss = latest_loss
                updated = True

            if updated:
                self.job_store.save_job(job)
                logger.info(
                    f"SQLite progress for {job_id}: step={current_step}, loss={latest_loss}"
                )

            return True

        except sqlite3.Error as e:
            logger.debug(f"SQLite read error for {job_id}: {e}")
            return False
        except Exception as e:
            logger.debug(f"Unexpected error reading SQLite for {job_id}: {e}")
            return False

    def _parse_progress(self, job_id: str, logs: str) -> None:
        """Parse ai-toolkit output for step/loss updates (fallback for tqdm).

        This is the fallback method when SQLite logging is not available.

        Args:
            job_id: Job identifier.
            logs: Recent log output.
        """
        job = self.job_store.get_job(job_id)
        if not job:
            return

        updated = False
        expected_steps = job.config.steps  # Original config value (reliable)

        # ai-toolkit training progress format:
        # constance_wu_f6242d29:  28%|██▊       | 141/500 [20:26<51:56, ...]
        #
        # We need to distinguish this from caching/loading progress like:
        # Caching latents to disk: 100%|██████████| 5/5 [...]
        # Loading checkpoint shards: 100%|██████████| 3/3 [...]
        #
        # Strategy: Look for matches where total equals expected_steps from config.
        # This is reliable because config.steps is set at job creation.

        step_patterns = [
            # ai-toolkit training: jobname: XX%|bars| current/total [
            r"[a-zA-Z0-9_]+:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s+\[",
            # Generic tqdm: | 71/100 [
            r"\|\s*(\d+)/(\d+)\s+\[",
            # Step X/Y format
            r"[Ss]tep[:\s]+(\d+)[/\s]+(\d+)",
            r"(\d+)/(\d+)\s+steps?",
        ]

        best_match = None
        best_is_training = False

        for pattern in step_patterns:
            for match in re.finditer(pattern, logs):
                current = int(match.group(1))
                total = int(match.group(2))

                # Skip obviously non-training matches (checkpoints, shards, etc)
                if total <= 10 and total != expected_steps:
                    continue

                # Prefer matches where total matches expected training steps
                # Keep updating best_match to get the LAST (most recent) match
                if total == expected_steps:
                    best_match = (current, total)
                    best_is_training = True
                    # Don't break - continue to find the last match in logs
                elif not best_is_training and total > 10:
                    # Fallback: keep any substantial match
                    best_match = (current, total)

            # If we found training matches, don't try other patterns
            if best_is_training:
                break

        if best_match:
            job.current_step = best_match[0]
            # Always use total from match if it equals expected_steps
            # This corrects any wrong total_steps values from earlier bad parses
            if best_match[1] == expected_steps:
                job.total_steps = expected_steps
            elif job.total_steps != expected_steps and best_match[1] > job.total_steps:
                # Fallback: update if the match has a larger total (likely training)
                job.total_steps = best_match[1]
            updated = True

        # Also check single-step pattern as last resort (no total in match)
        if not best_match:
            single_step = re.search(r"step[:\s]+(\d+)", logs, re.IGNORECASE)
            if single_step:
                job.current_step = int(single_step.group(1))
                updated = True

        # Parse loss from tqdm output: loss: 2.891e-01
        # Look for the LAST occurrence as it's the most recent
        loss_matches = list(re.finditer(r"loss:\s*([0-9.e+-]+)", logs, re.IGNORECASE))
        if loss_matches:
            try:
                job.latest_loss = float(loss_matches[-1].group(1))
                updated = True
            except ValueError:
                pass

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

        # ai-toolkit outputs to: {training_folder}/{dataset_name}_{job_id}/
        job_output_dir = self.outputs_path / job_id / f"{job.dataset_name}_{job_id}"
        updated = False

        # Scan checkpoints (in job output directory, not separate checkpoints dir)
        if job_output_dir.exists():
            checkpoints = sorted(job_output_dir.glob("*.safetensors"))
            new_checkpoints = [c.name for c in checkpoints]
            if new_checkpoints != job.checkpoints:
                job.checkpoints = new_checkpoints
                updated = True

        # Scan samples (ai-toolkit outputs JPGs in samples subdirectory)
        samples_dir = job_output_dir / "samples"
        if samples_dir.exists():
            samples = sorted(
                list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png"))
            )
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
        job.completed_at = datetime.now(UTC)
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

        # ai-toolkit outputs to: {training_folder}/{dataset_name}_{job_id}/
        job_output_dir = self.outputs_path / job_id / f"{job.dataset_name}_{job_id}"
        src = job_output_dir / checkpoint_name
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

    async def refresh_job_progress(self, job_id: str) -> Optional[TrainingJob]:
        """Manually refresh job progress by parsing container logs.

        Useful when monitoring task was interrupted (e.g., server restart).

        Args:
            job_id: Job identifier.

        Returns:
            Updated job if found and container exists, None otherwise.
        """
        job = self.job_store.get_job(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found")
            return None

        if not self._docker:
            logger.warning("Docker not available for refresh")
            return None

        # Try to find the container
        container_name = f"lora-training-{job_id}"
        try:
            container = await asyncio.to_thread(
                self._docker.containers.get, container_name
            )
        except Exception as e:
            logger.debug(f"Container {container_name} not found: {e}")
            return None

        # Update container_id if not set
        if not job.container_id:
            job.container_id = container.id
            self.job_store.save_job(job)

        # Check container status
        await asyncio.to_thread(container.reload)
        if container.status == "running":
            # Read progress: prefer SQLite, fallback to log parsing
            sqlite_success = self._read_progress_from_sqlite(job_id)
            if not sqlite_success:
                logs = await asyncio.to_thread(container.logs, tail=100)
                self._parse_progress(job_id, logs.decode())

            # Scan for outputs
            self._scan_outputs(job_id)

            # Ensure status is running
            job = self.job_store.get_job(job_id)
            if job and job.status != TrainingStatus.RUNNING:
                job.status = TrainingStatus.RUNNING
                job.completed_at = None
                self.job_store.save_job(job)

        return self.job_store.get_job(job_id)

    async def resume_orphaned_monitoring(self, job_id: str) -> bool:
        """Resume monitoring for an orphaned training container.

        Call this when a job shows as "running" but has no active monitoring task.

        Args:
            job_id: Job identifier.

        Returns:
            True if monitoring resumed, False otherwise.
        """
        job = self.job_store.get_job(job_id)
        if not job or job.status != TrainingStatus.RUNNING:
            return False

        if job_id in self._active_containers:
            logger.info(f"Job {job_id} already being monitored")
            return True

        if not self._docker:
            logger.warning("Docker not available")
            return False

        container_name = f"lora-training-{job_id}"
        try:
            container = await asyncio.to_thread(
                self._docker.containers.get, container_name
            )
            await asyncio.to_thread(container.reload)

            if container.status != "running":
                logger.info(f"Container {container_name} is {container.status}")
                return False

            # Resume monitoring
            self._active_containers[job_id] = container.id
            logger.info(f"Resuming monitoring for job {job_id}")

            # Start monitoring task (will complete when container finishes)
            asyncio.create_task(self._resume_monitor_task(job_id, container))
            return True

        except Exception as e:
            logger.warning(f"Failed to resume monitoring for {job_id}: {e}")
            return False

    async def _resume_monitor_task(self, job_id: str, container) -> None:
        """Monitor a resumed container until completion.

        Args:
            job_id: Job identifier.
            container: Docker container object.
        """
        job = self.job_store.get_job(job_id)
        if not job:
            return

        try:
            # Monitor progress
            await self._monitor_progress(job_id, container)

            # Wait for completion
            result = await asyncio.to_thread(container.wait)

            job = self.job_store.get_job(job_id)
            if not job:
                return

            if result["StatusCode"] == 0:
                job.status = TrainingStatus.COMPLETED
                logger.info(f"Training job {job_id} completed successfully")
            else:
                job.status = TrainingStatus.FAILED
                logs = await asyncio.to_thread(container.logs, tail=100)
                job.error_message = logs.decode()[-1000:]
                logger.error(f"Training job {job_id} failed")

        except Exception as e:
            job = self.job_store.get_job(job_id)
            if job:
                job.status = TrainingStatus.FAILED
                job.error_message = str(e)
                logger.error(f"Resume monitor error for {job_id}: {e}", exc_info=True)
        finally:
            job = self.job_store.get_job(job_id)
            if job:
                job.completed_at = datetime.now(UTC)
                self.job_store.save_job(job)
            self._active_containers.pop(job_id, None)

            # Cleanup container
            try:
                await asyncio.to_thread(container.remove)
            except Exception:
                pass
