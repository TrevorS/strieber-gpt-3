# ABOUTME: Job persistence for LoRA training.
# ABOUTME: JSON file-based storage implementation.

import json
import logging
from pathlib import Path
from threading import Lock
from typing import List, Optional

from lora_trainer.job_store_base import JobStoreBase
from lora_trainer.models import TrainingJob, TrainingStatus


logger = logging.getLogger(__name__)


class JsonJobStore(JobStoreBase):
    """JSON file-based storage for training jobs."""

    def __init__(self, jobs_file: Path):
        """Initialize job store.

        Args:
            jobs_file: Path to jobs.json file.
        """
        self.jobs_file = Path(jobs_file)
        self.jobs_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

        # Initialize file if not exists
        if not self.jobs_file.exists():
            self._save_all({})

    def _load_all(self) -> dict:
        """Load all jobs from disk."""
        try:
            with open(self.jobs_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_all(self, jobs: dict) -> None:
        """Save all jobs to disk."""
        with open(self.jobs_file, "w") as f:
            json.dump(jobs, f, indent=2, default=str)

    def save_job(self, job: TrainingJob) -> None:
        """Save a job to the store.

        Args:
            job: Job to save.
        """
        with self._lock:
            jobs = self._load_all()
            jobs[job.job_id] = job.model_dump(mode="json")
            self._save_all(jobs)

        logger.debug(f"Saved job {job.job_id}")

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a job by ID.

        Args:
            job_id: Job ID to retrieve.

        Returns:
            Job if found, None otherwise.
        """
        with self._lock:
            jobs = self._load_all()

        if job_id not in jobs:
            return None

        return TrainingJob(**jobs[job_id])

    def list_jobs(
        self,
        status: Optional[TrainingStatus] = None,
        limit: int = 100,
    ) -> List[TrainingJob]:
        """List jobs, optionally filtered by status.

        Args:
            status: Filter by status (optional).
            limit: Maximum number of jobs to return.

        Returns:
            List of jobs, most recent first.
        """
        with self._lock:
            jobs = self._load_all()

        result = []
        for job_data in jobs.values():
            try:
                job = TrainingJob(**job_data)
                if status is None or job.status == status:
                    result.append(job)
            except Exception as e:
                logger.warning(f"Failed to load job: {e}")

        # Sort by created time (most recent first)
        # Use timestamp to avoid timezone-aware vs naive datetime comparison issues
        def sort_key(j: TrainingJob) -> float:
            if j.started_at:
                # Convert to timestamp (handles both aware and naive datetimes)
                return j.started_at.timestamp()
            return 0.0

        result.sort(key=sort_key, reverse=True)

        return result[:limit]

    def delete_job(self, job_id: str) -> bool:
        """Delete a job from the store.

        Args:
            job_id: Job ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            jobs = self._load_all()
            if job_id not in jobs:
                return False
            del jobs[job_id]
            self._save_all(jobs)

        logger.info(f"Deleted job {job_id}")
        return True

    def update_job_status(
        self,
        job_id: str,
        status: TrainingStatus,
        error_message: Optional[str] = None,
    ) -> Optional[TrainingJob]:
        """Update job status.

        Args:
            job_id: Job ID.
            status: New status.
            error_message: Optional error message for failed jobs.

        Returns:
            Updated job if found, None otherwise.
        """
        job = self.get_job(job_id)
        if not job:
            return None

        job.status = status
        if error_message:
            job.error_message = error_message

        self.save_job(job)
        return job

    def update_job_progress(
        self,
        job_id: str,
        current_step: int,
        latest_loss: Optional[float] = None,
    ) -> Optional[TrainingJob]:
        """Update job training progress.

        Args:
            job_id: Job ID.
            current_step: Current training step.
            latest_loss: Latest loss value.

        Returns:
            Updated job if found, None otherwise.
        """
        job = self.get_job(job_id)
        if not job:
            return None

        job.current_step = current_step
        if latest_loss is not None:
            job.latest_loss = latest_loss

        self.save_job(job)
        return job

    def add_checkpoint(
        self, job_id: str, checkpoint_name: str
    ) -> Optional[TrainingJob]:
        """Add a checkpoint to job.

        Args:
            job_id: Job ID.
            checkpoint_name: Checkpoint filename.

        Returns:
            Updated job if found, None otherwise.
        """
        job = self.get_job(job_id)
        if not job:
            return None

        if checkpoint_name not in job.checkpoints:
            job.checkpoints.append(checkpoint_name)
            self.save_job(job)

        return job

    def add_sample_image(self, job_id: str, image_path: str) -> Optional[TrainingJob]:
        """Add a sample image to job.

        Args:
            job_id: Job ID.
            image_path: Path to sample image.

        Returns:
            Updated job if found, None otherwise.
        """
        job = self.get_job(job_id)
        if not job:
            return None

        if image_path not in job.sample_images:
            job.sample_images.append(image_path)
            self.save_job(job)

        return job
