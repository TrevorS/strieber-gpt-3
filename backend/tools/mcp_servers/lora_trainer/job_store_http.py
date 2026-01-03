# ABOUTME: HTTP-based job store using responses-api generic storage.
# ABOUTME: Provides distributed job persistence that survives container restarts.

import logging
from typing import List, Optional

import httpx

from lora_trainer.job_store_base import JobStoreBase
from lora_trainer.models import TrainingJob, TrainingStatus


logger = logging.getLogger(__name__)


class HttpJobStore(JobStoreBase):
    """Job store using responses-api HTTP storage backend.

    Persists jobs to /v1/storage/lora_jobs/{job_id} endpoint.
    Survives container restarts as long as responses-api is up.
    """

    def __init__(self, base_url: str = "http://responses-api:9150"):
        """Initialize HTTP job store.

        Args:
            base_url: Base URL of responses-api (default: http://responses-api:9150).
        """
        self.base_url = base_url.rstrip("/")
        self.collection = "lora_jobs"
        self.client = httpx.Client(timeout=10.0)

    def _storage_url(self, job_id: Optional[str] = None) -> str:
        """Build storage API URL.

        Args:
            job_id: Optional job ID for specific resource.

        Returns:
            Full URL for storage endpoint.
        """
        url = f"{self.base_url}/v1/storage/{self.collection}"
        if job_id:
            url = f"{url}/{job_id}"
        return url

    def save_job(self, job: TrainingJob) -> None:
        """Save a job to the store.

        Args:
            job: Job to save.
        """
        try:
            response = self.client.post(
                self._storage_url(),
                json={
                    "id": job.job_id,
                    "data": job.model_dump(mode="json"),
                },
            )
            response.raise_for_status()
            logger.debug(f"Saved job {job.job_id} to HTTP storage")
        except httpx.HTTPError as e:
            logger.warning(f"Failed to save job {job.job_id}: {e}")

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a job by ID.

        Args:
            job_id: Job ID to retrieve.

        Returns:
            Job if found, None otherwise.
        """
        try:
            response = self.client.get(self._storage_url(job_id))
            response.raise_for_status()
            data = response.json()
            return TrainingJob(**data.get("data", data))
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug(f"Job {job_id} not found")
                return None
            logger.warning(f"Failed to get job {job_id}: {e}")
            return None
        except httpx.HTTPError as e:
            logger.warning(f"Failed to get job {job_id}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to parse job {job_id}: {e}")
            return None

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
        try:
            params = {}
            if status:
                params["status"] = status.value

            response = self.client.get(self._storage_url(), params=params)
            response.raise_for_status()
            data = response.json()

            # Handle wrapped response format: {"collection": "...", "records": [...]}
            items = data.get("records", data) if isinstance(data, dict) else data

            result = []
            for item in items:
                try:
                    # Handle both {id, data} and flat formats
                    job_data = item.get("data", item)
                    job = TrainingJob(**job_data)

                    # Client-side status filtering if API doesn't support it
                    if status is None or job.status == status:
                        result.append(job)
                except Exception as e:
                    logger.warning(f"Failed to parse job from list: {e}")

            # Sort by started_at (most recent first)
            # Use timestamp to avoid timezone-aware vs naive datetime comparison
            def sort_key(j: TrainingJob) -> float:
                if j.started_at:
                    return j.started_at.timestamp()
                return 0.0

            result.sort(key=sort_key, reverse=True)

            return result[:limit]
        except httpx.HTTPError as e:
            logger.warning(f"Failed to list jobs: {e}")
            return []

    def delete_job(self, job_id: str) -> bool:
        """Delete a job from the store.

        Args:
            job_id: Job ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        try:
            response = self.client.delete(self._storage_url(job_id))
            response.raise_for_status()
            logger.info(f"Deleted job {job_id}")
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug(f"Job {job_id} not found for deletion")
                return False
            logger.warning(f"Failed to delete job {job_id}: {e}")
            return False
        except httpx.HTTPError as e:
            logger.warning(f"Failed to delete job {job_id}: {e}")
            return False

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
