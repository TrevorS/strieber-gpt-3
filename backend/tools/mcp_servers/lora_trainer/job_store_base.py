# ABOUTME: Abstract base class for job storage backends.
# ABOUTME: Enables swapping between JSON, HTTP, or other storage implementations.

from abc import ABC, abstractmethod
from typing import List, Optional

from lora_trainer.models import TrainingJob, TrainingStatus


class JobStoreBase(ABC):
    """Abstract base for job storage backends."""

    @abstractmethod
    def save_job(self, job: TrainingJob) -> None:
        """Save a job to the store.

        Args:
            job: Job to save.
        """
        ...

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a job by ID.

        Args:
            job_id: Job ID to retrieve.

        Returns:
            Job if found, None otherwise.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def delete_job(self, job_id: str) -> bool:
        """Delete a job from the store.

        Args:
            job_id: Job ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def add_sample_image(self, job_id: str, image_path: str) -> Optional[TrainingJob]:
        """Add a sample image to job.

        Args:
            job_id: Job ID.
            image_path: Path to sample image.

        Returns:
            Updated job if found, None otherwise.
        """
        ...
