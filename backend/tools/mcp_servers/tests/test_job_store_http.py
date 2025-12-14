# ABOUTME: Unit tests for HttpJobStore module.
# Tests HTTP-based job persistence via responses-api storage.

from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from lora_trainer.job_store_http import HttpJobStore
from lora_trainer.models import TrainingConfig, TrainingJob, TrainingStatus


class TestHttpJobStore:
    """Tests for HttpJobStore."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock httpx Client."""
        return Mock(spec=httpx.Client)

    @pytest.fixture
    def store(self, mock_client):
        """Create an HttpJobStore with mocked client."""
        with patch("httpx.Client", return_value=mock_client):
            store = HttpJobStore(base_url="http://test-api:8000")
        store.client = mock_client
        return store

    @pytest.fixture
    def sample_job(self):
        """Create a sample training job."""
        config = TrainingConfig(dataset="test_dataset", steps=100)
        return TrainingJob(
            job_id="job_123",
            dataset_name="test_dataset",
            trigger_token="ohwx",
            config=config,
        )

    def test_storage_url_without_job_id(self, store):
        """Test storage URL generation without job ID."""
        url = store._storage_url()
        assert url == "http://test-api:8000/v1/storage/lora_jobs"

    def test_storage_url_with_job_id(self, store):
        """Test storage URL generation with job ID."""
        url = store._storage_url("job_123")
        assert url == "http://test-api:8000/v1/storage/lora_jobs/job_123"

    def test_save_job_success(self, store, mock_client, sample_job):
        """Test successful job save."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response

        store.save_job(sample_job)

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://test-api:8000/v1/storage/lora_jobs"
        assert call_args[1]["json"]["id"] == "job_123"
        assert call_args[1]["json"]["data"]["job_id"] == "job_123"

    def test_save_job_http_error_logged(self, store, mock_client, sample_job, caplog):
        """Test that HTTP errors during save are logged."""
        mock_client.post.side_effect = httpx.HTTPError("Connection failed")

        # Should not raise, just log
        store.save_job(sample_job)

        assert "Failed to save job" in caplog.text

    def test_get_job_success(self, store, mock_client, sample_job):
        """Test successful job retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "job_123",
            "data": sample_job.model_dump(mode="json"),
        }
        mock_client.get.return_value = mock_response

        job = store.get_job("job_123")

        assert job is not None
        assert job.job_id == "job_123"
        assert job.dataset_name == "test_dataset"
        mock_client.get.assert_called_once_with(
            "http://test-api:8000/v1/storage/lora_jobs/job_123"
        )

    def test_get_job_not_found(self, store, mock_client):
        """Test job retrieval when not found."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not found", request=Mock(), response=mock_response
        )
        mock_client.get.return_value = mock_response

        job = store.get_job("nonexistent")

        assert job is None

    def test_get_job_http_error(self, store, mock_client, caplog):
        """Test job retrieval with HTTP error."""
        mock_client.get.side_effect = httpx.HTTPError("Connection failed")

        job = store.get_job("job_123")

        assert job is None
        assert "Failed to get job" in caplog.text

    def test_get_job_parse_error(self, store, mock_client, caplog):
        """Test job retrieval with invalid response format."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "data"}
        mock_client.get.return_value = mock_response

        job = store.get_job("job_123")

        assert job is None
        assert "Failed to parse job" in caplog.text

    def test_list_jobs_success(self, store, mock_client, sample_job):
        """Test listing jobs."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "job_123", "data": sample_job.model_dump(mode="json")},
        ]
        mock_client.get.return_value = mock_response

        jobs = store.list_jobs()

        assert len(jobs) == 1
        assert jobs[0].job_id == "job_123"

    def test_list_jobs_with_status_filter(self, store, mock_client, sample_job):
        """Test listing jobs with status filter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "job_123", "data": sample_job.model_dump(mode="json")},
        ]
        mock_client.get.return_value = mock_response

        jobs = store.list_jobs(status=TrainingStatus.PENDING)

        assert len(jobs) == 1
        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        assert call_args[1]["params"]["status"] == "pending"

    def test_list_jobs_empty(self, store, mock_client):
        """Test listing jobs when none exist."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_client.get.return_value = mock_response

        jobs = store.list_jobs()

        assert jobs == []

    def test_list_jobs_http_error(self, store, mock_client, caplog):
        """Test listing jobs with HTTP error."""
        mock_client.get.side_effect = httpx.HTTPError("Connection failed")

        jobs = store.list_jobs()

        assert jobs == []
        assert "Failed to list jobs" in caplog.text

    def test_list_jobs_filters_invalid_entries(self, store, mock_client, sample_job):
        """Test that invalid entries are filtered out during listing."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "job_123", "data": sample_job.model_dump(mode="json")},
            {"id": "invalid", "data": {"bad": "data"}},  # Invalid entry
        ]
        mock_client.get.return_value = mock_response

        jobs = store.list_jobs()

        # Should only have valid job
        assert len(jobs) == 1
        assert jobs[0].job_id == "job_123"

    def test_delete_job_success(self, store, mock_client):
        """Test successful job deletion."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.delete.return_value = mock_response

        result = store.delete_job("job_123")

        assert result is True
        mock_client.delete.assert_called_once_with(
            "http://test-api:8000/v1/storage/lora_jobs/job_123"
        )

    def test_delete_job_not_found(self, store, mock_client):
        """Test deletion of non-existent job."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not found", request=Mock(), response=mock_response
        )
        mock_client.delete.return_value = mock_response

        result = store.delete_job("nonexistent")

        assert result is False

    def test_delete_job_http_error(self, store, mock_client, caplog):
        """Test deletion with HTTP error."""
        mock_client.delete.side_effect = httpx.HTTPError("Connection failed")

        result = store.delete_job("job_123")

        assert result is False
        assert "Failed to delete job" in caplog.text

    def test_update_job_status(self, store, mock_client, sample_job):
        """Test updating job status."""
        # Mock get_job
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "data": sample_job.model_dump(mode="json")
        }

        # Mock save_job
        mock_post_response = Mock()
        mock_post_response.status_code = 200

        mock_client.get.return_value = mock_get_response
        mock_client.post.return_value = mock_post_response

        updated = store.update_job_status("job_123", TrainingStatus.RUNNING)

        assert updated is not None
        assert updated.status == TrainingStatus.RUNNING

    def test_update_job_status_with_error(self, store, mock_client, sample_job):
        """Test updating job status with error message."""
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "data": sample_job.model_dump(mode="json")
        }

        mock_post_response = Mock()
        mock_post_response.status_code = 200

        mock_client.get.return_value = mock_get_response
        mock_client.post.return_value = mock_post_response

        updated = store.update_job_status(
            "job_123", TrainingStatus.FAILED, error_message="Out of memory"
        )

        assert updated is not None
        assert updated.status == TrainingStatus.FAILED
        assert updated.error_message == "Out of memory"

    def test_update_job_status_not_found(self, store, mock_client):
        """Test updating status of non-existent job."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not found", request=Mock(), response=mock_response
        )
        mock_client.get.return_value = mock_response

        updated = store.update_job_status("nonexistent", TrainingStatus.RUNNING)

        assert updated is None

    def test_update_job_progress(self, store, mock_client, sample_job):
        """Test updating job progress."""
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "data": sample_job.model_dump(mode="json")
        }

        mock_post_response = Mock()
        mock_post_response.status_code = 200

        mock_client.get.return_value = mock_get_response
        mock_client.post.return_value = mock_post_response

        updated = store.update_job_progress(
            "job_123", current_step=50, latest_loss=0.025
        )

        assert updated is not None
        assert updated.current_step == 50
        assert updated.latest_loss == 0.025

    def test_update_job_progress_without_loss(self, store, mock_client, sample_job):
        """Test updating job progress without loss value."""
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "data": sample_job.model_dump(mode="json")
        }

        mock_post_response = Mock()
        mock_post_response.status_code = 200

        mock_client.get.return_value = mock_get_response
        mock_client.post.return_value = mock_post_response

        updated = store.update_job_progress("job_123", current_step=25)

        assert updated is not None
        assert updated.current_step == 25
        assert updated.latest_loss is None

    def test_add_checkpoint(self, store, mock_client, sample_job):
        """Test adding a checkpoint to a job."""
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "data": sample_job.model_dump(mode="json")
        }

        mock_post_response = Mock()
        mock_post_response.status_code = 200

        mock_client.get.return_value = mock_get_response
        mock_client.post.return_value = mock_post_response

        updated = store.add_checkpoint("job_123", "step_500.safetensors")

        assert updated is not None
        assert "step_500.safetensors" in updated.checkpoints

    def test_add_checkpoint_duplicate_ignored(self, store, mock_client, sample_job):
        """Test that duplicate checkpoints are not added."""
        # Set up job with existing checkpoint
        job_data = sample_job.model_dump(mode="json")
        job_data["checkpoints"] = ["step_500.safetensors"]

        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"data": job_data}

        mock_client.get.return_value = mock_get_response

        updated = store.add_checkpoint("job_123", "step_500.safetensors")

        assert updated is not None
        assert updated.checkpoints.count("step_500.safetensors") == 1
        # save_job should not be called since no change
        mock_client.post.assert_not_called()

    def test_add_sample_image(self, store, mock_client, sample_job):
        """Test adding a sample image to a job."""
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "data": sample_job.model_dump(mode="json")
        }

        mock_post_response = Mock()
        mock_post_response.status_code = 200

        mock_client.get.return_value = mock_get_response
        mock_client.post.return_value = mock_post_response

        updated = store.add_sample_image("job_123", "/path/to/sample.png")

        assert updated is not None
        assert "/path/to/sample.png" in updated.sample_images

    def test_add_sample_image_not_found(self, store, mock_client):
        """Test adding sample image to non-existent job."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not found", request=Mock(), response=mock_response
        )
        mock_client.get.return_value = mock_response

        updated = store.add_sample_image("nonexistent", "/path/to/sample.png")

        assert updated is None


class TestHttpJobStoreIntegration:
    """Integration-style tests for HttpJobStore (still mocked but testing flows)."""

    def test_full_job_lifecycle(self):
        """Test a complete job lifecycle through the store."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            store = HttpJobStore(base_url="http://test-api:8000")

            # Create job
            config = TrainingConfig(dataset="lifecycle_test", steps=1000)
            job = TrainingJob(
                job_id="lifecycle_job",
                dataset_name="lifecycle_test",
                trigger_token="ohwx",
                config=config,
            )

            # Mock responses for the lifecycle
            mock_post_response = Mock()
            mock_post_response.status_code = 200
            mock_client.post.return_value = mock_post_response

            # Save job
            store.save_job(job)
            assert mock_client.post.called

            # Mock get for status update
            mock_get_response = Mock()
            mock_get_response.status_code = 200
            mock_get_response.json.return_value = {"data": job.model_dump(mode="json")}
            mock_client.get.return_value = mock_get_response

            # Update status to running
            updated = store.update_job_status("lifecycle_job", TrainingStatus.RUNNING)
            assert updated.status == TrainingStatus.RUNNING

            # Update progress
            mock_client.get.return_value.json.return_value = {
                "data": updated.model_dump(mode="json")
            }
            updated = store.update_job_progress(
                "lifecycle_job", current_step=500, latest_loss=0.05
            )
            assert updated.current_step == 500

            # Add checkpoint
            mock_client.get.return_value.json.return_value = {
                "data": updated.model_dump(mode="json")
            }
            updated = store.add_checkpoint("lifecycle_job", "step_500.safetensors")
            assert "step_500.safetensors" in updated.checkpoints

            # Mark completed
            mock_client.get.return_value.json.return_value = {
                "data": updated.model_dump(mode="json")
            }
            updated = store.update_job_status("lifecycle_job", TrainingStatus.COMPLETED)
            assert updated.status == TrainingStatus.COMPLETED

    def test_default_base_url(self):
        """Test default base URL when not specified."""
        with patch("httpx.Client"):
            store = HttpJobStore()
            assert store.base_url == "http://responses-api:9150"
