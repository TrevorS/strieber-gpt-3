# ABOUTME: Unit tests for LoRA trainer MCP server components.
# Tests models, dataset manager, and job store functionality.

import pytest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from lora_trainer.models import (
    DatasetMetadata,
    LoRAType,
    TrainingConfig,
    TrainingJob,
    TrainingStatus,
)
from lora_trainer.dataset_manager import DatasetManager
from lora_trainer.job_store_json import JsonJobStore as JobStore


class TestModels:
    """Tests for Pydantic models."""

    def test_dataset_metadata_defaults(self):
        """Test DatasetMetadata default values."""
        meta = DatasetMetadata(
            name="test",
            trigger_token="ohwx",
            lora_type=LoRAType.CHARACTER,
        )
        assert meta.image_count == 0
        assert meta.has_captions is False
        assert meta.description is None
        assert meta.created_at is not None

    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig(dataset="test")
        assert config.steps == 3000
        assert config.lora_rank == 8
        assert config.lr == 0.0001
        assert config.batch_size == 1
        assert config.image_size == 1024
        assert config.checkpoint_every == 500
        assert config.sample_every == 250

    def test_training_job_defaults(self):
        """Test TrainingJob default values."""
        config = TrainingConfig(dataset="test")
        job = TrainingJob(
            job_id="abc123",
            dataset_name="test",
            trigger_token="ohwx",
            config=config,
        )
        assert job.status == TrainingStatus.PENDING
        assert job.current_step == 0
        assert job.total_steps == 3000
        assert job.latest_loss is None
        assert job.checkpoints == []
        assert job.sample_images == []
        assert job.error_message is None
        assert job.container_id is None

    def test_lora_type_values(self):
        """Test LoRAType enum values."""
        assert LoRAType.CHARACTER.value == "character"
        assert LoRAType.STYLE.value == "style"
        assert LoRAType.CONCEPT.value == "concept"

    def test_training_status_values(self):
        """Test TrainingStatus enum values."""
        assert TrainingStatus.PENDING.value == "pending"
        assert TrainingStatus.RUNNING.value == "running"
        assert TrainingStatus.COMPLETED.value == "completed"
        assert TrainingStatus.FAILED.value == "failed"
        assert TrainingStatus.STOPPED.value == "stopped"


class TestDatasetManager:
    """Tests for dataset management."""

    @pytest.fixture
    def manager(self, tmp_path: Path):
        """Create a DatasetManager with temp directory."""
        return DatasetManager(tmp_path)

    def test_create_dataset(self, manager: DatasetManager):
        """Test creating a new dataset."""
        meta = manager.create_dataset(
            name="my_char",
            trigger_token="ohwx",
            lora_type=LoRAType.CHARACTER,
            description="Test character",
        )
        assert meta.name == "my_char"
        assert meta.trigger_token == "ohwx"
        assert meta.lora_type == LoRAType.CHARACTER
        assert meta.description == "Test character"
        assert manager.dataset_exists("my_char")

    def test_duplicate_dataset_error(self, manager: DatasetManager):
        """Test that duplicate dataset names raise an error."""
        manager.create_dataset("test", "tok", LoRAType.CHARACTER)
        with pytest.raises(ValueError, match="already exists"):
            manager.create_dataset("test", "tok", LoRAType.CHARACTER)

    def test_invalid_name_error(self, manager: DatasetManager):
        """Test that invalid dataset names raise an error."""
        with pytest.raises(ValueError, match="Invalid name"):
            manager.create_dataset("123invalid", "tok", LoRAType.CHARACTER)
        with pytest.raises(ValueError, match="Invalid name"):
            manager.create_dataset("has-dash", "tok", LoRAType.CHARACTER)
        with pytest.raises(ValueError, match="Invalid name"):
            manager.create_dataset("has space", "tok", LoRAType.CHARACTER)

    def test_valid_name_patterns(self, manager: DatasetManager):
        """Test valid dataset name patterns."""
        # All should succeed
        manager.create_dataset("valid_name", "tok1", LoRAType.CHARACTER)
        manager.create_dataset("ValidName", "tok2", LoRAType.STYLE)
        manager.create_dataset("name123", "tok3", LoRAType.CONCEPT)
        manager.create_dataset("name_with_numbers_123", "tok4", LoRAType.CHARACTER)

    def test_add_image(self, manager: DatasetManager, tmp_path: Path):
        """Test adding an image to a dataset."""
        manager.create_dataset("test", "tok", LoRAType.CHARACTER)

        # Create a minimal PNG file
        # PNG header + minimal IHDR chunk
        png_bytes = (
            b"\x89PNG\r\n\x1a\n"  # PNG signature
            b"\x00\x00\x00\rIHDR"  # IHDR chunk header
            b"\x00\x00\x00\x01"  # Width: 1
            b"\x00\x00\x00\x01"  # Height: 1
            b"\x08\x02"  # Bit depth: 8, Color type: 2 (RGB)
            b"\x00\x00\x00"  # Compression, filter, interlace
            b"\x90wS\xde"  # CRC
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"  # IDAT
            b"\x00\x00\x00\x00IEND\xaeB`\x82"  # IEND
        )

        filename = manager.add_image("test", png_bytes)
        assert filename.endswith(".png")

        meta = manager.get_metadata("test")
        assert meta.image_count == 1

    def test_add_image_with_caption(self, manager: DatasetManager):
        """Test adding an image with a caption."""
        manager.create_dataset("test", "tok", LoRAType.CHARACTER)

        png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
            b"\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        manager.add_image("test", png_bytes, caption="tok, portrait, studio lighting")

        meta = manager.get_metadata("test")
        assert meta.has_captions is True

    def test_list_datasets(self, manager: DatasetManager):
        """Test listing all datasets."""
        manager.create_dataset("ds1", "tok1", LoRAType.CHARACTER)
        manager.create_dataset("ds2", "tok2", LoRAType.STYLE)

        datasets = manager.list_datasets()
        assert len(datasets) == 2
        names = {ds.name for ds in datasets}
        assert names == {"ds1", "ds2"}

    def test_validate_empty_dataset(self, manager: DatasetManager):
        """Test validation of empty dataset."""
        manager.create_dataset("empty", "tok", LoRAType.CHARACTER)
        report = manager.validate_dataset("empty")
        assert "Too few images" in report
        assert "Ready for training: NO" in report

    def test_get_nonexistent_dataset(self, manager: DatasetManager):
        """Test getting a non-existent dataset raises error."""
        with pytest.raises(ValueError, match="not found"):
            manager.get_metadata("nonexistent")


class TestJobStore:
    """Tests for job persistence."""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a JobStore with temp file."""
        return JobStore(tmp_path / "jobs.json")

    @pytest.fixture
    def sample_job(self):
        """Create a sample training job."""
        config = TrainingConfig(dataset="test", steps=100)
        return TrainingJob(
            job_id="test123",
            dataset_name="test",
            trigger_token="ohwx",
            config=config,
        )

    def test_save_and_get_job(self, store: JobStore, sample_job: TrainingJob):
        """Test saving and retrieving a job."""
        store.save_job(sample_job)
        retrieved = store.get_job("test123")

        assert retrieved is not None
        assert retrieved.job_id == "test123"
        assert retrieved.dataset_name == "test"
        assert retrieved.trigger_token == "ohwx"

    def test_get_nonexistent_job(self, store: JobStore):
        """Test getting a non-existent job returns None."""
        result = store.get_job("nonexistent")
        assert result is None

    def test_list_jobs_empty(self, store: JobStore):
        """Test listing jobs when store is empty."""
        jobs = store.list_jobs()
        assert jobs == []

    def test_list_jobs_with_filter(self, store: JobStore, sample_job: TrainingJob):
        """Test listing jobs with status filter."""
        store.save_job(sample_job)

        # Should find pending job
        pending = store.list_jobs(status=TrainingStatus.PENDING)
        assert len(pending) == 1

        # Should not find running job
        running = store.list_jobs(status=TrainingStatus.RUNNING)
        assert len(running) == 0

    def test_update_job_status(self, store: JobStore, sample_job: TrainingJob):
        """Test updating job status."""
        store.save_job(sample_job)
        updated = store.update_job_status("test123", TrainingStatus.RUNNING)

        assert updated is not None
        assert updated.status == TrainingStatus.RUNNING

    def test_update_job_progress(self, store: JobStore, sample_job: TrainingJob):
        """Test updating job progress."""
        store.save_job(sample_job)
        updated = store.update_job_progress(
            "test123", current_step=50, latest_loss=0.023
        )

        assert updated is not None
        assert updated.current_step == 50
        assert updated.latest_loss == 0.023

    def test_add_checkpoint(self, store: JobStore, sample_job: TrainingJob):
        """Test adding a checkpoint to job."""
        store.save_job(sample_job)
        updated = store.add_checkpoint("test123", "step_500.safetensors")

        assert updated is not None
        assert "step_500.safetensors" in updated.checkpoints

    def test_delete_job(self, store: JobStore, sample_job: TrainingJob):
        """Test deleting a job."""
        store.save_job(sample_job)
        assert store.delete_job("test123") is True
        assert store.get_job("test123") is None

    def test_delete_nonexistent_job(self, store: JobStore):
        """Test deleting a non-existent job returns False."""
        assert store.delete_job("nonexistent") is False

    def test_persistence_across_instances(
        self, tmp_path: Path, sample_job: TrainingJob
    ):
        """Test that jobs persist across store instances."""
        jobs_file = tmp_path / "jobs.json"

        # Save with first instance
        store1 = JobStore(jobs_file)
        store1.save_job(sample_job)

        # Retrieve with second instance
        store2 = JobStore(jobs_file)
        retrieved = store2.get_job("test123")

        assert retrieved is not None
        assert retrieved.job_id == "test123"


class TestProgressParsing:
    """Tests for training progress parsing from logs."""

    @pytest.fixture
    def runner_with_job(self, tmp_path: Path):
        """Create a TrainingRunner with a sample job (total_steps=100)."""
        from lora_trainer.training_runner import TrainingRunner

        store = JobStore(tmp_path / "jobs.json")
        config = TrainingConfig(dataset="test", steps=100)
        job = TrainingJob(
            job_id="test123",
            dataset_name="test",
            trigger_token="ohwx",
            config=config,
            total_steps=100,
        )
        store.save_job(job)

        runner = TrainingRunner(
            datasets_path=tmp_path / "datasets",
            outputs_path=tmp_path / "outputs",
            configs_path=tmp_path / "configs",
            job_store=store,
        )
        return runner, store

    def test_parse_training_progress_not_caching(self, runner_with_job):
        """Test that training progress is parsed, not latent caching progress."""
        runner, store = runner_with_job

        # Logs contain BOTH latent caching AND training progress
        logs = """
        | 0/27 [ 0%] Processing images (Caching latents)
        | 15/27 [55%] Caching latents
        | 27/27 [100%] Caching complete
        | 50/100 [50%] Training: loss: 0.0234
        """

        runner._parse_progress("test123", logs)

        job = store.get_job("test123")
        assert job.current_step == 50, "Should match training progress, not caching"
        assert job.total_steps == 100, "Should keep expected total steps"
        assert job.latest_loss == pytest.approx(0.0234)

    def test_parse_only_caching_progress_uses_fallback(self, runner_with_job):
        """Test that caching progress is used as fallback when no training match."""
        runner, store = runner_with_job

        # Only latent caching, no training started yet
        logs = "| 15/27 [55%] Caching latents"

        runner._parse_progress("test123", logs)

        job = store.get_job("test123")
        # Falls back to caching progress since no match for total_steps=100
        assert job.current_step == 15
        # Note: total_steps may be updated as fallback behavior

    def test_parse_progress_prefers_matching_total(self, runner_with_job):
        """Test that progress matching job.total_steps is preferred."""
        runner, store = runner_with_job

        # Multiple progress bars with different totals
        logs = """
        | 5/10 [50%] Loading checkpoints
        | 75/100 [75%] Training
        | 3/5 [60%] Sampling
        """

        runner._parse_progress("test123", logs)

        job = store.get_job("test123")
        assert job.current_step == 75, "Should prefer progress matching total_steps"
        assert job.total_steps == 100

    def test_parse_loss_scientific_notation(self, runner_with_job):
        """Test parsing loss in scientific notation."""
        runner, store = runner_with_job

        logs = "| 50/100 [50%] Training: loss: 2.891e-01"

        runner._parse_progress("test123", logs)

        job = store.get_job("test123")
        assert job.latest_loss == pytest.approx(0.2891)

    def test_parse_step_patterns_alternative_formats(self, runner_with_job):
        """Test alternative step format patterns."""
        runner, store = runner_with_job

        # Test "Step X/Y" format
        logs = "Step 42/100 completed"
        runner._parse_progress("test123", logs)

        job = store.get_job("test123")
        assert job.current_step == 42

    def test_parse_single_step_pattern(self, runner_with_job):
        """Test single step pattern (no total) as fallback."""
        runner, store = runner_with_job

        logs = "step: 85 - processing..."
        runner._parse_progress("test123", logs)

        job = store.get_job("test123")
        assert job.current_step == 85

    def test_parse_no_progress_no_update(self, runner_with_job):
        """Test that irrelevant logs don't update job state."""
        runner, store = runner_with_job

        logs = "Loading model weights..."
        runner._parse_progress("test123", logs)

        job = store.get_job("test123")
        assert job.current_step == 0  # Unchanged from initial
        assert job.latest_loss is None
