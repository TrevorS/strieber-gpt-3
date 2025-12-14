# ABOUTME: Pydantic models for LoRA training data structures.
# Defines schemas for datasets, training configs, and job state.

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class LoRAType(str, Enum):
    """Type of LoRA being trained."""

    CHARACTER = "character"
    STYLE = "style"
    CONCEPT = "concept"


class TrainingStatus(str, Enum):
    """Training job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class DatasetMetadata(BaseModel):
    """Metadata for a training dataset."""

    name: str
    trigger_token: str
    lora_type: LoRAType
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    image_count: int = 0
    has_captions: bool = False


class TrainingConfig(BaseModel):
    """ai-toolkit training configuration."""

    model: str = "z-image-turbo"
    training_adapter: str = "/weights/z-image-turbo/training_adapter_v2.safetensors"
    dataset: str
    image_size: int = 1024
    steps: int = 3000
    batch_size: int = 1
    lr: float = 0.0001
    lora_rank: int = 8
    checkpoint_every: int = 500
    sample_every: int = 250
    sample_prompts: List[str] = Field(default_factory=list)


class TrainingJob(BaseModel):
    """Training job state."""

    job_id: str
    dataset_name: str
    trigger_token: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_step: int = 0
    total_steps: int = 3000
    latest_loss: Optional[float] = None
    checkpoints: List[str] = Field(default_factory=list)
    sample_images: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    container_id: Optional[str] = None
