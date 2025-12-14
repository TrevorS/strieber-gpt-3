# ABOUTME: Factory for creating job storage backends.
# ABOUTME: Supports JSON file storage (default) and HTTP storage via responses-api.

import os
from pathlib import Path

from lora_trainer.job_store_base import JobStoreBase


def create_job_store(data_path: Path) -> JobStoreBase:
    """Create a job store based on environment configuration.

    Args:
        data_path: Base data directory path.

    Returns:
        JobStoreBase implementation (JsonJobStore or HttpJobStore).

    Environment variables:
        JOB_STORE_BACKEND: "json" (default) or "http"
        STORAGE_API_URL: API URL for HTTP backend (default: http://responses-api:9150)
    """
    backend = os.environ.get("JOB_STORE_BACKEND", "json")

    if backend == "http":
        from lora_trainer.job_store_http import HttpJobStore

        return HttpJobStore(
            os.environ.get("STORAGE_API_URL", "http://responses-api:8000")
        )
    else:
        from lora_trainer.job_store_json import JsonJobStore

        return JsonJobStore(data_path / "jobs.json")
