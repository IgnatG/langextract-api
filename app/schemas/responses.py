"""Response models for task and batch endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.constants import STATUS_REVOKED, STATUS_SUBMITTED
from app.schemas.enums import TaskState


class TaskSubmitResponse(BaseModel):
    """Returned immediately when a task is submitted."""

    task_id: str = Field(
        ...,
        description="Unique Celery task identifier",
    )
    status: str = Field(
        default=STATUS_SUBMITTED,
        description="Initial task status",
    )
    message: str = Field(
        default="Task submitted successfully",
        description="Human-readable status message",
    )


class BatchTaskSubmitResponse(BaseModel):
    """Returned when a batch is submitted."""

    batch_task_id: str = Field(
        ...,
        description=("Celery task ID for the batch orchestrator"),
    )
    document_task_ids: list[str] = Field(
        default_factory=list,
        description=("Per-document Celery task IDs (parallel mode)"),
    )
    status: str = Field(
        default=STATUS_SUBMITTED,
        description="Initial batch status",
    )
    message: str = Field(
        default="Batch submitted successfully",
        description="Human-readable status message",
    )


class TaskStatusResponse(BaseModel):
    """Returned when polling for task status."""

    task_id: str = Field(
        ...,
        description="Unique Celery task identifier",
    )
    state: TaskState = Field(
        ...,
        description="Current state of the task",
    )
    progress: dict[str, Any] | None = Field(
        default=None,
        description=("Progress metadata (available in PROGRESS state)"),
    )
    result: Any | None = Field(
        default=None,
        description=("Task result (available in SUCCESS state)"),
    )
    error: str | None = Field(
        default=None,
        description=("Error message (available in FAILURE state)"),
    )


class TaskRevokeResponse(BaseModel):
    """Returned when a task revocation is requested."""

    task_id: str = Field(
        ...,
        description="Identifier of the revoked task",
    )
    status: str = Field(
        default=STATUS_REVOKED,
        description="Revocation status",
    )
    message: str = Field(
        default="Task revocation signal sent",
        description="Human-readable status message",
    )
