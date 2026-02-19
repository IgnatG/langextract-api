"""Health check response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Returned by the health-check endpoint."""

    status: str = Field(
        ...,
        description="Service health status",
    )
    version: str = Field(
        ...,
        description="Application version",
    )


class CeleryHealthResponse(BaseModel):
    """Returned by the Celery health-check endpoint."""

    status: str = Field(
        ...,
        description="Overall Celery health status",
    )
    message: str = Field(
        ...,
        description="Human-readable summary",
    )
    workers: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-worker status details",
    )
