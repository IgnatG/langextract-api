"""Task state enumeration used across the application."""

from __future__ import annotations

from enum import StrEnum


class TaskState(StrEnum):
    """Possible states of a Celery task."""

    PENDING = "PENDING"
    STARTED = "STARTED"
    PROGRESS = "PROGRESS"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    REVOKED = "REVOKED"
    RETRY = "RETRY"
