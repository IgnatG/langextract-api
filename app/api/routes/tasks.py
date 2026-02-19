"""Task management routes (status polling & revocation)."""

from __future__ import annotations

import json
import logging

from celery.result import AsyncResult
from fastapi import APIRouter

from app.core.constants import REDIS_PREFIX_TASK_RESULT, STATUS_REVOKED
from app.core.redis import get_redis_client
from app.schemas import (
    TaskRevokeResponse,
    TaskState,
    TaskStatusResponse,
)
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tasks"])


def _fetch_redis_result(task_id: str) -> dict | None:
    """Look up a persisted result under the predictable key.

    Returns the parsed dict, or ``None`` when no stored result
    exists or Redis is unreachable.

    Args:
        task_id: The Celery task identifier.

    Returns:
        The stored result dict or ``None``.
    """
    try:
        client = get_redis_client()
        try:
            raw = client.get(f"{REDIS_PREFIX_TASK_RESULT}{task_id}")
        finally:
            client.close()
        if raw:
            return json.loads(raw)
    except Exception:
        logger.debug(
            "Redis fallback lookup failed for task %s",
            task_id,
            exc_info=True,
        )
    return None


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
)
def get_task_status(task_id: str) -> TaskStatusResponse:
    """Poll the current status and result of a task.

    When Celery reports ``PENDING`` (which also covers expired
    or unknown task IDs), the endpoint falls back to the
    predictable ``task_result:{task_id}`` key in Redis so that
    results survive backend expiry.
    """
    result = AsyncResult(task_id, app=celery_app)

    response = TaskStatusResponse(
        task_id=task_id,
        state=TaskState(result.state),
    )

    if result.state == TaskState.PENDING:
        # Celery returns PENDING for unknown / expired tasks.
        # Try the predictable Redis key as a fallback.
        redis_result = _fetch_redis_result(task_id)
        if redis_result is not None:
            response.state = TaskState.SUCCESS
            response.result = redis_result
        else:
            response.progress = {
                "status": "Task is waiting to be processed",
            }
    elif result.state == TaskState.PROGRESS:
        response.progress = result.info
    elif result.state == TaskState.SUCCESS:
        response.result = result.result
    elif result.state == TaskState.FAILURE:
        response.error = str(result.info)

    return response


@router.delete(
    "/tasks/{task_id}",
    response_model=TaskRevokeResponse,
)
def revoke_task(
    task_id: str,
    terminate: bool = False,
) -> TaskRevokeResponse:
    """Revoke a pending or running task.

    Set ``terminate=true`` to send SIGTERM to a running worker
    process.
    """
    celery_app.control.revoke(task_id, terminate=terminate)
    return TaskRevokeResponse(
        task_id=task_id,
        status=STATUS_REVOKED,
        message=(f"Task revocation signal sent (terminate={terminate})"),
    )
