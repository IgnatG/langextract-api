"""Batch finalisation Celery task (non-blocking polling)."""

from __future__ import annotations

import logging
from typing import Any

from celery.result import AsyncResult

from app.core.constants import STATUS_COMPLETED
from app.schemas.enums import TaskState
from app.services.webhook import fire_webhook
from app.workers.celery_app import celery_app
from app.workers.extract_task import _store_result_in_redis

logger = logging.getLogger(__name__)

# Maximum time (in seconds) to wait for child tasks before
# giving up and reporting a partial result.  With a 5-second
# retry countdown this allows for roughly 1 hour of waiting.
_FINALIZE_MAX_RETRIES: int = 720
_FINALIZE_COUNTDOWN_S: int = 5


@celery_app.task(
    bind=True,
    name="tasks.finalize_batch",
    max_retries=_FINALIZE_MAX_RETRIES,
    default_retry_delay=_FINALIZE_COUNTDOWN_S,
)
def finalize_batch(
    self,
    *,
    batch_id: str,
    child_task_ids: list[str],
    documents: list[dict[str, Any]],
    callback_url: str | None = None,
    callback_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Aggregate results from child extraction tasks.

    The batch API route dispatches per-document tasks via a
    Celery ``group()`` and then schedules this task.
    ``finalize_batch`` polls the children using Celery's retry
    mechanism so that no worker slot is blocked while children
    are still running.

    Once all children are ready (or the retry budget is
    exhausted), it aggregates success/failure results, fires
    an optional webhook, and persists the batch result in Redis.

    Args:
        batch_id: Unique identifier for this batch.
        child_task_ids: Celery task IDs of the per-document
            extraction tasks.
        documents: The original document dicts (for error
            source attribution).
        callback_url: Optional batch-level webhook URL.
        callback_headers: Optional extra HTTP headers for the
            webhook request.

    Returns:
        Aggregated batch result with per-document outcomes.
    """
    total = len(child_task_ids)
    children = [AsyncResult(tid, app=celery_app) for tid in child_task_ids]

    # ── Poll: re-schedule if children are still running ─────
    if not all(c.ready() for c in children):
        completed = sum(1 for c in children if c.ready())

        self.update_state(
            state=TaskState.PROGRESS,
            meta={
                "batch_id": batch_id,
                "document_task_ids": child_task_ids,
                "total": total,
                "completed": completed,
            },
        )

        if self.request.retries < self.max_retries:
            raise self.retry(
                countdown=_FINALIZE_COUNTDOWN_S,
            )

        logger.warning(
            "Batch %s: timed out after %d retries — finalising with partial results",
            batch_id,
            self.request.retries,
        )

    # ── Aggregate results ───────────────────────────────────
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for child, doc in zip(
        children,
        documents,
        strict=True,
    ):
        source = doc.get("document_url") or "<raw_text>"
        if child.successful():
            results.append(child.result)
        else:
            err_msg = str(child.result) if child.result else "Unknown error"
            errors.append(
                {"source": source, "error": err_msg},
            )

    batch_result: dict[str, Any] = {
        "status": STATUS_COMPLETED,
        "batch_id": batch_id,
        "total": total,
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
        "document_task_ids": child_task_ids,
    }

    if callback_url:
        fire_webhook(
            callback_url,
            {"task_id": self.request.id, **batch_result},
            extra_headers=callback_headers,
        )

    # Persist batch result under a predictable Redis key
    _store_result_in_redis(
        self.request.id,
        batch_result,
    )

    return batch_result
