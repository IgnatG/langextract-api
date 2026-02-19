"""
Celery tasks — thin wrappers around service functions.

Each task follows the pattern:
1. Accept a serialisable payload.
2. Delegate real work to ``app.services.*``.
3. Report progress via ``self.update_state()``.
4. Persist results and optionally fire a webhook.

Business logic lives in ``app.services.extractor`` so it can be
tested and reused independently of Celery.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from celery import group
from celery.exceptions import Retry

from app.core.config import get_redis_client, get_settings
from app.core.constants import (
    REDIS_PREFIX_TASK_RESULT,
    STATUS_COMPLETED,
)
from app.core.metrics import record_task_completed
from app.schemas.extraction import TaskState
from app.services.extractor import run_extraction
from app.services.webhook import fire_webhook
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


# ── Single-document extraction ──────────────────────────────────────────────


def _store_result_in_redis(
    task_id: str,
    result: dict[str, Any],
) -> None:
    """Persist *result* under a predictable Redis key.

    Stored separately from Celery's result backend so the
    task-status endpoint can fall back to this key when
    Celery metadata has expired or is unavailable.

    Args:
        task_id: The Celery task identifier.
        result: JSON-serialisable result dict.
    """
    try:
        settings = get_settings()
        client = get_redis_client()
        try:
            client.setex(
                f"{REDIS_PREFIX_TASK_RESULT}{task_id}",
                settings.RESULT_EXPIRES,
                json.dumps(result),
            )
        finally:
            client.close()
    except Exception:
        logger.warning(
            "Failed to persist result for task %s",
            task_id,
            exc_info=True,
        )


@celery_app.task(
    bind=True,
    name="tasks.extract_document",
    max_retries=3,
    default_retry_delay=60,
)
def extract_document(
    self,
    document_url: str | None = None,
    raw_text: str | None = None,
    provider: str = "gpt-4o",
    passes: int = 1,
    callback_url: str | None = None,
    extraction_config: dict[str, Any] | None = None,
    callback_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Extract structured data from a single document.

    Args:
        document_url: URL to the source document.
        raw_text: Raw text blob to process directly.
        provider: AI provider / model to use.
        passes: Number of extraction passes.
        callback_url: Optional webhook URL.
        extraction_config: Optional overrides for the extraction
            pipeline.
        callback_headers: Optional extra HTTP headers to send
            with the webhook request (e.g. Authorization).

    Returns:
        A dict containing the extraction result and metadata.
    """
    start_s = time.monotonic()
    try:
        result = run_extraction(
            task_self=self,
            document_url=document_url,
            raw_text=raw_text,
            provider=provider,
            passes=passes,
            extraction_config=extraction_config,
        )
        elapsed_s = time.monotonic() - start_s

        # Persist result under a predictable Redis key
        _store_result_in_redis(self.request.id, result)

        # Fire webhook if requested
        if callback_url:
            fire_webhook(
                callback_url,
                {"task_id": self.request.id, **result},
                extra_headers=callback_headers,
            )

        record_task_completed(success=True, duration_s=elapsed_s)
        return result

    except Retry:
        # Celery retry — do not record as a final failure.
        raise

    except Exception as exc:
        elapsed_s = time.monotonic() - start_s
        is_final = self.request.retries >= self.max_retries
        if is_final:
            record_task_completed(
                success=False,
                duration_s=elapsed_s,
            )
        logger.exception(
            "Extraction failed (attempt %d/%d) for %s: %s",
            self.request.retries + 1,
            self.max_retries + 1,
            document_url or "<raw_text>",
            exc,
        )
        raise self.retry(exc=exc) from exc


# ── Batch extraction ────────────────────────────────────────────────────────


@celery_app.task(
    bind=True,
    name="tasks.extract_batch",
    max_retries=1,
    default_retry_delay=120,
)
def extract_batch(
    self,
    batch_id: str,
    documents: list[dict[str, Any]],
    callback_url: str | None = None,
    callback_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fan out per-document Celery tasks via ``group()``.

    Each document is dispatched as an independent
    ``extract_document`` task so it gets its own task ID,
    independent retries, and result storage.

    Args:
        batch_id: Unique identifier for this batch.
        documents: List of extraction request dicts.
        callback_url: Optional batch-level webhook URL.
        callback_headers: Optional extra HTTP headers to send
            with the webhook request (e.g. Authorization).

    Returns:
        Aggregated batch result with per-document outcomes.
    """
    total = len(documents)

    logger.info(
        "Starting batch %s with %d documents via Celery group",
        batch_id,
        total,
    )

    # ── Fan-out via Celery group ────────────────────────────
    signatures = [
        extract_document.s(
            document_url=doc.get("document_url"),
            raw_text=doc.get("raw_text"),
            provider=doc.get("provider", "gpt-4o"),
            passes=doc.get("passes", 1),
            extraction_config=doc.get("extraction_config", {}),
        )
        for doc in documents
    ]

    job = group(signatures)
    group_result = job.apply_async()

    # Store the child task IDs on the parent for the route
    # to return immediately.
    child_ids = [r.id for r in group_result.children]
    self.update_state(
        state=TaskState.PROGRESS,
        meta={
            "batch_id": batch_id,
            "document_task_ids": child_ids,
            "total": total,
        },
    )

    # ── Wait for all children to finish ─────────────────────
    group_result.get(
        disable_sync_subtasks=False,
        propagate=False,
    )

    # ── Aggregate results ───────────────────────────────────
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for child, doc in zip(group_result.children, documents, strict=True):
        source = doc.get("document_url") or "<raw_text>"
        if child.successful():
            results.append(child.result)
        else:
            err_msg = str(child.result) if child.result else "Unknown error"
            errors.append({"source": source, "error": err_msg})

    batch_result: dict[str, Any] = {
        "status": STATUS_COMPLETED,
        "batch_id": batch_id,
        "total": total,
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
        "document_task_ids": child_ids,
    }

    if callback_url:
        fire_webhook(
            callback_url,
            {"task_id": self.request.id, **batch_result},
            extra_headers=callback_headers,
        )

    # Persist batch result under a predictable Redis key
    _store_result_in_redis(self.request.id, batch_result)

    return batch_result
