"""Single-document extraction Celery task."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from celery.exceptions import Retry

from app.core.config import get_settings
from app.core.constants import REDIS_PREFIX_TASK_RESULT
from app.core.metrics import record_task_completed
from app.core.redis import get_redis_client
from app.services.extractor import async_run_extraction
from app.services.webhook import fire_webhook
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


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
        extraction_config: Optional overrides for the
            extraction pipeline.
        callback_headers: Optional extra HTTP headers to
            send with the webhook request.

    Returns:
        A dict containing the extraction result and metadata.
    """
    start_s = time.monotonic()
    try:
        # Use the async extraction path to enable I/O-CPU overlap
        # via ``lx.async_extract()`` and ``litellm.acompletion()``.
        # ``asyncio.run()`` is safe here because Celery worker threads
        # do not have a running event loop.
        result = asyncio.run(
            async_run_extraction(
                task_self=self,
                document_url=document_url,
                raw_text=raw_text,
                provider=provider,
                passes=passes,
                extraction_config=extraction_config,
            )
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

        record_task_completed(
            success=True,
            duration_s=elapsed_s,
        )
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
