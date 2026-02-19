"""Batch extraction route."""

from __future__ import annotations

import logging

from celery import group
from fastapi import APIRouter, HTTPException

from app.api.routes.extract import _validate_request_urls
from app.core.constants import STATUS_SUBMITTED
from app.core.metrics import record_task_submitted
from app.core.security import validate_url
from app.schemas import (
    BatchExtractionRequest,
    BatchTaskSubmitResponse,
)
from app.workers.batch_task import finalize_batch
from app.workers.extract_task import extract_document

logger = logging.getLogger(__name__)

router = APIRouter(tags=["extraction"])


@router.post(
    "/extract/batch",
    response_model=BatchTaskSubmitResponse,
)
def submit_batch_extraction(
    request: BatchExtractionRequest,
) -> BatchTaskSubmitResponse:
    """Submit a batch of documents for extraction.

    Dispatches per-document tasks via a Celery ``group()``
    at the API level so that child task IDs are available
    immediately.  A lightweight ``finalize_batch`` task
    monitors the children (via non-blocking retry-based
    polling) and aggregates results once all documents are
    done.

    Returns a batch-level task ID plus per-document task IDs
    so callers can retry or poll individual documents
    independently.  If a batch-level ``callback_url`` is
    supplied the aggregated result is POSTed there on
    completion.
    """
    # Validate all URLs up-front
    for doc in request.documents:
        _validate_request_urls(doc)

    if request.callback_url:
        try:
            validate_url(
                str(request.callback_url),
                purpose="batch callback_url",
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    documents = [doc.model_dump(mode="json") for doc in request.documents]
    # Convert nested ExtractionConfig → flat dicts
    for doc_dict in documents:
        cfg = doc_dict.get("extraction_config")
        if isinstance(cfg, dict) and cfg:
            doc_dict["extraction_config"] = {
                k: v for k, v in cfg.items() if v is not None
            }

    # ── Fan-out: dispatch group to get child IDs ────────────
    signatures = [
        extract_document.s(
            document_url=doc_dict.get("document_url"),
            raw_text=doc_dict.get("raw_text"),
            provider=doc_dict.get("provider", "gpt-4o"),
            passes=doc_dict.get("passes", 1),
            extraction_config=doc_dict.get(
                "extraction_config",
                {},
            ),
            callback_url=doc_dict.get("callback_url"),
            callback_headers=doc_dict.get(
                "callback_headers",
            ),
        )
        for doc_dict in documents
    ]
    group_result = group(signatures).apply_async()
    child_ids = [r.id for r in group_result.children]

    # ── Aggregation: non-blocking finalize task ─────────────
    task = finalize_batch.apply_async(
        kwargs={
            "batch_id": request.batch_id,
            "child_task_ids": child_ids,
            "documents": documents,
            "callback_url": (
                str(request.callback_url) if request.callback_url else None
            ),
            "callback_headers": request.callback_headers,
        },
        countdown=2,
    )

    record_task_submitted()

    return BatchTaskSubmitResponse(
        batch_task_id=task.id,
        document_task_ids=child_ids,
        status=STATUS_SUBMITTED,
        message=(
            f"Batch '{request.batch_id}' submitted "
            f"with {len(request.documents)} document(s)"
        ),
    )
