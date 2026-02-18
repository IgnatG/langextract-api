"""
Celery tasks — long-running extraction jobs processed by workers.

Each task follows the pattern:
1. Accept a serialisable payload.
2. Report progress via ``self.update_state(state="PROGRESS", ...)``.
3. Return a JSON-serialisable result dict.
4. Optionally POST the result to a ``callback_url`` (webhook).

Extraction is performed by Google's ``langextract`` library
(https://github.com/google/langextract).
"""

import logging
import time
from typing import Any

import httpx
import langextract as lx

from app.dependencies import get_settings
from app.extraction_defaults import (
    DEFAULT_EXAMPLES,
    DEFAULT_PROMPT_DESCRIPTION,
)
from app.worker import celery_app

logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _fire_webhook(
    callback_url: str,
    payload: dict[str, Any],
) -> None:
    """POST *payload* to *callback_url*, logging but never raising."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(callback_url, json=payload)
            resp.raise_for_status()
        logger.info(
            "Webhook delivered to %s (status %s)",
            callback_url,
            resp.status_code,
        )
    except Exception as exc:
        logger.error(
            "Webhook delivery to %s failed: %s",
            callback_url,
            exc,
        )


# ── LangExtract helpers ─────────────────────────────────────────────────────


def _build_examples(
    raw_examples: list[dict[str, Any]],
) -> list[lx.data.ExampleData]:
    """Convert plain-dict examples into ``lx.data.ExampleData`` objects.

    Args:
        raw_examples: List of dicts, each with ``text`` and
            ``extractions`` keys.

    Returns:
        A list of ``ExampleData`` ready for ``lx.extract()``.
    """
    return [
        lx.data.ExampleData(
            text=ex["text"],
            extractions=[
                lx.data.Extraction(
                    extraction_class=e["extraction_class"],
                    extraction_text=e["extraction_text"],
                    attributes=e.get("attributes"),
                )
                for e in ex.get("extractions", [])
            ],
        )
        for ex in raw_examples
    ]


def _resolve_api_key(provider: str) -> str | None:
    """Pick the correct API key for *provider* from settings.

    Args:
        provider: Model ID string (e.g. ``gpt-4o``,
            ``gpt-4o``).

    Returns:
        An API key string, or ``None`` if nothing is configured.
    """
    settings = get_settings()
    lower = provider.lower()
    if "gpt" in lower or "openai" in lower:
        return settings.OPENAI_API_KEY or None
    # Prefer the dedicated langextract key, fall back to Gemini key
    return settings.LANGEXTRACT_API_KEY or settings.GEMINI_API_KEY or None


def _is_openai_model(provider: str) -> bool:
    """Return ``True`` if *provider* is an OpenAI model.

    Args:
        provider: Model ID string.

    Returns:
        Boolean indicating whether OpenAI-specific flags are needed.
    """
    lower = provider.lower()
    return "gpt" in lower or "openai" in lower


def _convert_extractions(
    result: lx.data.AnnotatedDocument,
) -> list[dict[str, Any]]:
    """Flatten ``AnnotatedDocument.extractions`` into JSON-safe dicts.

    Args:
        result: The annotated document returned by ``lx.extract()``.

    Returns:
        A list of entity dicts matching ``ExtractedEntity`` schema.
    """
    entities: list[dict[str, Any]] = []
    for ext in result.extractions or []:
        entity: dict[str, Any] = {
            "extraction_class": ext.extraction_class,
            "extraction_text": ext.extraction_text,
            "attributes": ext.attributes or {},
            "char_start": (ext.char_interval.start_pos if ext.char_interval else None),
            "char_end": (ext.char_interval.end_pos if ext.char_interval else None),
        }
        entities.append(entity)
    return entities


# ── Core extraction logic ───────────────────────────────────────────────────


def _run_extraction(
    task_self: Any | None,
    document_url: str | None = None,
    raw_text: str | None = None,
    provider: str = "gpt-4o",
    passes: int = 1,
    extraction_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Core extraction logic shared by single and batch tasks.

    Delegates to ``langextract.extract()`` for the heavy lifting.

    Args:
        task_self: Bound Celery task instance (for progress updates).
        document_url: URL to the source document.
        raw_text: Raw text blob to process directly.
        provider: LLM model ID to use (e.g. ``gpt-4o``).
        passes: Number of extraction passes.
        extraction_config: Optional overrides for prompt, examples,
            and LangExtract parameters.

    Returns:
        A dict containing the extraction result and metadata.
    """
    settings = get_settings()
    extraction_config = extraction_config or {}
    source = document_url or "<raw_text>"
    start_ms = int(time.time() * 1000)

    logger.info(
        "Starting extraction for %s (model=%s, passes=%d)",
        source,
        provider,
        passes,
    )

    # ── Step 1: Determine input ─────────────────────────────────────────
    if task_self:
        task_self.update_state(
            state="PROGRESS",
            meta={
                "step": "preparing",
                "source": source,
                "percent": 5,
            },
        )

    # langextract accepts a URL string directly (it downloads it)
    # or plain text.  Prefer URL when available.
    text_input: str = document_url if document_url else (raw_text or "")

    # ── Step 2: Build prompt & examples ─────────────────────────────────
    prompt_description: str = extraction_config.get(
        "prompt_description",
        DEFAULT_PROMPT_DESCRIPTION,
    )

    raw_examples: list[dict[str, Any]] = extraction_config.get(
        "examples",
        DEFAULT_EXAMPLES,
    )
    examples = _build_examples(raw_examples)

    # ── Step 3: Assemble lx.extract() kwargs ────────────────────────────
    if task_self:
        task_self.update_state(
            state="PROGRESS",
            meta={
                "step": "extracting",
                "source": source,
                "percent": 10,
            },
        )

    extract_kwargs: dict[str, Any] = {
        "text_or_documents": text_input,
        "prompt_description": prompt_description,
        "examples": examples,
        "model_id": provider,
        "extraction_passes": passes,
        "max_workers": extraction_config.get(
            "max_workers",
            settings.DEFAULT_MAX_WORKERS,
        ),
        "max_char_buffer": extraction_config.get(
            "max_char_buffer",
            settings.DEFAULT_MAX_CHAR_BUFFER,
        ),
        "show_progress": False,  # disable tqdm inside Celery worker
    }

    # Optional overrides
    if "additional_context" in extraction_config:
        extract_kwargs["additional_context"] = extraction_config["additional_context"]
    if "temperature" in extraction_config:
        extract_kwargs["temperature"] = extraction_config["temperature"]
    if "context_window_chars" in extraction_config:
        extract_kwargs["context_window_chars"] = extraction_config[
            "context_window_chars"
        ]

    # API key
    api_key = _resolve_api_key(provider)
    if api_key:
        extract_kwargs["api_key"] = api_key

    # OpenAI-specific flags
    if _is_openai_model(provider):
        extract_kwargs["fence_output"] = True
        extract_kwargs["use_schema_constraints"] = False

    # ── Step 4: Run LangExtract ─────────────────────────────────────────
    logger.info(
        "Calling lx.extract() for %s (model_id=%s, passes=%d)",
        source,
        provider,
        passes,
    )

    lx_result = lx.extract(**extract_kwargs)

    # lx.extract() returns AnnotatedDocument for string input
    if isinstance(lx_result, list):
        lx_result = lx_result[0] if lx_result else (lx.data.AnnotatedDocument())

    # ── Step 5: Convert to response schema ──────────────────────────────
    if task_self:
        task_self.update_state(
            state="PROGRESS",
            meta={
                "step": "post_processing",
                "source": source,
                "percent": 90,
            },
        )

    entities = _convert_extractions(lx_result)
    elapsed_ms = int(time.time() * 1000) - start_ms

    result: dict[str, Any] = {
        "status": "completed",
        "source": source,
        "data": {
            "entities": entities,
            "metadata": {
                "provider": provider,
                "tokens_used": 0,
                "processing_time_ms": elapsed_ms,
            },
        },
    }

    logger.info(
        "Extraction completed for %s — %d entities in %d ms",
        source,
        len(entities),
        elapsed_ms,
    )
    return result


# ── Extraction task ─────────────────────────────────────────────────────────


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
) -> dict[str, Any]:
    """
    Extract structured data from a single document.

    Args:
        document_url: URL to the source document.
        raw_text: Raw text blob to process directly.
        provider: AI provider / model to use.
        passes: Number of extraction passes.
        callback_url: Optional webhook URL to POST the result to.
        extraction_config: Optional overrides for the extraction
            pipeline (prompt template, chunking strategy, etc.).

    Returns:
        A dict containing the extraction result and metadata.
    """
    try:
        result = _run_extraction(
            task_self=self,
            document_url=document_url,
            raw_text=raw_text,
            provider=provider,
            passes=passes,
            extraction_config=extraction_config,
        )

        # Fire webhook if requested
        if callback_url:
            _fire_webhook(
                callback_url,
                {"task_id": self.request.id, **result},
            )

        return result

    except Exception as exc:
        logger.exception(
            "Extraction failed for %s: %s",
            document_url or "<raw_text>",
            exc,
        )
        raise self.retry(exc=exc) from exc


# ── Batch extraction task ──────────────────────────────────────────────────


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
) -> dict[str, Any]:
    """
    Process a batch of documents sequentially with progress tracking.

    Args:
        batch_id: Unique identifier for this batch.
        documents: List of dicts, each with ``document_url`` and/or
            ``raw_text``, plus optional ``provider``, ``passes``,
            ``callback_url``, and ``extraction_config``.
        callback_url: Optional batch-level webhook URL. Overrides
            any per-document ``callback_url``.

    Returns:
        Aggregated batch result with per-document outcomes.
    """
    total = len(documents)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    logger.info("Starting batch %s with %d documents", batch_id, total)

    for idx, doc in enumerate(documents):
        source = doc.get("document_url") or "<raw_text>"

        try:
            outcome = _run_extraction(
                task_self=self,
                document_url=doc.get("document_url"),
                raw_text=doc.get("raw_text"),
                provider=doc.get("provider", "gpt-4o"),
                passes=doc.get("passes", 1),
                extraction_config=doc.get("extraction_config", {}),
            )
            results.append(outcome)
        except Exception as exc:
            logger.error(
                "Batch %s — document %s failed: %s",
                batch_id,
                source,
                exc,
            )
            errors.append({"source": source, "error": str(exc)})

        self.update_state(
            state="PROGRESS",
            meta={
                "batch_id": batch_id,
                "current": idx + 1,
                "total": total,
                "successful": len(results),
                "failed": len(errors),
                "percent": int((idx + 1) / total * 100),
            },
        )

    batch_result: dict[str, Any] = {
        "status": "completed",
        "batch_id": batch_id,
        "total": total,
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }

    if callback_url:
        _fire_webhook(
            callback_url,
            {"task_id": self.request.id, **batch_result},
        )

    return batch_result
