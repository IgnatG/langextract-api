"""
Centralised constants used across the application.

Keeping magic strings in one place makes it easy to rename keys,
avoids silent typos, and keeps ``grep`` useful when debugging.
"""

from __future__ import annotations

# ── Redis key prefixes ──────────────────────────────────────────────────────
# Every Redis key written by the application starts with one of
# these prefixes so the keyspace stays organised and collisions
# are impossible.

REDIS_PREFIX_TASK_RESULT: str = "task_result:"
"""Prefix for persisted extraction results (fallback store)."""

REDIS_PREFIX_IDEMPOTENCY: str = "idempotency:"
"""Prefix for idempotency-key → task-ID mappings."""

REDIS_PREFIX_METRICS: str = "metrics:"
"""Prefix for atomic metric counters."""

REDIS_PREFIX_EXTRACTION_CACHE: str = "extraction_cache:"
"""Prefix for extraction-result cache entries."""


# ── Task / result status strings ────────────────────────────────────────────
# Used in Celery ``update_state()`` calls and in result dicts
# returned by workers and the extractor service.

STATUS_COMPLETED: str = "completed"
"""Result status when an extraction finishes successfully."""

STATUS_SUBMITTED: str = "submitted"
"""Response status returned immediately after task submission."""

STATUS_REVOKED: str = "revoked"
"""Response status returned when a task revocation is requested."""
