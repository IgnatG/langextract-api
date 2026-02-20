"""
Multi-tier extraction-result cache.

Provides an ``ExtractionCache`` that sits **above** the LLM call
layer and caches complete extraction results (entity lists +
metadata).  A cache hit skips the entire LangExtract pipeline,
returning pre-computed results in < 500 ms with zero API cost.

**Cache key composition**

The cache key is a SHA-256 digest of every parameter that
affects the extraction output:

- input text (or a SHA-256 of text for large documents)
- prompt description
- examples (serialised to canonical JSON)
- model ID / consensus providers
- temperature
- passes count

Any change to prompt, schema, or model automatically
invalidates the cache — no manual versioning required.

**Backends**

===============  ====================================
``redis``        Default.  Cross-worker, cross-job.
``disk``         ``diskcache``-backed.  Dev/offline.
``none``         Caching disabled.
===============  ====================================

Select via ``EXTRACTION_CACHE_BACKEND`` env-var.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import time
from typing import Any

from app.core.constants import REDIS_PREFIX_EXTRACTION_CACHE

logger = logging.getLogger(__name__)

# Maximum text length included verbatim in the cache key.
# Longer texts are SHA-256 hashed first to keep keys compact.
_TEXT_HASH_THRESHOLD: int = 50_000


def _stable_json(obj: Any) -> str:
    """Serialise *obj* to canonical JSON (sorted keys, no whitespace).

    Args:
        obj: Any JSON-serialisable object.

    Returns:
        Deterministic JSON string suitable for hashing.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def build_cache_key(
    text: str,
    prompt_description: str,
    examples: list[dict[str, Any]],
    model_id: str,
    temperature: float | None,
    passes: int,
    consensus_providers: list[str] | None = None,
    consensus_threshold: float | None = None,
) -> str:
    """Build a deterministic cache key for an extraction job.

    The key is a SHA-256 hex digest that includes **every**
    parameter affecting the extraction output.  Any change in
    prompt, schema, model, or temperature produces a different
    key — automatic invalidation with no manual versioning.

    Args:
        text: Raw document text (or pre-downloaded content).
        prompt_description: The prompt sent to the LLM.
        examples: Few-shot examples (plain dicts).
        model_id: Primary LLM model identifier.
        temperature: Sampling temperature (``None`` → default).
        passes: Number of extraction passes.
        consensus_providers: Optional list of consensus model IDs.
        consensus_threshold: Consensus similarity threshold.

    Returns:
        64-character hex SHA-256 digest.
    """
    # For very large texts, hash the content first to keep
    # the pre-image compact and avoid memory spikes during
    # concatenation.
    if len(text) > _TEXT_HASH_THRESHOLD:
        text_component = hashlib.sha256(text.encode()).hexdigest()
    else:
        text_component = text

    parts: list[str] = [
        text_component,
        prompt_description,
        _stable_json(examples),
        model_id,
        str(temperature),
        str(passes),
    ]

    if consensus_providers:
        parts.append(_stable_json(sorted(consensus_providers)))
        parts.append(str(consensus_threshold))

    raw = "\x00".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()


# ── Backend protocol ────────────────────────────────────────


class _CacheBackend:
    """Minimal protocol that concrete backends implement."""

    def get(self, key: str) -> dict[str, Any] | None:
        """Retrieve a cached result or ``None``."""
        raise NotImplementedError

    def set(
        self,
        key: str,
        value: dict[str, Any],
        ttl: int,
    ) -> None:
        """Store a result with an expiry TTL in seconds."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources (optional)."""


# ── Redis backend ───────────────────────────────────────────


class _RedisBackend(_CacheBackend):
    """Redis-backed extraction cache using the shared pool.

    Stores results as JSON strings under the
    ``extraction_cache:`` prefix with a configurable TTL.
    """

    def get(self, key: str) -> dict[str, Any] | None:
        """Fetch a cached extraction result from Redis.

        Args:
            key: SHA-256 cache key.

        Returns:
            Parsed result dict or ``None`` on miss.
        """
        from app.core.redis import get_redis_client

        redis_key = f"{REDIS_PREFIX_EXTRACTION_CACHE}{key}"
        client = get_redis_client()
        try:
            raw = client.get(redis_key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception:
            logger.warning(
                "Redis extraction-cache GET failed for %s",
                redis_key,
                exc_info=True,
            )
            return None
        finally:
            client.close()

    def set(
        self,
        key: str,
        value: dict[str, Any],
        ttl: int,
    ) -> None:
        """Write an extraction result to Redis with TTL.

        Args:
            key: SHA-256 cache key.
            value: Extraction result dict (must be JSON-serialisable).
            ttl: Time-to-live in seconds.
        """
        from app.core.redis import get_redis_client

        redis_key = f"{REDIS_PREFIX_EXTRACTION_CACHE}{key}"
        client = get_redis_client()
        try:
            client.setex(redis_key, ttl, json.dumps(value))
        except Exception:
            logger.warning(
                "Redis extraction-cache SET failed for %s",
                redis_key,
                exc_info=True,
            )
        finally:
            client.close()


# ── Disk backend ────────────────────────────────────────────


class _DiskBackend(_CacheBackend):
    """``diskcache``-backed extraction cache for local development.

    Falls back gracefully if ``diskcache`` is not installed.
    Cache directory defaults to ``.extraction_cache/`` in the
    working directory; override with ``EXTRACTION_CACHE_DIR``.
    """

    def __init__(self) -> None:
        try:
            import diskcache
        except ImportError as exc:
            raise ImportError(
                "diskcache is required for the 'disk' cache backend. "
                "Install it with: pip install diskcache"
            ) from exc

        cache_dir = os.environ.get("EXTRACTION_CACHE_DIR", ".extraction_cache")
        self._cache = diskcache.Cache(cache_dir)
        logger.info(
            "Disk extraction cache initialised at %s",
            cache_dir,
        )

    def get(self, key: str) -> dict[str, Any] | None:
        """Fetch a cached extraction result from disk.

        Args:
            key: SHA-256 cache key.

        Returns:
            Parsed result dict or ``None`` on miss.
        """
        prefixed = f"{REDIS_PREFIX_EXTRACTION_CACHE}{key}"
        try:
            raw = self._cache.get(prefixed)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception:
            logger.warning(
                "Disk cache GET failed for %s",
                prefixed,
                exc_info=True,
            )
            return None

    def set(
        self,
        key: str,
        value: dict[str, Any],
        ttl: int,
    ) -> None:
        """Write an extraction result to disk with TTL.

        Args:
            key: SHA-256 cache key.
            value: Extraction result dict.
            ttl: Time-to-live in seconds.
        """
        prefixed = f"{REDIS_PREFIX_EXTRACTION_CACHE}{key}"
        try:
            self._cache.set(
                prefixed,
                json.dumps(value),
                expire=ttl,
            )
        except Exception:
            logger.warning(
                "Disk cache SET failed for %s",
                prefixed,
                exc_info=True,
            )

    def close(self) -> None:
        """Close the underlying diskcache store."""
        with contextlib.suppress(Exception):
            self._cache.close()


# ── Facade ──────────────────────────────────────────────────


class ExtractionCache:
    """Multi-tier extraction-result cache.

    Wraps one of several backends (``redis``, ``disk``, ``none``)
    selected by the ``EXTRACTION_CACHE_BACKEND`` env-var.

    Usage::

        cache = ExtractionCache.instance()
        key = build_cache_key(text, prompt, examples, ...)
        hit = cache.get(key)
        if hit is not None:
            return hit
        result = run_expensive_extraction(...)
        cache.put(key, result)
        return result
    """

    _instance: ExtractionCache | None = None

    def __init__(self, backend: _CacheBackend | None) -> None:
        self._backend = backend
        self._ttl = int(os.environ.get("EXTRACTION_CACHE_TTL", "86400"))
        self._enabled = backend is not None
        logger.info(
            "ExtractionCache initialised (enabled=%s, ttl=%ds)",
            self._enabled,
            self._ttl,
        )

    # ── Singleton ───────────────────────────────────────────

    @classmethod
    def instance(cls) -> ExtractionCache:
        """Return a lazily-initialised singleton.

        Backend is selected via ``EXTRACTION_CACHE_BACKEND``:
        ``redis`` (default), ``disk``, or ``none``.

        Returns:
            The shared ``ExtractionCache`` instance.
        """
        if cls._instance is None:
            backend_name = (
                os.environ.get(
                    "EXTRACTION_CACHE_BACKEND",
                    "redis",
                )
                .lower()
                .strip()
            )
            cache_enabled = os.environ.get(
                "EXTRACTION_CACHE_ENABLED",
                "true",
            ).lower() in ("1", "true", "yes")

            if not cache_enabled or backend_name == "none":
                cls._instance = cls(backend=None)
                logger.info(
                    "Extraction cache disabled (enabled=%s, backend=%s)",
                    cache_enabled,
                    backend_name,
                )
            elif backend_name == "disk":
                cls._instance = cls(backend=_DiskBackend())
            else:
                # Default: redis
                cls._instance = cls(backend=_RedisBackend())
        return cls._instance

    # ── Public API ──────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """Whether the cache is active."""
        return self._enabled

    def get(self, key: str) -> dict[str, Any] | None:
        """Look up a cached extraction result.

        Args:
            key: Cache key (from ``build_cache_key``).

        Returns:
            The cached result dict or ``None`` on miss.
        """
        if not self._enabled:
            return None

        t0 = time.monotonic()
        result = self._backend.get(key)  # type: ignore[union-attr]
        elapsed_ms = (time.monotonic() - t0) * 1000

        if result is not None:
            logger.info(
                "Extraction cache HIT (key=%.12s…, %.1f ms)",
                key,
                elapsed_ms,
            )
        else:
            logger.debug(
                "Extraction cache MISS (key=%.12s…, %.1f ms)",
                key,
                elapsed_ms,
            )
        return result

    def put(
        self,
        key: str,
        value: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """Store an extraction result in the cache.

        Args:
            key: Cache key (from ``build_cache_key``).
            value: Extraction result dict.
            ttl: Override TTL in seconds; defaults to
                ``EXTRACTION_CACHE_TTL``.
        """
        if not self._enabled:
            return

        self._backend.set(  # type: ignore[union-attr]
            key,
            value,
            ttl or self._ttl,
        )
        logger.debug(
            "Extraction cache SET (key=%.12s…, ttl=%ds)",
            key,
            ttl or self._ttl,
        )

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful in tests)."""
        if cls._instance and cls._instance._backend:
            cls._instance._backend.close()
        cls._instance = None
