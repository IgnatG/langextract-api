"""
Redis connection management.

Provides a shared ``ConnectionPool`` and a convenience factory
for ``redis.Redis`` clients.  FastAPI-specific dependency
injection (``Depends(get_redis)``) lives in ``app.api.deps``.
"""

from __future__ import annotations

import redis

from app.core.config import get_settings

# ── Global Redis connection pool ────────────────────────

_redis_pool: redis.ConnectionPool | None = None


def get_redis_pool() -> redis.ConnectionPool:
    """Return a module-level Redis ``ConnectionPool``.

    Reusing a single pool avoids the overhead of creating
    and tearing down connections per request.

    Returns:
        A shared ``ConnectionPool`` instance.
    """
    global _redis_pool
    if _redis_pool is None:
        settings = get_settings()
        _redis_pool = redis.ConnectionPool.from_url(
            settings.REDIS_URL,
            decode_responses=True,
        )
    return _redis_pool


def get_redis_client() -> redis.Redis:
    """Return a Redis client for non-dependency use.

    Used by Celery tasks and router helpers that cannot rely
    on FastAPI ``Depends()``.  The caller is responsible for
    calling ``client.close()`` when finished.

    Returns:
        A ``redis.Redis`` instance on the shared pool.
    """
    return redis.Redis(connection_pool=get_redis_pool())
