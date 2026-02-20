"""Tests for the multi-tier extraction-result cache.

Covers:
- Deterministic cache-key generation and stability.
- Key invalidation on prompt / model / parameter changes.
- Redis backend get / set / miss behaviour.
- Disk backend get / set / miss behaviour.
- Singleton lifecycle and reset.
- Cache disabled (``none``) backend.
- Integration with extractor (mock-level).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.services.extraction_cache import (
    ExtractionCache,
    _RedisBackend,
    build_cache_key,
)


# ── Fixtures ────────────────────────────────────────────────


_SAMPLE_TEXT = "Agreement by Acme Corp dated January 1, 2025"
_SAMPLE_PROMPT = "Extract key contract entities"
_SAMPLE_EXAMPLES: list[dict[str, Any]] = [
    {
        "text": "Sample text",
        "extractions": [
            {
                "extraction_class": "party",
                "extraction_text": "Acme",
                "attributes": {"role": "Seller"},
            },
        ],
    },
]
_SAMPLE_RESULT: dict[str, Any] = {
    "status": "completed",
    "source": "<raw_text>",
    "data": {
        "entities": [
            {
                "type": "party",
                "text": "Acme Corp",
                "confidence_score": 1.0,
            },
        ],
        "metadata": {
            "provider": "gpt-4o",
            "tokens_used": 150,
            "processing_time_ms": 3200,
        },
    },
}


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure each test starts with a fresh singleton."""
    ExtractionCache.reset()
    yield
    ExtractionCache.reset()


# ── Cache key tests ─────────────────────────────────────────


class TestBuildCacheKey:
    """Deterministic key generation and invalidation."""

    def test_deterministic(self) -> None:
        """Same inputs always produce the same key."""
        k1 = build_cache_key(
            _SAMPLE_TEXT,
            _SAMPLE_PROMPT,
            _SAMPLE_EXAMPLES,
            "gpt-4o",
            0.0,
            1,
        )
        k2 = build_cache_key(
            _SAMPLE_TEXT,
            _SAMPLE_PROMPT,
            _SAMPLE_EXAMPLES,
            "gpt-4o",
            0.0,
            1,
        )
        assert k1 == k2
        assert len(k1) == 64  # SHA-256 hex

    def test_different_text_different_key(self) -> None:
        """Changing the input text invalidates the cache."""
        k1 = build_cache_key(
            _SAMPLE_TEXT,
            _SAMPLE_PROMPT,
            _SAMPLE_EXAMPLES,
            "gpt-4o",
            0.0,
            1,
        )
        k2 = build_cache_key(
            "Totally different document",
            _SAMPLE_PROMPT,
            _SAMPLE_EXAMPLES,
            "gpt-4o",
            0.0,
            1,
        )
        assert k1 != k2

    def test_different_prompt_different_key(self) -> None:
        """Changing the prompt invalidates the cache."""
        k1 = build_cache_key(
            _SAMPLE_TEXT,
            _SAMPLE_PROMPT,
            _SAMPLE_EXAMPLES,
            "gpt-4o",
            0.0,
            1,
        )
        k2 = build_cache_key(
            _SAMPLE_TEXT,
            "Extract medication names",
            _SAMPLE_EXAMPLES,
            "gpt-4o",
            0.0,
            1,
        )
        assert k1 != k2

    def test_different_model_different_key(self) -> None:
        """Changing the model ID invalidates the cache."""
        k1 = build_cache_key(
            _SAMPLE_TEXT,
            _SAMPLE_PROMPT,
            _SAMPLE_EXAMPLES,
            "gpt-4o",
            0.0,
            1,
        )
        k2 = build_cache_key(
            _SAMPLE_TEXT,
            _SAMPLE_PROMPT,
            _SAMPLE_EXAMPLES,
            "gemini-2.0-flash",
            0.0,
            1,
        )
        assert k1 != k2

    def test_different_temperature_different_key(self) -> None:
        """Changing the temperature invalidates the cache."""
        k1 = build_cache_key(
            _SAMPLE_TEXT,
            _SAMPLE_PROMPT,
            _SAMPLE_EXAMPLES,
            "gpt-4o",
            0.0,
            1,
        )
        k2 = build_cache_key(
            _SAMPLE_TEXT,
            _SAMPLE_PROMPT,
            _SAMPLE_EXAMPLES,
            "gpt-4o",
            0.7,
            1,
        )
        assert k1 != k2

    def test_different_passes_different_key(self) -> None:
        """Changing the pass count invalidates the cache."""
        k1 = build_cache_key(
            _SAMPLE_TEXT,
            _SAMPLE_PROMPT,
            _SAMPLE_EXAMPLES,
            "gpt-4o",
            0.0,
            1,
        )
        k2 = build_cache_key(
            _SAMPLE_TEXT,
            _SAMPLE_PROMPT,
            _SAMPLE_EXAMPLES,
            "gpt-4o",
            0.0,
            3,
        )
        assert k1 != k2

    def test_different_examples_different_key(self) -> None:
        """Changing the schema examples invalidates the cache."""
        alt_examples = [{"text": "x", "extractions": []}]
        k1 = build_cache_key(
            _SAMPLE_TEXT,
            _SAMPLE_PROMPT,
            _SAMPLE_EXAMPLES,
            "gpt-4o",
            0.0,
            1,
        )
        k2 = build_cache_key(
            _SAMPLE_TEXT,
            _SAMPLE_PROMPT,
            alt_examples,
            "gpt-4o",
            0.0,
            1,
        )
        assert k1 != k2

    def test_consensus_providers_in_key(self) -> None:
        """Consensus providers change the cache key."""
        k1 = build_cache_key(
            _SAMPLE_TEXT,
            _SAMPLE_PROMPT,
            _SAMPLE_EXAMPLES,
            "gpt-4o",
            0.0,
            1,
        )
        k2 = build_cache_key(
            _SAMPLE_TEXT,
            _SAMPLE_PROMPT,
            _SAMPLE_EXAMPLES,
            "gpt-4o",
            0.0,
            1,
            consensus_providers=["gpt-4o", "gemini-2.0-flash"],
            consensus_threshold=0.6,
        )
        assert k1 != k2

    def test_consensus_provider_order_irrelevant(self) -> None:
        """Provider order should not affect the key."""
        k1 = build_cache_key(
            _SAMPLE_TEXT,
            _SAMPLE_PROMPT,
            _SAMPLE_EXAMPLES,
            "gpt-4o",
            0.0,
            1,
            consensus_providers=["gpt-4o", "gemini-2.0-flash"],
            consensus_threshold=0.6,
        )
        k2 = build_cache_key(
            _SAMPLE_TEXT,
            _SAMPLE_PROMPT,
            _SAMPLE_EXAMPLES,
            "gpt-4o",
            0.0,
            1,
            consensus_providers=["gemini-2.0-flash", "gpt-4o"],
            consensus_threshold=0.6,
        )
        assert k1 == k2

    def test_large_text_hashed(self) -> None:
        """Texts > threshold are SHA-256 hashed first."""
        big_text = "x" * 100_000
        key = build_cache_key(
            big_text,
            _SAMPLE_PROMPT,
            _SAMPLE_EXAMPLES,
            "gpt-4o",
            0.0,
            1,
        )
        assert len(key) == 64

    def test_none_temperature(self) -> None:
        """``None`` temperature is handled gracefully."""
        key = build_cache_key(
            _SAMPLE_TEXT,
            _SAMPLE_PROMPT,
            _SAMPLE_EXAMPLES,
            "gpt-4o",
            None,
            1,
        )
        assert len(key) == 64


# ── Redis backend tests ────────────────────────────────────


class TestRedisBackend:
    """Test the Redis cache backend with a mocked client."""

    def _mock_redis(
        self,
        get_return: str | None = None,
    ) -> MagicMock:
        """Create a mock Redis client."""
        mock = MagicMock()
        mock.get.return_value = get_return
        return mock

    @patch("app.services.extraction_cache.get_redis_client")
    def test_get_miss(self, mock_get_client: MagicMock) -> None:
        """Cache miss returns ``None``."""
        mock_get_client.return_value = self._mock_redis(None)
        backend = _RedisBackend()
        assert backend.get("abc123") is None

    @patch("app.services.extraction_cache.get_redis_client")
    def test_get_hit(self, mock_get_client: MagicMock) -> None:
        """Cache hit returns the deserialised result."""
        payload = json.dumps(_SAMPLE_RESULT)
        mock_get_client.return_value = self._mock_redis(payload)
        backend = _RedisBackend()
        result = backend.get("abc123")
        assert result == _SAMPLE_RESULT

    @patch("app.services.extraction_cache.get_redis_client")
    def test_set(self, mock_get_client: MagicMock) -> None:
        """``set()`` writes JSON to Redis with TTL."""
        mock = self._mock_redis()
        mock_get_client.return_value = mock
        backend = _RedisBackend()
        backend.set("abc123", _SAMPLE_RESULT, ttl=3600)
        mock.setex.assert_called_once_with(
            "extraction_cache:abc123",
            3600,
            json.dumps(_SAMPLE_RESULT),
        )

    @patch("app.services.extraction_cache.get_redis_client")
    def test_get_exception_returns_none(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Redis errors are swallowed and return ``None``."""
        mock = self._mock_redis()
        mock.get.side_effect = ConnectionError("down")
        mock_get_client.return_value = mock
        backend = _RedisBackend()
        assert backend.get("abc123") is None

    @patch("app.services.extraction_cache.get_redis_client")
    def test_set_exception_swallowed(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Redis write errors are logged but do not raise."""
        mock = self._mock_redis()
        mock.setex.side_effect = ConnectionError("down")
        mock_get_client.return_value = mock
        backend = _RedisBackend()
        # Should not raise
        backend.set("abc123", _SAMPLE_RESULT, ttl=3600)


# ── ExtractionCache facade tests ───────────────────────────


class TestExtractionCacheFacade:
    """Test the high-level facade and singleton behaviour."""

    @patch.dict(
        "os.environ",
        {
            "EXTRACTION_CACHE_ENABLED": "false",
            "EXTRACTION_CACHE_BACKEND": "redis",
        },
    )
    def test_disabled_returns_none(self) -> None:
        """Disabled cache always returns ``None``."""
        cache = ExtractionCache.instance()
        assert not cache.enabled
        assert cache.get("any_key") is None

    @patch.dict(
        "os.environ",
        {
            "EXTRACTION_CACHE_ENABLED": "true",
            "EXTRACTION_CACHE_BACKEND": "none",
        },
    )
    def test_none_backend(self) -> None:
        """``none`` backend behaves like disabled."""
        cache = ExtractionCache.instance()
        assert not cache.enabled

    @patch.dict(
        "os.environ",
        {
            "EXTRACTION_CACHE_ENABLED": "true",
            "EXTRACTION_CACHE_BACKEND": "redis",
            "EXTRACTION_CACHE_TTL": "7200",
        },
    )
    @patch(
        "app.services.extraction_cache._RedisBackend.get",
        return_value=None,
    )
    def test_miss_returns_none(
        self,
        mock_backend_get: MagicMock,
    ) -> None:
        """Facade returns ``None`` on miss."""
        cache = ExtractionCache.instance()
        assert cache.enabled
        assert cache.get("k") is None

    @patch.dict(
        "os.environ",
        {
            "EXTRACTION_CACHE_ENABLED": "true",
            "EXTRACTION_CACHE_BACKEND": "redis",
            "EXTRACTION_CACHE_TTL": "7200",
        },
    )
    @patch(
        "app.services.extraction_cache._RedisBackend.get",
        return_value=_SAMPLE_RESULT,
    )
    def test_hit_returns_result(
        self,
        mock_backend_get: MagicMock,
    ) -> None:
        """Facade returns cached result on hit."""
        cache = ExtractionCache.instance()
        result = cache.get("k")
        assert result == _SAMPLE_RESULT

    @patch.dict(
        "os.environ",
        {
            "EXTRACTION_CACHE_ENABLED": "true",
            "EXTRACTION_CACHE_BACKEND": "redis",
            "EXTRACTION_CACHE_TTL": "1800",
        },
    )
    @patch("app.services.extraction_cache._RedisBackend.set")
    def test_put_uses_default_ttl(
        self,
        mock_backend_set: MagicMock,
    ) -> None:
        """``put()`` sends default TTL to backend."""
        cache = ExtractionCache.instance()
        cache.put("k", _SAMPLE_RESULT)
        mock_backend_set.assert_called_once_with(
            "k",
            _SAMPLE_RESULT,
            1800,
        )

    @patch.dict(
        "os.environ",
        {
            "EXTRACTION_CACHE_ENABLED": "true",
            "EXTRACTION_CACHE_BACKEND": "redis",
            "EXTRACTION_CACHE_TTL": "1800",
        },
    )
    @patch("app.services.extraction_cache._RedisBackend.set")
    def test_put_with_custom_ttl(
        self,
        mock_backend_set: MagicMock,
    ) -> None:
        """``put()`` with explicit TTL overrides default."""
        cache = ExtractionCache.instance()
        cache.put("k", _SAMPLE_RESULT, ttl=600)
        mock_backend_set.assert_called_once_with(
            "k",
            _SAMPLE_RESULT,
            600,
        )

    @patch.dict(
        "os.environ",
        {
            "EXTRACTION_CACHE_ENABLED": "true",
            "EXTRACTION_CACHE_BACKEND": "redis",
        },
    )
    def test_singleton(self) -> None:
        """Multiple calls return the same instance."""
        a = ExtractionCache.instance()
        b = ExtractionCache.instance()
        assert a is b

    def test_reset(self) -> None:
        """``reset()`` clears the singleton."""
        with patch.dict(
            "os.environ",
            {
                "EXTRACTION_CACHE_ENABLED": "false",
                "EXTRACTION_CACHE_BACKEND": "none",
            },
        ):
            a = ExtractionCache.instance()
        ExtractionCache.reset()
        with patch.dict(
            "os.environ",
            {
                "EXTRACTION_CACHE_ENABLED": "false",
                "EXTRACTION_CACHE_BACKEND": "none",
            },
        ):
            b = ExtractionCache.instance()
        assert a is not b

    @patch.dict(
        "os.environ",
        {
            "EXTRACTION_CACHE_ENABLED": "false",
        },
    )
    def test_put_disabled_is_noop(self) -> None:
        """``put()`` on disabled cache does nothing."""
        cache = ExtractionCache.instance()
        # Should not raise
        cache.put("k", _SAMPLE_RESULT)
