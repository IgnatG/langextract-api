"""Tests for Pydantic request/response schemas.

Validates:
- ``ExtractionRequest`` field defaults and validation rules
- ``BatchExtractionRequest`` structure
- ``ExtractedEntity`` serialisation
- ``TaskState`` enum values
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    BatchExtractionRequest,
    ExtractedEntity,
    ExtractionMetadata,
    ExtractionRequest,
    ExtractionResult,
    HealthResponse,
    TaskState,
    TaskStatusResponse,
    TaskSubmitResponse,
)

# ── ExtractionRequest ──────────────────────────────────────────────────────


class TestExtractionRequest:
    """Validation tests for ``ExtractionRequest``."""

    def test_valid_with_url(self):
        """A request with only document_url is valid."""
        req = ExtractionRequest(
            document_url="https://example.com/doc.pdf",
        )
        assert req.document_url is not None
        assert req.raw_text is None

    def test_valid_with_raw_text(self):
        """A request with only raw_text is valid."""
        req = ExtractionRequest(raw_text="Some contract text")
        assert req.raw_text == "Some contract text"
        assert req.document_url is None

    def test_valid_with_both_inputs(self):
        """A request with both url and text is valid."""
        req = ExtractionRequest(
            document_url="https://example.com/doc.pdf",
            raw_text="Also this text",
        )
        assert req.document_url is not None
        assert req.raw_text is not None

    def test_fails_without_input(self):
        """A request with neither url nor text raises ValidationError."""
        with pytest.raises(ValidationError, match=r"(?i)at least one"):
            ExtractionRequest(provider="gpt-4o")

    def test_default_provider(self):
        """Default provider is gpt-4o."""
        req = ExtractionRequest(raw_text="test")
        assert req.provider == "gpt-4o"

    def test_default_passes(self):
        """Default passes is 1."""
        req = ExtractionRequest(raw_text="test")
        assert req.passes == 1

    def test_passes_range_min(self):
        """Passes below 1 is invalid."""
        with pytest.raises(ValidationError):
            ExtractionRequest(raw_text="test", passes=0)

    def test_passes_range_max(self):
        """Passes above 5 is invalid."""
        with pytest.raises(ValidationError):
            ExtractionRequest(raw_text="test", passes=6)

    def test_invalid_url_rejected(self):
        """A non-URL string for document_url is rejected."""
        with pytest.raises(ValidationError):
            ExtractionRequest(document_url="not-a-url")

    def test_invalid_callback_url_rejected(self):
        """A non-URL string for callback_url is rejected."""
        with pytest.raises(ValidationError):
            ExtractionRequest(
                raw_text="test",
                callback_url="not-a-url",
            )

    def test_extraction_config_default(self):
        """Default extraction_config is an empty dict."""
        req = ExtractionRequest(raw_text="test")
        assert req.extraction_config == {}

    def test_extraction_config_accepts_custom(self):
        """Custom extraction_config is preserved."""
        cfg = {"prompt_description": "Custom", "temperature": 0.5}
        req = ExtractionRequest(raw_text="test", extraction_config=cfg)
        assert req.extraction_config == cfg


# ── BatchExtractionRequest ─────────────────────────────────────────────────


class TestBatchExtractionRequest:
    """Validation tests for ``BatchExtractionRequest``."""

    def test_valid_batch(self):
        """A batch with at least one document is valid."""
        req = BatchExtractionRequest(
            batch_id="batch-001",
            documents=[
                ExtractionRequest(raw_text="Doc A"),
            ],
        )
        assert req.batch_id == "batch-001"
        assert len(req.documents) == 1

    def test_empty_documents_rejected(self):
        """A batch with zero documents is invalid."""
        with pytest.raises(ValidationError):
            BatchExtractionRequest(
                batch_id="batch-002",
                documents=[],
            )

    def test_batch_callback_url(self):
        """Batch-level callback_url is accepted."""
        req = BatchExtractionRequest(
            batch_id="batch-003",
            documents=[ExtractionRequest(raw_text="A")],
            callback_url="https://hook.example.com",
        )
        assert req.callback_url is not None


# ── ExtractedEntity ────────────────────────────────────────────────────────


class TestExtractedEntity:
    """Tests for ``ExtractedEntity`` serialisation."""

    def test_full_entity(self):
        """Entity with all fields populated."""
        entity = ExtractedEntity(
            extraction_class="party",
            extraction_text="Acme Corp",
            attributes={"role": "Buyer"},
            char_start=10,
            char_end=19,
        )
        d = entity.model_dump()
        assert d["extraction_class"] == "party"
        assert d["extraction_text"] == "Acme Corp"
        assert d["attributes"] == {"role": "Buyer"}
        assert d["char_start"] == 10
        assert d["char_end"] == 19

    def test_defaults(self):
        """Optional fields default correctly."""
        entity = ExtractedEntity(
            extraction_class="date",
            extraction_text="Jan 1",
        )
        assert entity.attributes == {}
        assert entity.char_start is None
        assert entity.char_end is None


# ── ExtractionResult ───────────────────────────────────────────────────────


class TestExtractionResult:
    """Tests for ``ExtractionResult`` model."""

    def test_empty_result(self):
        """Empty result has no entities."""
        result = ExtractionResult(
            metadata=ExtractionMetadata(provider="gpt-4o"),
        )
        assert result.entities == []
        assert result.metadata.tokens_used == 0

    def test_with_entities(self):
        """Result with entities serialises correctly."""
        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    extraction_class="party",
                    extraction_text="Corp",
                ),
            ],
            metadata=ExtractionMetadata(
                provider="gpt-4o",
                tokens_used=100,
                processing_time_ms=500,
            ),
        )
        d = result.model_dump()
        assert len(d["entities"]) == 1
        assert d["metadata"]["tokens_used"] == 100


# ── TaskState ──────────────────────────────────────────────────────────────


class TestTaskState:
    """Tests for ``TaskState`` enum."""

    def test_all_states_defined(self):
        """Ensure all expected states exist."""
        expected = {
            "PENDING",
            "STARTED",
            "PROGRESS",
            "SUCCESS",
            "FAILURE",
            "REVOKED",
            "RETRY",
        }
        assert set(TaskState.__members__) == expected

    def test_state_values_are_strings(self):
        """State values are uppercase string names."""
        for state in TaskState:
            assert state.value == state.name


# ── Response models ────────────────────────────────────────────────────────


class TestResponseModels:
    """Smoke tests for response model construction."""

    def test_task_submit_response(self):
        """TaskSubmitResponse defaults are sensible."""
        resp = TaskSubmitResponse(task_id="abc-123")
        assert resp.status == "submitted"
        assert resp.message == "Task submitted successfully"

    def test_task_status_response(self):
        """TaskStatusResponse can represent each state."""
        resp = TaskStatusResponse(
            task_id="abc",
            state=TaskState.SUCCESS,
            result={"entities": []},
        )
        assert resp.state == TaskState.SUCCESS
        assert resp.result is not None

    def test_health_response(self):
        """HealthResponse requires status and version."""
        resp = HealthResponse(status="ok", version="0.1.0")
        assert resp.status == "ok"
