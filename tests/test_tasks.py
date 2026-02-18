"""Unit tests for task helpers and integration tests for extraction tasks.

Tests cover:
- ``_build_examples()`` — dict → ExampleData conversion
- ``_resolve_api_key()`` — API key selection logic
- ``_is_openai_model()`` — OpenAI model detection
- ``_convert_extractions()`` — AnnotatedDocument → dict conversion
- ``_fire_webhook()`` — webhook delivery
- ``_run_extraction()`` — full extraction pipeline (mocked lx.extract)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import (
    FakeAnnotatedDocument,
    FakeCharInterval,
    FakeExtraction,
)

# ── _build_examples ─────────────────────────────────────────────────────────


class TestBuildExamples:
    """Tests for the ``_build_examples`` helper."""

    def test_converts_single_example(self):
        """A single example dict is converted to an ExampleData list."""
        from app.tasks import _build_examples

        raw = [
            {
                "text": "Contract text here",
                "extractions": [
                    {
                        "extraction_class": "party",
                        "extraction_text": "Acme",
                        "attributes": {"role": "Buyer"},
                    },
                ],
            },
        ]
        result = _build_examples(raw)

        assert len(result) == 1
        assert result[0].text == "Contract text here"
        assert len(result[0].extractions) == 1
        assert result[0].extractions[0].extraction_class == "party"
        assert result[0].extractions[0].extraction_text == "Acme"

    def test_converts_multiple_examples(self):
        """Multiple example dicts produce multiple ExampleData objects."""
        from app.tasks import _build_examples

        raw = [
            {"text": "First", "extractions": []},
            {"text": "Second", "extractions": []},
        ]
        result = _build_examples(raw)
        assert len(result) == 2
        assert result[0].text == "First"
        assert result[1].text == "Second"

    def test_empty_list_returns_empty(self):
        """An empty input list returns an empty output list."""
        from app.tasks import _build_examples

        assert _build_examples([]) == []

    def test_missing_extractions_key_defaults_to_empty(self):
        """If 'extractions' key is absent, default to an empty list."""
        from app.tasks import _build_examples

        raw = [{"text": "No extractions key here"}]
        result = _build_examples(raw)
        assert len(result) == 1
        assert result[0].extractions == []

    def test_attributes_are_optional(self):
        """Extraction dicts without 'attributes' should still work."""
        from app.tasks import _build_examples

        raw = [
            {
                "text": "Test",
                "extractions": [
                    {
                        "extraction_class": "date",
                        "extraction_text": "Jan 1",
                    },
                ],
            },
        ]
        result = _build_examples(raw)
        assert result[0].extractions[0].attributes is None


# ── _resolve_api_key ────────────────────────────────────────────────────────


class TestResolveApiKey:
    """Tests for the ``_resolve_api_key`` helper."""

    def test_returns_openai_key_for_gpt_models(self, mock_settings):
        """GPT model names should resolve to OPENAI_API_KEY."""
        from app.tasks import _resolve_api_key

        with patch("app.tasks.get_settings", return_value=mock_settings):
            assert _resolve_api_key("gpt-4o") == "test-openai-key"
            assert _resolve_api_key("GPT-4-turbo") == "test-openai-key"
            assert _resolve_api_key("openai/gpt-4o") == "test-openai-key"

    def test_returns_langextract_key_for_gemini(self, mock_settings):
        """Gemini models should prefer LANGEXTRACT_API_KEY."""
        from app.tasks import _resolve_api_key

        mock_settings.LANGEXTRACT_API_KEY = "lx-key"
        with patch("app.tasks.get_settings", return_value=mock_settings):
            assert _resolve_api_key("gemini-2.5-flash") == "lx-key"

    def test_falls_back_to_gemini_key(self, mock_settings):
        """If no LANGEXTRACT_API_KEY, fall back to GEMINI_API_KEY."""
        from app.tasks import _resolve_api_key

        mock_settings.LANGEXTRACT_API_KEY = ""
        with patch("app.tasks.get_settings", return_value=mock_settings):
            assert _resolve_api_key("gemini-2.5-flash") == "test-gemini-key"

    def test_returns_none_when_no_keys(self, mock_settings):
        """If no keys are configured, return None."""
        from app.tasks import _resolve_api_key

        mock_settings.LANGEXTRACT_API_KEY = ""
        mock_settings.GEMINI_API_KEY = ""
        mock_settings.OPENAI_API_KEY = ""
        with patch("app.tasks.get_settings", return_value=mock_settings):
            assert _resolve_api_key("gemini-2.5-flash") is None
            assert _resolve_api_key("gpt-4o") is None


# ── _is_openai_model ────────────────────────────────────────────────────────


class TestIsOpenaiModel:
    """Tests for the ``_is_openai_model`` helper."""

    def test_detects_gpt_models(self):
        """Model IDs containing 'gpt' are OpenAI."""
        from app.tasks import _is_openai_model

        assert _is_openai_model("gpt-4o") is True
        assert _is_openai_model("GPT-4-turbo") is True
        assert _is_openai_model("gpt-4o-mini") is True

    def test_detects_openai_prefix(self):
        """Model IDs containing 'openai' are OpenAI."""
        from app.tasks import _is_openai_model

        assert _is_openai_model("openai/gpt-4o") is True

    def test_rejects_non_openai_models(self):
        """Gemini and other models are not OpenAI."""
        from app.tasks import _is_openai_model

        assert _is_openai_model("gemini-2.5-flash") is False
        assert _is_openai_model("claude-3-opus") is False
        assert _is_openai_model("llama-3") is False


# ── _convert_extractions ────────────────────────────────────────────────────


class TestConvertExtractions:
    """Tests for the ``_convert_extractions`` helper."""

    def test_converts_annotated_document(self, fake_annotated_document):
        """A populated AnnotatedDocument produces the expected dicts."""
        from app.tasks import _convert_extractions

        entities = _convert_extractions(fake_annotated_document)

        assert len(entities) == 2

        party = entities[0]
        assert party["extraction_class"] == "party"
        assert party["extraction_text"] == "Acme Corp"
        assert party["attributes"] == {"role": "Seller"}
        assert party["char_start"] == 14
        assert party["char_end"] == 23

        date = entities[1]
        assert date["extraction_class"] == "date"
        assert date["extraction_text"] == "January 1, 2025"

    def test_empty_document_returns_empty_list(self):
        """An AnnotatedDocument with no extractions returns []."""
        from app.tasks import _convert_extractions

        empty_doc = FakeAnnotatedDocument(text="nothing", extractions=[])
        assert _convert_extractions(empty_doc) == []

    def test_handles_none_extractions(self):
        """If extractions is None, return an empty list."""
        from app.tasks import _convert_extractions

        doc = FakeAnnotatedDocument(text="test", extractions=None)
        assert _convert_extractions(doc) == []

    def test_handles_missing_char_interval(self):
        """Extractions without char_interval get None offsets."""
        from app.tasks import _convert_extractions

        doc = FakeAnnotatedDocument(
            text="test",
            extractions=[
                FakeExtraction(
                    extraction_class="term",
                    extraction_text="30 days",
                    char_interval=None,
                ),
            ],
        )
        entities = _convert_extractions(doc)
        assert entities[0]["char_start"] is None
        assert entities[0]["char_end"] is None

    def test_handles_none_attributes(self):
        """Extractions with attributes=None get an empty dict."""
        from app.tasks import _convert_extractions

        doc = FakeAnnotatedDocument(
            text="test",
            extractions=[
                FakeExtraction(
                    extraction_class="party",
                    extraction_text="Corp",
                    attributes=None,
                    char_interval=FakeCharInterval(0, 4),
                ),
            ],
        )
        entities = _convert_extractions(doc)
        assert entities[0]["attributes"] == {}


# ── _fire_webhook ───────────────────────────────────────────────────────────


class TestFireWebhook:
    """Tests for the ``_fire_webhook`` helper."""

    @patch("app.tasks.httpx.Client")
    def test_successful_delivery(self, mock_client_cls):
        """Webhook is delivered via POST with JSON payload."""
        from app.tasks import _fire_webhook

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value.__enter__ = lambda s: mock_client
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        _fire_webhook("https://example.com/hook", {"task_id": "abc"})

        mock_client.post.assert_called_once_with(
            "https://example.com/hook",
            json={"task_id": "abc"},
        )
        mock_resp.raise_for_status.assert_called_once()

    @patch("app.tasks.httpx.Client")
    def test_failure_does_not_raise(self, mock_client_cls):
        """Webhook failures are logged but never re-raised."""
        from app.tasks import _fire_webhook

        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("Connection refused")
        mock_client_cls.return_value.__enter__ = lambda s: mock_client
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        # Should not raise
        _fire_webhook("https://example.com/hook", {"task_id": "abc"})


# ── _run_extraction (integration test with mocked lx.extract) ──────────────


class TestRunExtraction:
    """Integration tests for ``_run_extraction`` with mocked LangExtract."""

    def test_returns_completed_result(
        self,
        mock_settings,
        mock_lx_extract,
    ):
        """Successful extraction returns a result dict with entities."""
        from app.tasks import _run_extraction

        with patch("app.tasks.get_settings", return_value=mock_settings):
            result = _run_extraction(
                task_self=None,
                raw_text="Agreement by Acme Corp dated January 1, 2025",
                provider="gpt-4o",
                passes=1,
            )

        assert result["status"] == "completed"
        assert result["source"] == "<raw_text>"
        assert len(result["data"]["entities"]) == 2
        assert result["data"]["metadata"]["provider"] == "gpt-4o"
        assert result["data"]["metadata"]["processing_time_ms"] >= 0

    def test_uses_default_prompt_and_examples(
        self,
        mock_settings,
        mock_lx_extract,
    ):
        """Without extraction_config, defaults are used."""
        from app.extraction_defaults import (
            DEFAULT_PROMPT_DESCRIPTION,
        )
        from app.tasks import _run_extraction

        with patch("app.tasks.get_settings", return_value=mock_settings):
            _run_extraction(
                task_self=None,
                raw_text="some text",
            )

        call_kwargs = mock_lx_extract.call_args.kwargs
        assert call_kwargs["prompt_description"] == DEFAULT_PROMPT_DESCRIPTION
        assert call_kwargs["text_or_documents"] == "some text"

    def test_custom_extraction_config(
        self,
        mock_settings,
        mock_lx_extract,
    ):
        """Custom prompt, temperature, etc. are forwarded to lx.extract."""
        from app.tasks import _run_extraction

        cfg: dict[str, Any] = {
            "prompt_description": "Custom prompt",
            "examples": [{"text": "x", "extractions": []}],
            "temperature": 0.5,
            "additional_context": "Extra info",
            "max_workers": 5,
        }

        with patch("app.tasks.get_settings", return_value=mock_settings):
            _run_extraction(
                task_self=None,
                raw_text="test",
                extraction_config=cfg,
            )

        call_kwargs = mock_lx_extract.call_args.kwargs
        assert call_kwargs["prompt_description"] == "Custom prompt"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["additional_context"] == "Extra info"
        assert call_kwargs["max_workers"] == 5

    def test_openai_flags_set_for_gpt(
        self,
        mock_settings,
        mock_lx_extract,
    ):
        """GPT models get fence_output=True, use_schema_constraints=False."""
        from app.tasks import _run_extraction

        with patch("app.tasks.get_settings", return_value=mock_settings):
            _run_extraction(
                task_self=None,
                raw_text="test",
                provider="gpt-4o",
            )

        call_kwargs = mock_lx_extract.call_args.kwargs
        assert call_kwargs["fence_output"] is True
        assert call_kwargs["use_schema_constraints"] is False

    def test_gemini_does_not_set_openai_flags(
        self,
        mock_settings,
        mock_lx_extract,
    ):
        """Gemini models do NOT get OpenAI-specific flags."""
        from app.tasks import _run_extraction

        with patch("app.tasks.get_settings", return_value=mock_settings):
            _run_extraction(
                task_self=None,
                raw_text="test",
                provider="gemini-2.5-flash",
            )

        call_kwargs = mock_lx_extract.call_args.kwargs
        assert "fence_output" not in call_kwargs
        assert "use_schema_constraints" not in call_kwargs

    def test_document_url_preferred_over_raw_text(
        self,
        mock_settings,
        mock_lx_extract,
    ):
        """When document_url is provided, it is used as the input."""
        from app.tasks import _run_extraction

        with patch("app.tasks.get_settings", return_value=mock_settings):
            result = _run_extraction(
                task_self=None,
                document_url="https://example.com/doc.pdf",
                raw_text="fallback text",
            )

        call_kwargs = mock_lx_extract.call_args.kwargs
        assert call_kwargs["text_or_documents"] == "https://example.com/doc.pdf"
        assert result["source"] == "https://example.com/doc.pdf"

    def test_progress_updates_with_task_self(
        self,
        mock_settings,
        mock_lx_extract,
    ):
        """When task_self is provided, update_state is called."""
        from app.tasks import _run_extraction

        mock_task = MagicMock()
        with patch("app.tasks.get_settings", return_value=mock_settings):
            _run_extraction(
                task_self=mock_task,
                raw_text="test",
            )

        # At least 3 progress updates: preparing, extracting, post_processing
        assert mock_task.update_state.call_count >= 3
        steps = [
            c.kwargs["meta"]["step"] for c in mock_task.update_state.call_args_list
        ]
        assert "preparing" in steps
        assert "extracting" in steps
        assert "post_processing" in steps

    def test_handles_list_result_from_lx(self, mock_settings):
        """If lx.extract returns a list, first element is used."""
        from app.tasks import _run_extraction

        doc = FakeAnnotatedDocument(
            text="test",
            extractions=[
                FakeExtraction(
                    extraction_class="party",
                    extraction_text="Corp",
                ),
            ],
        )
        with (
            patch("app.tasks.get_settings", return_value=mock_settings),
            patch("app.tasks.lx.extract", return_value=[doc]),
        ):
            result = _run_extraction(task_self=None, raw_text="test")

        assert len(result["data"]["entities"]) == 1

    def test_handles_empty_list_result(self, mock_settings):
        """If lx.extract returns an empty list, no entities are extracted."""
        from app.tasks import _run_extraction

        with (
            patch("app.tasks.get_settings", return_value=mock_settings),
            patch("app.tasks.lx.extract", return_value=[]),
            patch("app.tasks.lx.data.AnnotatedDocument") as mock_ad,
        ):
            mock_ad.return_value = FakeAnnotatedDocument()
            result = _run_extraction(task_self=None, raw_text="test")

        assert result["data"]["entities"] == []


# ── extract_document task ──────────────────────────────────────────────────
#
# bind=True Celery tasks inject the task instance as ``self``.  When
# calling the task directly in tests (``extract_document(...)``), Celery's
# ``__call__`` pushes a *new* request context, so any ``mock_self`` passed
# as the first positional arg ends up as ``document_url`` instead.
#
# The correct testing approach: push a fake request context with a known
# ``id``, mock ``_run_extraction`` (already tested above), and call
# ``task.run(...)`` directly.
# ────────────────────────────────────────────────────────────────────────────


class TestExtractDocumentTask:
    """Tests for the ``extract_document`` Celery task."""

    def test_calls_run_extraction(self, mock_settings):
        """The task delegates to _run_extraction."""
        from app.tasks import extract_document

        mock_result = {
            "status": "completed",
            "source": "<raw_text>",
            "data": {"entities": []},
        }

        extract_document.push_request(id="task-id-123")
        try:
            with (
                patch(
                    "app.tasks._run_extraction",
                    return_value=mock_result,
                ) as mock_run,
                patch(
                    "app.tasks.get_settings",
                    return_value=mock_settings,
                ),
            ):
                result = extract_document.run(
                    raw_text="test contract text",
                    provider="gpt-4o",
                )
        finally:
            extract_document.pop_request()

        mock_run.assert_called_once()
        assert result["status"] == "completed"

    def test_fires_webhook_on_success(self, mock_settings):
        """Webhook is triggered when callback_url is provided."""
        from app.tasks import extract_document

        mock_result = {
            "status": "completed",
            "source": "<raw_text>",
            "data": {"entities": []},
        }

        extract_document.push_request(id="task-id-456")
        try:
            with (
                patch(
                    "app.tasks._run_extraction",
                    return_value=mock_result,
                ),
                patch("app.tasks._fire_webhook") as mock_webhook,
            ):
                extract_document.run(
                    raw_text="test",
                    callback_url="https://hook.example.com/done",
                )
        finally:
            extract_document.pop_request()

        mock_webhook.assert_called_once()
        webhook_url = mock_webhook.call_args[0][0]
        assert webhook_url == "https://hook.example.com/done"

    def test_retries_on_failure(self, mock_settings):
        """The task retries on exception (called_directly → re-raises)."""
        from app.tasks import extract_document

        extract_document.push_request(id="task-id-789")
        try:
            with (
                patch(
                    "app.tasks._run_extraction",
                    side_effect=RuntimeError("API error"),
                ),
                pytest.raises(RuntimeError, match="API error"),
            ):
                extract_document.run(raw_text="test")
        finally:
            extract_document.pop_request()


# ── extract_batch task ──────────────────────────────────────────────────────


class TestExtractBatchTask:
    """Tests for the ``extract_batch`` Celery task."""

    def test_processes_all_documents(self, mock_settings):
        """All documents in the batch are processed."""
        from app.tasks import extract_batch

        mock_result = {
            "status": "completed",
            "source": "<raw_text>",
            "data": {"entities": []},
        }

        docs = [
            {"raw_text": "Doc A"},
            {"raw_text": "Doc B"},
            {"raw_text": "Doc C"},
        ]

        extract_batch.push_request(id="batch-task-1")
        try:
            with (
                patch(
                    "app.tasks._run_extraction",
                    return_value=mock_result,
                ),
                patch("celery.app.task.Task.update_state"),
            ):
                result = extract_batch.run("batch-001", docs)
        finally:
            extract_batch.pop_request()

        assert result["status"] == "completed"
        assert result["batch_id"] == "batch-001"
        assert result["total"] == 3
        assert result["successful"] == 3
        assert result["failed"] == 0
        assert len(result["results"]) == 3

    def test_tracks_progress(self, mock_settings):
        """Progress updates are sent for each document."""
        from app.tasks import extract_batch

        mock_result = {
            "status": "completed",
            "source": "<raw_text>",
            "data": {"entities": []},
        }

        docs = [{"raw_text": "A"}, {"raw_text": "B"}]

        extract_batch.push_request(id="batch-task-2")
        try:
            with (
                patch(
                    "app.tasks._run_extraction",
                    return_value=mock_result,
                ),
                patch(
                    "celery.app.task.Task.update_state",
                ) as mock_update,
            ):
                extract_batch.run("batch-002", docs)
        finally:
            extract_batch.pop_request()

        # One update_state call per document in the loop
        batch_calls = [
            c
            for c in mock_update.call_args_list
            if c.kwargs.get("meta", {}).get("batch_id") == "batch-002"
        ]
        assert len(batch_calls) == 2

    def test_handles_partial_failure(self, mock_settings):
        """Documents that fail are captured in errors, others succeed."""
        from app.tasks import extract_batch

        call_count = 0
        ok_result = {
            "status": "completed",
            "source": "<raw_text>",
            "data": {"entities": []},
        }

        def run_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Extraction failed")
            return ok_result

        docs = [
            {"raw_text": "Good doc"},
            {"raw_text": "Bad doc"},
        ]

        extract_batch.push_request(id="batch-task-3")
        try:
            with (
                patch(
                    "app.tasks._run_extraction",
                    side_effect=run_side_effect,
                ),
                patch("celery.app.task.Task.update_state"),
            ):
                result = extract_batch.run("batch-003", docs)
        finally:
            extract_batch.pop_request()

        assert result["successful"] == 1
        assert result["failed"] == 1
        assert len(result["errors"]) == 1
        assert "Extraction failed" in result["errors"][0]["error"]

    def test_fires_batch_webhook(self, mock_settings):
        """Batch-level webhook is triggered on completion."""
        from app.tasks import extract_batch

        mock_result = {
            "status": "completed",
            "source": "<raw_text>",
            "data": {"entities": []},
        }

        extract_batch.push_request(id="batch-task-4")
        try:
            with (
                patch(
                    "app.tasks._run_extraction",
                    return_value=mock_result,
                ),
                patch("celery.app.task.Task.update_state"),
                patch("app.tasks._fire_webhook") as mock_webhook,
            ):
                extract_batch.run(
                    "batch-004",
                    [{"raw_text": "Doc"}],
                    callback_url="https://hook.example.com/batch",
                )
        finally:
            extract_batch.pop_request()

        mock_webhook.assert_called_once()
        assert mock_webhook.call_args[0][0] == "https://hook.example.com/batch"
