"""Tests for model wrapper utilities (audit + guardrails)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any
from unittest import mock

import pytest
from langextract.core.base_model import BaseLanguageModel
from langextract.core.types import ScoredOutput

from app.services.model_wrappers import (
    _build_audit_sinks,
    _build_validators,
    apply_model_wrappers,
    wrap_with_audit,
    wrap_with_guardrails,
)

# ── Fake provider for testing ──────────────────────────────


class FakeModel(BaseLanguageModel):
    """Minimal BaseLanguageModel stub for unit tests."""

    def __init__(self) -> None:
        super().__init__()
        self.model_id = "fake/test-model"

    def infer(
        self,
        batch_prompts: Sequence[str],
        **kwargs: Any,
    ) -> Iterator[Sequence[ScoredOutput]]:
        """Yield a dummy scored output per prompt."""
        for prompt in batch_prompts:
            yield [ScoredOutput(score=1.0, output=f"echo: {prompt}")]

    async def async_infer(
        self,
        batch_prompts: Sequence[str],
        **kwargs: Any,
    ) -> list[Sequence[ScoredOutput]]:
        """Return a dummy scored output per prompt."""
        return [[ScoredOutput(score=1.0, output=f"echo: {p}")] for p in batch_prompts]


# ── Settings fixture ───────────────────────────────────────


@pytest.fixture
def _default_settings(monkeypatch):
    """Patch ``get_settings`` to return deterministic defaults."""
    fake = mock.MagicMock()
    fake.AUDIT_ENABLED = False
    fake.AUDIT_SINK = "logging"
    fake.AUDIT_LOG_PATH = "audit.jsonl"
    fake.AUDIT_SAMPLE_LENGTH = None
    fake.GUARDRAILS_ENABLED = False
    fake.GUARDRAILS_MAX_RETRIES = 3
    fake.GUARDRAILS_MAX_CONCURRENCY = None
    fake.GUARDRAILS_INCLUDE_OUTPUT_IN_CORRECTION = True
    fake.GUARDRAILS_MAX_CORRECTION_PROMPT_LENGTH = None
    fake.GUARDRAILS_MAX_CORRECTION_OUTPUT_LENGTH = None
    monkeypatch.setattr(
        "app.services.model_wrappers.get_settings",
        lambda: fake,
    )
    return fake


# ── Audit sink factory tests ───────────────────────────────


class TestBuildAuditSinks:
    """Test the _build_audit_sinks factory function."""

    def test_default_logging_sink(self, _default_settings):
        """Default sink type is LoggingSink."""
        from langextract_audit import LoggingSink

        sinks = _build_audit_sinks(_default_settings)

        assert len(sinks) == 1
        assert isinstance(sinks[0], LoggingSink)

    def test_jsonfile_sink(self, _default_settings):
        """JsonFileSink when AUDIT_SINK=jsonfile."""
        from langextract_audit import JsonFileSink

        _default_settings.AUDIT_SINK = "jsonfile"
        _default_settings.AUDIT_LOG_PATH = "/tmp/test_audit.jsonl"

        sinks = _build_audit_sinks(_default_settings)

        assert len(sinks) == 1
        assert isinstance(sinks[0], JsonFileSink)

    def test_unknown_sink_falls_back_to_logging(self, _default_settings):
        """Unknown sink type falls back to LoggingSink."""
        from langextract_audit import LoggingSink

        _default_settings.AUDIT_SINK = "nonexistent"

        sinks = _build_audit_sinks(_default_settings)

        assert len(sinks) == 1
        assert isinstance(sinks[0], LoggingSink)


# ── Validator factory tests ─────────────────────────────────


class TestBuildValidators:
    """Test the _build_validators factory function."""

    def test_json_schema_validator(self):
        """A json_schema key produces a JsonSchemaValidator."""
        from langextract_guardrails import JsonSchemaValidator

        config = {
            "json_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        }
        validators = _build_validators(config)

        assert len(validators) == 1
        assert isinstance(validators[0], JsonSchemaValidator)

    def test_regex_validator(self):
        """A regex_pattern key produces a RegexValidator."""
        from langextract_guardrails import RegexValidator

        config = {
            "regex_pattern": r"\d{4}-\d{2}-\d{2}",
            "regex_description": "ISO date",
        }
        validators = _build_validators(config)

        assert len(validators) == 1
        assert isinstance(validators[0], RegexValidator)

    def test_both_validators(self):
        """Both json_schema and regex_pattern produce two validators."""
        config = {
            "json_schema": {"type": "object"},
            "regex_pattern": r'"name"',
        }
        validators = _build_validators(config)

        assert len(validators) == 2

    def test_empty_config_falls_back_to_syntax_only(self):
        """Empty config produces a syntax-only JsonSchemaValidator."""
        from langextract_guardrails import JsonSchemaValidator

        validators = _build_validators({})

        assert len(validators) == 1
        assert isinstance(validators[0], JsonSchemaValidator)
        assert validators[0].schema is None


# ── Guardrails wrapping tests ──────────────────────────────


class TestWrapWithGuardrails:
    """Test wrap_with_guardrails."""

    def test_disabled_returns_original_model(
        self,
        _default_settings,
    ):
        """When guardrails is disabled, the original model is returned."""
        model = FakeModel()
        result = wrap_with_guardrails(
            model,
            "test-model",
            {"enabled": False},
        )

        assert result is model

    def test_global_disabled_returns_original(
        self,
        _default_settings,
    ):
        """When global setting is False and no per-request override."""
        _default_settings.GUARDRAILS_ENABLED = False
        model = FakeModel()

        result = wrap_with_guardrails(model, "test-model", {})

        assert result is model

    def test_enabled_wraps_model(self, _default_settings):
        """When enabled, the model is wrapped with GuardrailLanguageModel."""
        from langextract_guardrails import GuardrailLanguageModel

        _default_settings.GUARDRAILS_ENABLED = True
        model = FakeModel()

        result = wrap_with_guardrails(model, "test-model", {})

        assert isinstance(result, GuardrailLanguageModel)
        assert result.inner is model

    def test_per_request_override_enables(self, _default_settings):
        """Per-request enabled=True wins over global disabled."""
        from langextract_guardrails import GuardrailLanguageModel

        _default_settings.GUARDRAILS_ENABLED = False
        model = FakeModel()

        result = wrap_with_guardrails(
            model,
            "test-model",
            {"enabled": True},
        )

        assert isinstance(result, GuardrailLanguageModel)

    def test_custom_max_retries(self, _default_settings):
        """Per-request max_retries overrides global setting."""
        from langextract_guardrails import GuardrailLanguageModel

        _default_settings.GUARDRAILS_ENABLED = True
        model = FakeModel()

        result = wrap_with_guardrails(
            model,
            "test-model",
            {"max_retries": 5},
        )

        assert isinstance(result, GuardrailLanguageModel)
        assert result.max_retries == 5

    def test_model_id_prefix(self, _default_settings):
        """Wrapped model_id is prefixed with 'guardrails/'."""
        from langextract_guardrails import GuardrailLanguageModel

        _default_settings.GUARDRAILS_ENABLED = True
        model = FakeModel()

        result = wrap_with_guardrails(
            model,
            "gpt-4o",
            {},
        )

        assert isinstance(result, GuardrailLanguageModel)
        assert result.model_id == "guardrails/gpt-4o"


# ── Audit wrapping tests ───────────────────────────────────


class TestWrapWithAudit:
    """Test wrap_with_audit."""

    def test_disabled_returns_original_model(
        self,
        _default_settings,
    ):
        """When audit is disabled, the original model is returned."""
        model = FakeModel()
        result = wrap_with_audit(
            model,
            "test-model",
            {"enabled": False},
        )

        assert result is model

    def test_global_disabled_returns_original(
        self,
        _default_settings,
    ):
        """When global AUDIT_ENABLED is False."""
        _default_settings.AUDIT_ENABLED = False
        model = FakeModel()

        result = wrap_with_audit(model, "test-model")

        assert result is model

    def test_enabled_wraps_model(self, _default_settings):
        """When enabled, the model is wrapped with AuditLanguageModel."""
        from langextract_audit import AuditLanguageModel

        _default_settings.AUDIT_ENABLED = True
        model = FakeModel()

        result = wrap_with_audit(model, "test-model")

        assert isinstance(result, AuditLanguageModel)
        assert result.inner is model

    def test_per_request_override_enables(self, _default_settings):
        """Per-request enabled=True wins over global disabled."""
        from langextract_audit import AuditLanguageModel

        _default_settings.AUDIT_ENABLED = False
        model = FakeModel()

        result = wrap_with_audit(
            model,
            "test-model",
            {"enabled": True},
        )

        assert isinstance(result, AuditLanguageModel)

    def test_sample_length_from_config(self, _default_settings):
        """Per-request sample_length is applied."""
        from langextract_audit import AuditLanguageModel

        _default_settings.AUDIT_ENABLED = True
        model = FakeModel()

        result = wrap_with_audit(
            model,
            "test-model",
            {"sample_length": 100},
        )

        assert isinstance(result, AuditLanguageModel)

    def test_model_id_prefix(self, _default_settings):
        """Wrapped model_id is prefixed with 'audit/'."""
        from langextract_audit import AuditLanguageModel

        _default_settings.AUDIT_ENABLED = True
        model = FakeModel()

        result = wrap_with_audit(model, "gpt-4o")

        assert isinstance(result, AuditLanguageModel)
        assert result.model_id == "audit/gpt-4o"


# ── End-to-end apply_model_wrappers tests ──────────────────


class TestApplyModelWrappers:
    """Test the apply_model_wrappers orchestration function."""

    def test_no_wrappers_when_both_disabled(
        self,
        _default_settings,
    ):
        """When both audit and guardrails are disabled."""
        model = FakeModel()
        result = apply_model_wrappers(model, "test-model", {})

        assert result is model

    def test_both_wrappers_applied(self, _default_settings):
        """When both are enabled, model is double-wrapped."""
        from langextract_audit import AuditLanguageModel
        from langextract_guardrails import GuardrailLanguageModel

        _default_settings.AUDIT_ENABLED = True
        _default_settings.GUARDRAILS_ENABLED = True
        model = FakeModel()

        result = apply_model_wrappers(model, "test-model", {})

        # Outer wrapper is Audit
        assert isinstance(result, AuditLanguageModel)
        # Inner wrapper is Guardrails
        assert isinstance(result.inner, GuardrailLanguageModel)
        # Innermost is the original model
        assert result.inner.inner is model

    def test_guardrails_only(self, _default_settings):
        """Only guardrails enabled, audit disabled."""
        from langextract_guardrails import GuardrailLanguageModel

        _default_settings.GUARDRAILS_ENABLED = True
        _default_settings.AUDIT_ENABLED = False
        model = FakeModel()

        result = apply_model_wrappers(model, "test-model", {})

        assert isinstance(result, GuardrailLanguageModel)
        assert result.inner is model

    def test_audit_only(self, _default_settings):
        """Only audit enabled, guardrails disabled."""
        from langextract_audit import AuditLanguageModel

        _default_settings.AUDIT_ENABLED = True
        _default_settings.GUARDRAILS_ENABLED = False
        model = FakeModel()

        result = apply_model_wrappers(model, "test-model", {})

        assert isinstance(result, AuditLanguageModel)
        assert result.inner is model

    def test_per_request_config_overrides(self, _default_settings):
        """Per-request config overrides global settings."""
        from langextract_audit import AuditLanguageModel
        from langextract_guardrails import GuardrailLanguageModel

        _default_settings.AUDIT_ENABLED = False
        _default_settings.GUARDRAILS_ENABLED = False
        model = FakeModel()

        extraction_config = {
            "guardrails": {
                "enabled": True,
                "json_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
            "audit": {
                "enabled": True,
                "sample_length": 50,
            },
        }
        result = apply_model_wrappers(
            model,
            "test-model",
            extraction_config,
        )

        assert isinstance(result, AuditLanguageModel)
        assert isinstance(result.inner, GuardrailLanguageModel)

    def test_wrapping_preserves_infer_contract(
        self,
        _default_settings,
    ):
        """Wrapped model still yields valid ScoredOutput sequences."""
        # Enable only audit (not guardrails) so the FakeModel's
        # plain-text output is not rejected by JSON validation.
        _default_settings.AUDIT_ENABLED = True
        _default_settings.GUARDRAILS_ENABLED = False
        model = FakeModel()

        wrapped = apply_model_wrappers(model, "test-model", {})
        results = list(wrapped.infer(["test prompt"]))

        assert len(results) == 1
        outputs = list(results[0])
        assert len(outputs) >= 1
        assert outputs[0].output == "echo: test prompt"

    def test_guardrails_passes_valid_json(
        self,
        _default_settings,
    ):
        """Guardrails wrapper passes through valid JSON output."""

        class JsonModel(BaseLanguageModel):
            """Model that returns valid JSON."""

            def __init__(self) -> None:
                super().__init__()
                self.model_id = "fake/json"

            def infer(
                self,
                batch_prompts: Sequence[str],
                **kwargs: Any,
            ) -> Iterator[Sequence[ScoredOutput]]:
                for _prompt in batch_prompts:
                    yield [ScoredOutput(score=1.0, output='{"name": "Alice"}')]

            async def async_infer(
                self,
                batch_prompts: Sequence[str],
                **kwargs: Any,
            ) -> list[Sequence[ScoredOutput]]:
                return [
                    [ScoredOutput(score=1.0, output='{"name": "Alice"}')]
                    for _ in batch_prompts
                ]

        _default_settings.GUARDRAILS_ENABLED = True
        _default_settings.AUDIT_ENABLED = False
        model = JsonModel()

        wrapped = apply_model_wrappers(model, "test-model", {})
        results = list(wrapped.infer(["test prompt"]))

        assert len(results) == 1
        outputs = list(results[0])
        assert len(outputs) >= 1
        assert '"Alice"' in outputs[0].output
