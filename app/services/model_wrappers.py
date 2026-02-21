"""
Model wrapper utilities for audit logging and guardrails.

Provides factory functions that decorate a ``BaseLanguageModel``
with ``langextract-audit`` and/or ``langextract-guardrails``
providers based on application settings and per-request
configuration.

Wrapping order (inside → out):
    base model → guardrails → audit

Guardrails is innermost so it can validate and retry directly
against the LLM.  Audit is outermost so it logs the final
post-validation output.
"""

from __future__ import annotations

import logging
from typing import Any

from langextract.core.base_model import BaseLanguageModel
from langextract_audit import (
    AuditLanguageModel,
    AuditSink,
    JsonFileSink,
    LoggingSink,
)
from langextract_guardrails import (
    GuardrailLanguageModel,
    GuardrailValidator,
    JsonSchemaValidator,
    RegexValidator,
)

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


# ── Sink factory ────────────────────────────────────────────


def _build_audit_sinks(settings: Settings) -> list[AuditSink]:
    """Build audit sinks from application settings.

    Supports ``logging``, ``jsonfile``, and ``otel`` sink types.
    Falls back to ``LoggingSink`` on unknown values.

    Args:
        settings: The application ``Settings`` instance.

    Returns:
        A list containing a single ``AuditSink``.
    """
    sink_type = settings.AUDIT_SINK.lower()

    if sink_type == "jsonfile":
        logger.info(
            "Audit sink: JsonFileSink (path=%s)",
            settings.AUDIT_LOG_PATH,
        )
        return [JsonFileSink(path=settings.AUDIT_LOG_PATH)]

    if sink_type == "otel":
        try:
            from langextract_audit.sinks import OtelSpanSink

            logger.info("Audit sink: OtelSpanSink")
            return [OtelSpanSink()]
        except ImportError:
            logger.warning(
                "OpenTelemetry packages not installed — falling back to LoggingSink"
            )
            return [LoggingSink()]

    # Default: stdlib logging
    logger.info("Audit sink: LoggingSink")
    return [LoggingSink()]


# ── Validator factory ───────────────────────────────────────


def _build_validators(
    guardrails_config: dict[str, Any],
) -> list[GuardrailValidator]:
    """Build guardrail validators from per-request config.

    Supports ``json_schema`` and ``regex_pattern`` keys.  When
    neither is provided, a permissive ``JsonSchemaValidator``
    (syntax-only) is returned so that at minimum the LLM output
    is valid JSON.

    Args:
        guardrails_config: Guardrails configuration dict from
            the request's ``extraction_config``.

    Returns:
        A list of ``GuardrailValidator`` instances.
    """
    validators: list[GuardrailValidator] = []

    json_schema: dict[str, Any] | None = guardrails_config.get(
        "json_schema",
    )
    strict: bool = guardrails_config.get("json_schema_strict", True)
    if json_schema is not None:
        validators.append(
            JsonSchemaValidator(schema=json_schema, strict=strict),
        )

    regex_pattern: str | None = guardrails_config.get("regex_pattern")
    if regex_pattern is not None:
        description = guardrails_config.get(
            "regex_description",
            "output format",
        )
        validators.append(
            RegexValidator(pattern=regex_pattern, description=description),
        )

    # If no explicit validators, use syntax-only JSON check
    if not validators:
        validators.append(JsonSchemaValidator(schema=None, strict=False))

    return validators


# ── Public wrapping API ─────────────────────────────────────


def wrap_with_guardrails(
    model: BaseLanguageModel,
    model_id: str,
    guardrails_config: dict[str, Any],
) -> BaseLanguageModel:
    """Wrap a model with the guardrails provider.

    Args:
        model: The base ``BaseLanguageModel`` to wrap.
        model_id: The model identifier string.
        guardrails_config: Per-request guardrails configuration
            (from ``ExtractionConfig.guardrails``).

    Returns:
        A ``GuardrailLanguageModel`` wrapping the base model,
        or the original model if guardrails should not be
        applied.
    """
    settings = get_settings()

    # Resolve enabled flag: per-request > global setting
    enabled = guardrails_config.get("enabled")
    if enabled is None:
        enabled = settings.GUARDRAILS_ENABLED
    if not enabled:
        return model

    validators = _build_validators(guardrails_config)

    max_retries = guardrails_config.get(
        "max_retries",
        settings.GUARDRAILS_MAX_RETRIES,
    )
    include_output = guardrails_config.get(
        "include_output_in_correction",
        settings.GUARDRAILS_INCLUDE_OUTPUT_IN_CORRECTION,
    )
    max_concurrency = settings.GUARDRAILS_MAX_CONCURRENCY
    max_prompt_len = (
        guardrails_config.get(
            "max_correction_prompt_length",
        )
        or settings.GUARDRAILS_MAX_CORRECTION_PROMPT_LENGTH
    )
    max_output_len = (
        guardrails_config.get(
            "max_correction_output_length",
        )
        or settings.GUARDRAILS_MAX_CORRECTION_OUTPUT_LENGTH
    )

    wrapped = GuardrailLanguageModel(
        model_id=f"guardrails/{model_id}",
        inner=model,
        validators=validators,
        max_retries=max_retries,
        max_concurrency=max_concurrency,
        max_correction_prompt_length=max_prompt_len,
        max_correction_output_length=max_output_len,
        include_output_in_correction=include_output,
    )

    validator_names = [type(v).__name__ for v in validators]
    logger.info(
        "Wrapped model %s with guardrails (validators=%s, max_retries=%d)",
        model_id,
        validator_names,
        max_retries,
    )
    return wrapped


def wrap_with_audit(
    model: BaseLanguageModel,
    model_id: str,
    audit_config: dict[str, Any] | None = None,
) -> BaseLanguageModel:
    """Wrap a model with the audit logging provider.

    Args:
        model: The ``BaseLanguageModel`` to wrap (may already
            be wrapped with guardrails).
        model_id: The model identifier string.
        audit_config: Optional per-request audit overrides
            (from ``ExtractionConfig.audit``).

    Returns:
        An ``AuditLanguageModel`` wrapping the model, or the
        original model if audit is disabled.
    """
    settings = get_settings()
    audit_config = audit_config or {}

    # Resolve enabled flag: per-request > global setting
    enabled = audit_config.get("enabled")
    if enabled is None:
        enabled = settings.AUDIT_ENABLED
    if not enabled:
        return model

    sinks = _build_audit_sinks(settings)

    sample_length = audit_config.get(
        "sample_length",
        settings.AUDIT_SAMPLE_LENGTH,
    )

    wrapped = AuditLanguageModel(
        model_id=f"audit/{model_id}",
        inner=model,
        sinks=sinks,
        sample_length=sample_length,
    )

    logger.info(
        "Wrapped model %s with audit logging (sink=%s, sample_length=%s)",
        model_id,
        settings.AUDIT_SINK,
        sample_length,
    )
    return wrapped


def apply_model_wrappers(
    model: BaseLanguageModel,
    model_id: str,
    extraction_config: dict[str, Any],
) -> BaseLanguageModel:
    """Apply guardrails and audit wrappers to a model.

    Wrapping order: base → guardrails → audit.

    This is the single entry point called from the extraction
    orchestrator.  Configuration is resolved from both the
    per-request ``extraction_config`` and global application
    settings.

    Args:
        model: The base ``BaseLanguageModel`` instance.
        model_id: The model identifier string.
        extraction_config: The flat extraction configuration
            dict that may contain ``guardrails`` and ``audit``
            sub-dicts.

    Returns:
        The (possibly wrapped) model instance.
    """
    guardrails_config = extraction_config.get("guardrails") or {}
    audit_config = extraction_config.get("audit") or {}

    # Step 1: Guardrails (innermost wrapper)
    model = wrap_with_guardrails(model, model_id, guardrails_config)

    # Step 2: Audit (outermost wrapper)
    model = wrap_with_audit(model, model_id, audit_config)

    return model
