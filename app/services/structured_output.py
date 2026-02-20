"""
Structured output helpers for LiteLLM ``response_format``.

Generates a JSON Schema from LangExtract examples and provides
capability detection so that the extraction orchestrator can
enable schema-constrained output for providers that support it.

The generated schema matches the format that LangExtract's
resolver expects: a top-level ``extractions`` array where each
item uses dynamic keys (the extraction class name for the text
and ``<class>_attributes`` for the associated attributes).

Example LLM output the schema describes::

    {
        "extractions": [
            {
                "party": "Acme Corp",
                "party_attributes": {"role": "Seller"}
            },
            {
                "date": "January 15, 2025",
                "date_attributes": {"type": "effective_date"}
            }
        ]
    }
"""

from __future__ import annotations

import logging
from typing import Any

import litellm

logger = logging.getLogger(__name__)

# Suffix appended to an extraction class name to form the
# attributes key (must match ``langextract.core.data.ATTRIBUTE_SUFFIX``).
_ATTRIBUTE_SUFFIX: str = "_attributes"

# Wrapper key for the top-level array
# (must match ``langextract.core.data.EXTRACTIONS_KEY``).
_EXTRACTIONS_KEY: str = "extractions"


def _collect_extraction_classes(
    raw_examples: list[dict[str, Any]],
) -> dict[str, dict[str, set[type]]]:
    """Scan examples and collect extraction classes with attributes.

    Args:
        raw_examples: List of example dicts, each with ``text``
            and ``extractions`` keys.

    Returns:
        Mapping from class name to a dict of attribute names and
        their observed Python types.
    """
    classes: dict[str, dict[str, set[type]]] = {}
    for example in raw_examples:
        for ext in example.get("extractions", []):
            cls_name: str = ext.get("extraction_class", "")
            if not cls_name:
                continue
            if cls_name not in classes:
                classes[cls_name] = {}
            for attr_name, attr_value in (ext.get("attributes") or {}).items():
                if attr_name not in classes[cls_name]:
                    classes[cls_name][attr_name] = set()
                classes[cls_name][attr_name].add(type(attr_value))
    return classes


def _attr_json_type(python_types: set[type]) -> dict[str, Any]:
    """Map a set of observed Python types to a JSON Schema type.

    Args:
        python_types: Set of Python types observed for an
            attribute value.

    Returns:
        A JSON Schema type descriptor (e.g. ``{"type": "string"}``).
    """
    if list in python_types:
        return {"type": "array", "items": {"type": "string"}}
    if int in python_types and float not in python_types:
        return {"type": "string"}
    return {"type": "string"}


def build_response_format(
    raw_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a ``response_format`` dict for ``litellm.completion()``.

    Generates a JSON Schema (non-strict) from the provided
    LangExtract examples.  The schema describes the expected
    output structure so that cloud providers constrain the
    LLM's output to valid JSON matching the extraction format.

    When no extraction classes can be determined from examples,
    a minimal schema requiring only the ``extractions`` array
    is returned.

    Args:
        raw_examples: LangExtract example dicts (each with
            ``text`` and ``extractions`` keys).

    Returns:
        A dict suitable for ``response_format`` kwarg in
        ``litellm.completion()``::

            {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_result",
                    "strict": False,
                    "schema": { ... }
                }
            }
    """
    classes = _collect_extraction_classes(raw_examples)

    # Build per-class sub-schemas for anyOf constraint.
    class_schemas: list[dict[str, Any]] = []
    for cls_name, attrs in classes.items():
        attr_properties: dict[str, Any] = {}
        for attr_name, attr_types in attrs.items():
            attr_properties[attr_name] = _attr_json_type(attr_types)

        attributes_field = f"{cls_name}{_ATTRIBUTE_SUFFIX}"
        props: dict[str, Any] = {
            cls_name: {"type": "string"},
        }
        if attr_properties:
            props[attributes_field] = {
                "type": "object",
                "properties": attr_properties,
            }
        else:
            # Even without known attributes, include the
            # attributes field so the LLM can populate it.
            props[attributes_field] = {"type": "object"}

        class_schemas.append(
            {
                "type": "object",
                "properties": props,
                "required": [cls_name],
            }
        )

    # Item schema — use anyOf when classes are known, otherwise
    # fall back to a minimal object schema.
    if class_schemas:
        items_schema: dict[str, Any] = {
            "type": "object",
            "anyOf": class_schemas,
        }
    else:
        items_schema = {
            "type": "object",
            "description": (
                "Each key is an extraction class name with its "
                "value being the extracted text.  Attributes use "
                "the '<class>_attributes' suffix."
            ),
        }

    json_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            _EXTRACTIONS_KEY: {
                "type": "array",
                "items": items_schema,
            },
        },
        "required": [_EXTRACTIONS_KEY],
    }

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "extraction_result",
            "strict": False,
            "schema": json_schema,
        },
    }


def supports_structured_output(model_id: str) -> bool:
    """Check whether *model_id* supports ``response_format``
    with JSON Schema.

    Uses ``litellm.supports_response_schema()`` for the check.
    Returns ``False`` on any error (missing model info, import
    failure, etc.) so that callers can safely fall back to
    prompt-only extraction.

    Args:
        model_id: LLM model identifier (e.g. ``gpt-4o``).

    Returns:
        ``True`` if the provider advertises JSON Schema support.
    """
    try:
        return bool(
            litellm.supports_response_schema(
                model=model_id,
                custom_llm_provider=None,
            )
        )
    except Exception:
        logger.debug(
            "Could not determine response_schema support for %s "
            "— falling back to prompt-only extraction",
            model_id,
            exc_info=True,
        )
        return False
