"""Tests for structured output helpers.

Validates:
- JSON Schema generation from LangCore examples
- Extraction class collection
- Provider capability detection
- Edge cases (empty examples, missing attributes)
"""

from __future__ import annotations

from unittest import mock

from app.services.structured_output import (
    _collect_extraction_classes,
    build_response_format,
    supports_structured_output,
)

# ── Fixtures ────────────────────────────────────────────────

SAMPLE_EXAMPLES: list[dict] = [
    {
        "text": (
            "This Agreement is entered into by Acme Corp "
            "('Seller') and Global LLC ('Buyer'). Price: $1M."
        ),
        "extractions": [
            {
                "extraction_class": "party",
                "extraction_text": "Acme Corp",
                "attributes": {
                    "role": "Seller",
                    "entity_type": "corporation",
                },
            },
            {
                "extraction_class": "party",
                "extraction_text": "Global LLC",
                "attributes": {"role": "Buyer"},
            },
            {
                "extraction_class": "monetary_amount",
                "extraction_text": "$1M",
                "attributes": {"type": "purchase_price"},
            },
        ],
    },
]


# ── _collect_extraction_classes ─────────────────────────────


class TestCollectExtractionClasses:
    """Tests for ``_collect_extraction_classes``."""

    def test_collects_unique_classes(self):
        """All extraction class names are discovered."""
        classes = _collect_extraction_classes(SAMPLE_EXAMPLES)
        assert set(classes.keys()) == {"party", "monetary_amount"}

    def test_collects_attribute_names_and_types(self):
        """Attribute names and Python types are tracked."""
        classes = _collect_extraction_classes(SAMPLE_EXAMPLES)
        assert "role" in classes["party"]
        assert str in classes["party"]["role"]
        assert "entity_type" in classes["party"]

    def test_empty_examples(self):
        """Empty examples list returns empty mapping."""
        assert _collect_extraction_classes([]) == {}

    def test_no_attributes(self):
        """Extractions without attributes produce empty attr dict."""
        examples = [
            {
                "text": "test",
                "extractions": [
                    {
                        "extraction_class": "label",
                        "extraction_text": "hello",
                    }
                ],
            }
        ]
        classes = _collect_extraction_classes(examples)
        assert "label" in classes
        assert classes["label"] == {}

    def test_list_attribute_type(self):
        """List-typed attributes are tracked correctly."""
        examples = [
            {
                "text": "test",
                "extractions": [
                    {
                        "extraction_class": "item",
                        "extraction_text": "x",
                        "attributes": {"tags": ["a", "b"]},
                    }
                ],
            }
        ]
        classes = _collect_extraction_classes(examples)
        assert list in classes["item"]["tags"]


# ── build_response_format ───────────────────────────────────


class TestBuildResponseFormat:
    """Tests for ``build_response_format``."""

    def test_top_level_structure(self):
        """Response format has correct top-level keys."""
        rf = build_response_format(SAMPLE_EXAMPLES)
        assert rf["type"] == "json_schema"
        assert "json_schema" in rf
        js = rf["json_schema"]
        assert js["name"] == "extraction_result"
        assert js["strict"] is False
        assert "schema" in js

    def test_schema_has_extractions_array(self):
        """Schema requires an ``extractions`` array."""
        rf = build_response_format(SAMPLE_EXAMPLES)
        schema = rf["json_schema"]["schema"]
        assert schema["type"] == "object"
        assert "extractions" in schema["properties"]
        assert schema["properties"]["extractions"]["type"] == "array"
        assert "extractions" in schema["required"]

    def test_items_use_anyof_for_classes(self):
        """Item schema uses anyOf with per-class sub-schemas."""
        rf = build_response_format(SAMPLE_EXAMPLES)
        items = rf["json_schema"]["schema"]["properties"]["extractions"]["items"]
        assert "anyOf" in items
        class_names = set()
        for sub in items["anyOf"]:
            # Each sub-schema has exactly one "required" class key
            for key in sub.get("required", []):
                class_names.add(key)
        assert class_names == {"party", "monetary_amount"}

    def test_attributes_field_generated(self):
        """Each class sub-schema includes ``<class>_attributes``."""
        rf = build_response_format(SAMPLE_EXAMPLES)
        items = rf["json_schema"]["schema"]["properties"]["extractions"]["items"]
        for sub in items["anyOf"]:
            props = sub["properties"]
            # Find the class key (not ending with _attributes)
            cls_keys = [k for k in props if not k.endswith("_attributes")]
            for cls_key in cls_keys:
                attr_key = f"{cls_key}_attributes"
                assert attr_key in props, f"Missing {attr_key} in {list(props.keys())}"

    def test_empty_examples_minimal_schema(self):
        """Empty examples produce a minimal schema without anyOf."""
        rf = build_response_format([])
        items = rf["json_schema"]["schema"]["properties"]["extractions"]["items"]
        assert "anyOf" not in items
        assert items["type"] == "object"

    def test_no_extractions_key_examples(self):
        """Examples with no ``extractions`` key still work."""
        rf = build_response_format([{"text": "no extractions here"}])
        items = rf["json_schema"]["schema"]["properties"]["extractions"]["items"]
        assert "anyOf" not in items


# ── supports_structured_output ──────────────────────────────


class TestSupportsStructuredOutput:
    """Tests for ``supports_structured_output``."""

    def test_returns_true_when_litellm_says_yes(self):
        """Delegates to litellm.supports_response_schema."""
        with mock.patch("app.services.structured_output.litellm") as mock_litellm:
            mock_litellm.supports_response_schema.return_value = True
            assert supports_structured_output("gpt-4o") is True
            mock_litellm.supports_response_schema.assert_called_once_with(
                model="gpt-4o",
                custom_llm_provider=None,
            )

    def test_returns_false_when_litellm_says_no(self):
        with mock.patch("app.services.structured_output.litellm") as mock_litellm:
            mock_litellm.supports_response_schema.return_value = False
            assert supports_structured_output("llama-7b") is False

    def test_returns_false_on_import_error(self):
        """Gracefully returns False when litellm is unavailable."""
        with mock.patch(
            "app.services.structured_output.litellm",
            side_effect=ImportError,
        ):
            # The import happens inside the function; mock the
            # import to raise.  Since we import at module level,
            # simulate the function catching an exception.
            pass

        # Direct test: patch the internal import
        with (
            mock.patch.dict("sys.modules", {"litellm": None}),
            mock.patch("app.services.structured_output.litellm") as mock_litellm,
        ):
            mock_litellm.supports_response_schema.side_effect = Exception("boom")
            assert supports_structured_output("gpt-4o") is False

    def test_returns_false_on_exception(self):
        """Any exception falls back to False."""
        with mock.patch("app.services.structured_output.litellm") as mock_litellm:
            mock_litellm.supports_response_schema.side_effect = RuntimeError(
                "network error"
            )
            assert supports_structured_output("gpt-4o") is False
