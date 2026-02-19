"""
LLM provider resolution helpers.

Centralises API-key selection and provider detection so that
the main extraction orchestrator stays focused on business
logic.
"""

from __future__ import annotations

from app.core.config import get_settings


def resolve_api_key(provider: str) -> str | None:
    """Pick the correct API key for *provider* from settings.

    Args:
        provider: Model ID string (e.g. ``gpt-4o``).

    Returns:
        An API key string, or ``None`` if nothing is configured.
    """
    settings = get_settings()
    lower = provider.lower()
    if "gpt" in lower or "openai" in lower:
        return settings.OPENAI_API_KEY or None
    return settings.LANGEXTRACT_API_KEY or settings.GEMINI_API_KEY or None


def is_openai_model(provider: str) -> bool:
    """Return ``True`` if *provider* is an OpenAI model.

    Args:
        provider: Model ID string.

    Returns:
        Boolean indicating whether OpenAI-specific flags apply.
    """
    lower = provider.lower()
    return "gpt" in lower or "openai" in lower
