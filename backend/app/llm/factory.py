"""Builds the configured LLM provider."""

from app.core.config import Settings
from app.llm.base import LLMProvider
from app.llm.claude import ClaudeProvider
from app.llm.gemini import GeminiProvider


def build_llm(settings: Settings) -> LLMProvider | None:
    """Return the configured provider, or None when its API key is missing."""
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            return None
        return ClaudeProvider(
            settings.anthropic_api_key, settings.anthropic_model, effort=settings.anthropic_effort
        )
    if not settings.gemini_api_key:
        return None
    return GeminiProvider(
        settings.gemini_api_key,
        settings.gemini_model,
        thinking_level=settings.gemini_thinking_level,
    )
