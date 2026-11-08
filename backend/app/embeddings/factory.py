"""Builds the configured model providers."""

from app.core.config import Settings
from app.db.models import EMBEDDING_DIMENSION
from app.embeddings.base import Embedder, Reranker
from app.embeddings.voyage import VoyageEmbedder, VoyageReranker

# A student waiting on an answer should not sit through long rate-limit backoffs.
INTERACTIVE_RETRIES = 1
INTERACTIVE_DELAY_SECONDS = 1.0


def build_embedder(settings: Settings, *, interactive: bool = False) -> Embedder | None:
    """Return the embedder, or None when no API key is configured."""
    if not settings.voyage_api_key:
        return None
    if interactive:
        return VoyageEmbedder(
            settings.voyage_api_key,
            settings.embedding_model,
            EMBEDDING_DIMENSION,
            max_retries=INTERACTIVE_RETRIES,
            base_delay=INTERACTIVE_DELAY_SECONDS,
        )
    return VoyageEmbedder(settings.voyage_api_key, settings.embedding_model, EMBEDDING_DIMENSION)


def build_reranker(settings: Settings, *, interactive: bool = False) -> Reranker | None:
    """Return the re-ranker, or None when no API key is configured."""
    if not settings.voyage_api_key:
        return None
    if interactive:
        return VoyageReranker(
            settings.voyage_api_key,
            settings.rerank_model,
            max_retries=INTERACTIVE_RETRIES,
            base_delay=INTERACTIVE_DELAY_SECONDS,
        )
    return VoyageReranker(settings.voyage_api_key, settings.rerank_model)
