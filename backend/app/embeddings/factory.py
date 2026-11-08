"""Builds the configured embedding provider."""

from app.core.config import Settings
from app.db.models import EMBEDDING_DIMENSION
from app.embeddings.base import Embedder
from app.embeddings.voyage import VoyageEmbedder


def build_embedder(settings: Settings) -> Embedder | None:
    """Return the embedder, or None when no API key is configured."""
    if not settings.voyage_api_key:
        return None
    return VoyageEmbedder(settings.voyage_api_key, settings.embedding_model, EMBEDDING_DIMENSION)
