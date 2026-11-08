"""Small interfaces the rest of the app uses, so providers can be swapped by config."""

from dataclasses import dataclass
from typing import Protocol


class ProviderError(RuntimeError):
    """An external model provider could not complete a request."""


class EmbeddingError(ProviderError):
    """The provider could not return embeddings."""


class RerankError(ProviderError):
    """The provider could not re-rank documents."""


@dataclass(frozen=True)
class RerankResult:
    index: int  # position in the list of documents that was sent
    score: float


class Embedder(Protocol):
    @property
    def model(self) -> str: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    async def rerank(self, query: str, documents: list[str]) -> list[RerankResult]: ...
