"""The small interface the rest of the app uses, so providers can be swapped by config."""

from typing import Protocol


class EmbeddingError(RuntimeError):
    """The provider could not return embeddings."""


class Embedder(Protocol):
    @property
    def model(self) -> str: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...
