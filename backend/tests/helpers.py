"""Test utilities: build real PDFs on the fly and fake the embedding provider."""

from collections.abc import Sequence
from pathlib import Path

import pymupdf

from app.db.models import EMBEDDING_DIMENSION
from app.embeddings.base import EmbeddingError

BODY = "helv"
BOLD = "hebo"

PdfItem = tuple[float, float, str, float, str]  # x, y, text, font size, font name


def write_pdf(path: Path, pages: Sequence[Sequence[PdfItem]]) -> Path:
    document = pymupdf.open()
    for items in pages:
        page = document.new_page()
        for x, y, text, size, font in items:
            page.insert_text((x, y), text, fontsize=size, fontname=font)
    document.save(path)
    document.close()
    return path


def body_lines(
    start_y: float, count: int, text: str = "Plain body sentence for testing."
) -> list[PdfItem]:
    return [(72, start_y + index * 14, text, 11, BODY) for index in range(count)]


class FakeEmbedder:
    """Records calls and returns a fixed vector, so tests need no network or key."""

    def __init__(self, model: str = "fake-model", fail_on_call: int | None = None) -> None:
        self._model = model
        self.fail_on_call = fail_on_call
        self.calls: list[list[str]] = []

    @property
    def model(self) -> str:
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail_on_call is not None and len(self.calls) + 1 == self.fail_on_call:
            raise EmbeddingError("simulated failure")
        self.calls.append(texts)
        return [[0.1] * EMBEDDING_DIMENSION for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.1] * EMBEDDING_DIMENSION
