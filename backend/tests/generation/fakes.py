"""Fake LLM and sample retrieval results for generation tests."""

import datetime as dt
import uuid
from collections.abc import AsyncIterator

from app.llm.base import LLMError, LLMEvent, LLMUsage, StreamEnd, TextDelta
from app.retrieval.types import Candidate, RetrievalConfig, RetrievalResult, RetrievedChunk

Script = list[str | LLMError]


class FakeLLM:
    """Plays back scripted attempts. An LLMError in a script is raised at that point."""

    def __init__(self, *attempts: Script) -> None:
        self._attempts = list(attempts)
        self.calls = 0
        self.last_prompt = ""
        self.last_system = ""

    @property
    def model(self) -> str:
        return "fake-llm"

    async def stream(
        self, *, system: str, prompt: str, max_output_tokens: int
    ) -> AsyncIterator[LLMEvent]:
        self.calls += 1
        self.last_prompt, self.last_system = prompt, system
        for item in self._attempts.pop(0):
            if isinstance(item, LLMError):
                raise item
            yield TextDelta(item)
        yield StreamEnd(usage=LLMUsage(100, 20), truncated=False)


def make_chunk(
    text: str = "Fees are due in August.",
    *,
    title: str = "Fee Policy",
    rerank_score: float | None = 0.9,
    page: int | None = 3,
    heading: str = "1. Fees",
    doc_date: dt.date | None = dt.date(2026, 5, 1),
) -> RetrievedChunk:
    candidate = Candidate(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text=text,
        heading_path=heading,
        page_start=page,
        page_end=page,
        title=title,
        doc_type="policy",
        doc_date=doc_date,
        source_file="fee.pdf",
        url=None,
    )
    return RetrievedChunk(
        candidate=candidate,
        score=rerank_score or 0.03,
        fusion_score=0.03,
        rerank_score=rerank_score,
        vector_rank=1,
        keyword_rank=1,
    )


def make_retrieval(*chunks: RetrievedChunk, degraded: list[str] | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunks=list(chunks),
        config=RetrievalConfig(),
        timings_ms={"vector": 5.0},
        degraded=degraded or [],
    )
