"""Data structures for retrieval."""

import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Literal

RetrievalMode = Literal["vector", "keyword", "hybrid"]


@dataclass(frozen=True)
class RetrievalConfig:
    mode: RetrievalMode = "hybrid"
    rerank: bool = True
    candidates: int = 30  # taken from each search method
    rerank_candidates: int = 20  # sent to the re-ranker after fusion
    top_k: int = 6  # returned to the caller
    rrf_k: int = 60
    recency_boost: float = 0.05
    recency_half_life_days: int = 365


@dataclass(frozen=True)
class SearchFilters:
    doc_types: tuple[str, ...] = ()
    date_from: dt.date | None = None
    date_to: dt.date | None = None

    @property
    def active(self) -> bool:
        return bool(self.doc_types or self.date_from or self.date_to)


@dataclass(frozen=True)
class Candidate:
    """A chunk found by one search method."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    text: str
    heading_path: str
    page_start: int | None
    page_end: int | None
    title: str
    doc_type: str
    doc_date: dt.date | None
    source_file: str
    url: str | None


@dataclass(frozen=True)
class RetrievedChunk:
    candidate: Candidate
    score: float  # final score used for ordering
    fusion_score: float
    rerank_score: float | None  # None when re-ranking was off or failed
    vector_rank: int | None  # 1-based position, None if this method did not find it
    keyword_rank: int | None


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    config: RetrievalConfig
    timings_ms: dict[str, float] = field(default_factory=dict)
    # Stages that failed and were skipped, so callers and logs can see reduced quality.
    degraded: list[str] = field(default_factory=list)
