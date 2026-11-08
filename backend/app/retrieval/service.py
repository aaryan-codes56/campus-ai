"""The retrieval pipeline: search, fuse, re-rank, then apply recency."""

import datetime as dt
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.embeddings.base import Embedder, EmbeddingError, Reranker, RerankError
from app.embeddings.factory import build_embedder, build_reranker
from app.ingestion.chunking import build_embedding_text
from app.retrieval.fusion import apply_recency, reciprocal_rank_fusion
from app.retrieval.search import keyword_search, vector_search
from app.retrieval.types import (
    Candidate,
    RetrievalConfig,
    RetrievalResult,
    RetrievedChunk,
    SearchFilters,
)

logger = logging.getLogger(__name__)

QUERY_CACHE_SIZE = 256


class RetrievalError(RuntimeError):
    """Retrieval could not run at all, and no fallback was available."""


def config_from_settings(settings: Settings) -> RetrievalConfig:
    return RetrievalConfig(
        mode=settings.retrieval_mode,
        rerank=settings.rerank_enabled,
        candidates=settings.retrieval_candidates,
        rerank_candidates=settings.rerank_candidates,
        top_k=settings.retrieval_top_k,
        rrf_k=settings.rrf_k,
        recency_boost=settings.recency_boost,
        recency_half_life_days=settings.recency_half_life_days,
    )


@dataclass
class _Scored:
    candidate: Candidate
    fusion_score: float
    rerank_score: float | None = None


class Retriever:
    def __init__(
        self,
        embedder: Embedder | None,
        reranker: Reranker | None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._reranker = reranker
        self._config = config or RetrievalConfig()
        self._query_vectors: OrderedDict[str, list[float]] = OrderedDict()

    async def _embed_query(self, embedder: Embedder, query: str) -> list[float]:
        # Repeated questions and eval runs should not spend the rate limit twice.
        cached = self._query_vectors.get(query)
        if cached is not None:
            self._query_vectors.move_to_end(query)
            return cached
        vector = await embedder.embed_query(query)
        self._query_vectors[query] = vector
        if len(self._query_vectors) > QUERY_CACHE_SIZE:
            self._query_vectors.popitem(last=False)
        return vector

    async def search(
        self,
        session: AsyncSession,
        query: str,
        *,
        filters: SearchFilters | None = None,
        config: RetrievalConfig | None = None,
        today: dt.date | None = None,
    ) -> RetrievalResult:
        active = config or self._config
        active_filters = filters or SearchFilters()
        result = RetrievalResult(chunks=[], config=active)

        vector_hits = await self._run_vector(session, query, active, active_filters, result)
        keyword_hits: list[Candidate] = []
        if active.mode in ("keyword", "hybrid"):
            started = time.perf_counter()
            keyword_hits = await keyword_search(
                session, query, filters=active_filters, limit=active.candidates
            )
            result.timings_ms["keyword"] = _elapsed_ms(started)

        by_id = {hit.chunk_id: hit for hit in [*vector_hits, *keyword_hits]}
        vector_ids = [hit.chunk_id for hit in vector_hits]
        keyword_ids = [hit.chunk_id for hit in keyword_hits]
        fused = reciprocal_rank_fusion([vector_ids, keyword_ids], k=active.rrf_k)
        scored = [_Scored(by_id[chunk_id], score) for chunk_id, score in fused]

        if active.rerank:
            scored = await self._rerank(query, scored, active, result)

        reference_day = today or dt.date.today()
        chunks = [
            _build_chunk(item, active, reference_day, vector_ids, keyword_ids) for item in scored
        ]
        chunks.sort(key=lambda chunk: chunk.score, reverse=True)
        result.chunks = chunks[: active.top_k]
        return result

    async def _run_vector(
        self,
        session: AsyncSession,
        query: str,
        config: RetrievalConfig,
        filters: SearchFilters,
        result: RetrievalResult,
    ) -> list[Candidate]:
        if config.mode == "keyword":
            return []
        try:
            if self._embedder is None:
                raise EmbeddingError("no embedding provider is configured")
            started = time.perf_counter()
            vector = await self._embed_query(self._embedder, query)
            result.timings_ms["embed"] = _elapsed_ms(started)
        except EmbeddingError as error:
            if config.mode == "vector":
                raise RetrievalError(str(error)) from error
            logger.warning("Vector search skipped: %s", error)
            result.degraded.append("vector")
            return []

        started = time.perf_counter()
        hits = await vector_search(
            session, vector, model=self._embedder.model, filters=filters, limit=config.candidates
        )
        result.timings_ms["vector"] = _elapsed_ms(started)
        return hits

    async def _rerank(
        self,
        query: str,
        scored: list[_Scored],
        config: RetrievalConfig,
        result: RetrievalResult,
    ) -> list[_Scored]:
        """Re-score the best candidates. On failure, keep the fused order."""
        shortlist = scored[: config.rerank_candidates]
        if not shortlist:
            return shortlist
        if self._reranker is None:
            result.degraded.append("rerank")
            return shortlist

        documents = [
            build_embedding_text(
                item.candidate.title, item.candidate.heading_path, item.candidate.text
            )
            for item in shortlist
        ]
        started = time.perf_counter()
        try:
            ranked = await self._reranker.rerank(query, documents)
        except RerankError as error:
            logger.warning("Re-ranking skipped: %s", error)
            result.degraded.append("rerank")
            return shortlist
        result.timings_ms["rerank"] = _elapsed_ms(started)
        return [
            _Scored(
                shortlist[entry.index].candidate,
                shortlist[entry.index].fusion_score,
                entry.score,
            )
            for entry in ranked
        ]


def _rank_of(chunk_id: uuid.UUID, ranking: list[uuid.UUID]) -> int | None:
    return ranking.index(chunk_id) + 1 if chunk_id in ranking else None


def _build_chunk(
    item: _Scored,
    config: RetrievalConfig,
    today: dt.date,
    vector_ids: list[uuid.UUID],
    keyword_ids: list[uuid.UUID],
) -> RetrievedChunk:
    base = item.rerank_score if item.rerank_score is not None else item.fusion_score
    score = apply_recency(
        base,
        item.candidate.doc_date,
        today=today,
        boost=config.recency_boost,
        half_life_days=config.recency_half_life_days,
    )
    return RetrievedChunk(
        candidate=item.candidate,
        score=score,
        fusion_score=item.fusion_score,
        rerank_score=item.rerank_score,
        vector_rank=_rank_of(item.candidate.chunk_id, vector_ids),
        keyword_rank=_rank_of(item.candidate.chunk_id, keyword_ids),
    )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def build_retriever(settings: Settings, *, interactive: bool = True) -> Retriever:
    return Retriever(
        build_embedder(settings, interactive=interactive),
        build_reranker(settings, interactive=interactive),
        config_from_settings(settings),
    )
