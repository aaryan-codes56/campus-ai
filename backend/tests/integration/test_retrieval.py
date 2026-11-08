"""Retrieval against real Postgres, with predictable fake model providers."""

import datetime as dt
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.retrieval.search import keyword_search, vector_search
from app.retrieval.service import RetrievalError, Retriever
from app.retrieval.types import RetrievalConfig, RetrievalResult, SearchFilters
from tests.helpers import FakeQueryEmbedder, FakeReranker, axis_vector

pytestmark = pytest.mark.integration

NO_FILTERS = SearchFilters()
MODEL = "fake-model"
TODAY = dt.date(2026, 11, 9)


async def add_chunk(
    session: AsyncSession,
    text: str,
    *,
    axis: int | None = None,
    title: str = "General Policy",
    heading: str = "",
    doc_type: str = "policy",
    doc_date: dt.date | None = None,
    model: str = MODEL,
) -> uuid.UUID:
    unique = uuid.uuid4().hex
    document = Document(
        source_file=f"{unique}.pdf",
        title=title,
        doc_type=doc_type,
        doc_date=doc_date,
        file_hash=unique,
        text_hash=unique,
        pipeline_version=1,
    )
    session.add(document)
    await session.flush()
    chunk = Chunk(
        document_id=document.id,
        chunk_index=0,
        text=text,
        heading_path=heading,
        search_title=title,
        token_estimate=10,
        content_hash=unique,
        embedding=axis_vector(axis) if axis is not None else None,
        embedding_model=model if axis is not None else None,
    )
    session.add(chunk)
    await session.commit()
    return chunk.id


def texts(result: RetrievalResult) -> list[str]:
    return [chunk.candidate.text for chunk in result.chunks]


async def test_vector_search_orders_by_similarity(db_session: AsyncSession) -> None:
    near = await add_chunk(db_session, "near", axis=0)
    await add_chunk(db_session, "far", axis=1)
    hits = await vector_search(db_session, axis_vector(0), model=MODEL, filters=NO_FILTERS, limit=5)
    assert hits[0].chunk_id == near and len(hits) == 2


async def test_vector_search_skips_other_models_and_unembedded(db_session: AsyncSession) -> None:
    await add_chunk(db_session, "other model", axis=0, model="old-model")
    await add_chunk(db_session, "not embedded")
    assert (
        await vector_search(db_session, axis_vector(0), model=MODEL, filters=NO_FILTERS, limit=5)
        == []
    )


async def keyword_texts(session: AsyncSession, query: str, limit: int = 5) -> list[str]:
    hits = await keyword_search(session, query, filters=NO_FILTERS, limit=limit)
    return [hit.text for hit in hits]


async def test_keyword_search_matches_word_forms(db_session: AsyncSession) -> None:
    await add_chunk(db_session, "Students must register for the examination in advance.")
    assert await keyword_texts(db_session, "examinations registered") != []


async def test_keyword_search_finds_partial_word_matches(db_session: AsyncSession) -> None:
    await add_chunk(db_session, "The hostel closes at nine.")
    assert await keyword_texts(db_session, "what time does the hostel gate shut") != []


async def test_more_matching_words_rank_higher(db_session: AsyncSession) -> None:
    await add_chunk(db_session, "Library fines are charged daily.")
    await add_chunk(db_session, "Library opening hours are nine to five.")
    results = await keyword_texts(db_session, "library opening hours")
    assert results[0].startswith("Library opening")


async def test_heading_match_outranks_body_match(db_session: AsyncSession) -> None:
    await add_chunk(db_session, "A short remark that mentions refunds once among other things.")
    await add_chunk(db_session, "Payments are returned within a month.", heading="Refunds")
    results = await keyword_texts(db_session, "refunds")
    assert results[0].startswith("Payments")


async def test_document_title_is_searchable(db_session: AsyncSession) -> None:
    await add_chunk(db_session, "Applications open in June.", title="Venture Studio Policy")
    assert await keyword_texts(db_session, "venture studio") != []


async def test_question_of_only_stopwords_returns_nothing(db_session: AsyncSession) -> None:
    await add_chunk(db_session, "Some ordinary content.")
    assert await keyword_texts(db_session, "what is the") == []


async def test_punctuation_in_a_question_is_safe(db_session: AsyncSession) -> None:
    await add_chunk(db_session, "Fees are due in August.")
    assert await keyword_texts(db_session, "fees? (due) & 'august' | !!") != []


async def test_filters_apply_to_both_searches(db_session: AsyncSession) -> None:
    await add_chunk(
        db_session, "exam notice", axis=0, doc_type="notice", doc_date=dt.date(2026, 5, 1)
    )
    await add_chunk(
        db_session, "exam policy", axis=0, doc_type="policy", doc_date=dt.date(2024, 5, 1)
    )
    by_type = SearchFilters(doc_types=("notice",))
    by_date = SearchFilters(date_from=dt.date(2025, 1, 1), date_to=dt.date(2026, 12, 31))
    for filters in (by_type, by_date):
        vector = await vector_search(
            db_session, axis_vector(0), model=MODEL, filters=filters, limit=5
        )
        keyword = await keyword_search(db_session, "exam", filters=filters, limit=5)
        assert [hit.text for hit in vector] == ["exam notice"]
        assert [hit.text for hit in keyword] == ["exam notice"]


QUESTION = "hostel fees"


def retriever(
    *,
    fail_embedding: bool = False,
    reranker: FakeReranker | None = None,
) -> tuple[Retriever, FakeQueryEmbedder]:
    embedder = FakeQueryEmbedder({QUESTION: axis_vector(0)}, fail=fail_embedding)
    return Retriever(embedder, reranker), embedder


def config(mode: str = "hybrid", rerank: bool = False, top_k: int = 5) -> RetrievalConfig:
    return RetrievalConfig(mode=mode, rerank=rerank, top_k=top_k)  # type: ignore[arg-type]


async def seed_modes(session: AsyncSession) -> None:
    await add_chunk(session, "semantic only passage", axis=0)
    await add_chunk(session, "Hostel fees are payable termly.", axis=5)
    await add_chunk(session, "unrelated", axis=9)


async def test_each_mode_uses_the_right_search(db_session: AsyncSession) -> None:
    await seed_modes(db_session)
    service, _ = retriever()

    vector = await service.search(db_session, QUESTION, config=config("vector"), today=TODAY)
    assert texts(vector)[0] == "semantic only passage"
    assert "keyword" not in vector.timings_ms

    keyword = await service.search(db_session, QUESTION, config=config("keyword"), today=TODAY)
    assert texts(keyword) == ["Hostel fees are payable termly."]
    assert "vector" not in keyword.timings_ms

    hybrid = await service.search(db_session, QUESTION, config=config("hybrid"), today=TODAY)
    assert {"semantic only passage", "Hostel fees are payable termly."} <= set(texts(hybrid))


async def test_chunk_found_by_both_searches_ranks_first(db_session: AsyncSession) -> None:
    await add_chunk(db_session, "semantic only", axis=0)
    both = await add_chunk(db_session, "Hostel fees and charges.", axis=1)
    await add_chunk(db_session, "Hostel rules.", axis=2)
    service, _ = retriever()
    result = await service.search(db_session, QUESTION, config=config(), today=TODAY)
    first = result.chunks[0]
    assert first.candidate.chunk_id == both
    assert first.vector_rank is not None and first.keyword_rank is not None


async def test_rerank_decides_final_order(db_session: AsyncSession) -> None:
    await seed_modes(db_session)
    reranker = FakeReranker(prefer="unrelated")
    service, _ = retriever(reranker=reranker)
    result = await service.search(db_session, QUESTION, config=config(rerank=True), today=TODAY)
    assert texts(result)[0] == "unrelated"
    assert result.chunks[0].rerank_score == 0.9
    assert all("General Policy" in document for document in reranker.received)


async def test_only_the_shortlist_reaches_the_reranker(db_session: AsyncSession) -> None:
    for number in range(6):
        await add_chunk(db_session, f"chunk {number}", axis=number)
    reranker = FakeReranker()
    service, _ = retriever(reranker=reranker)
    limited = RetrievalConfig(mode="vector", rerank=True, rerank_candidates=3)
    await service.search(db_session, QUESTION, config=limited, today=TODAY)
    assert len(reranker.received) == 3


async def test_rerank_failure_falls_back_to_fused_order(db_session: AsyncSession) -> None:
    await seed_modes(db_session)
    service, _ = retriever(reranker=FakeReranker(fail=True))
    result = await service.search(db_session, QUESTION, config=config(rerank=True), today=TODAY)
    assert result.degraded == ["rerank"]
    assert result.chunks and result.chunks[0].rerank_score is None


async def test_missing_reranker_is_reported(db_session: AsyncSession) -> None:
    await seed_modes(db_session)
    service, _ = retriever(reranker=None)
    result = await service.search(db_session, QUESTION, config=config(rerank=True), today=TODAY)
    assert result.degraded == ["rerank"]


async def test_hybrid_falls_back_to_keyword_when_embedding_fails(db_session: AsyncSession) -> None:
    await seed_modes(db_session)
    service, _ = retriever(fail_embedding=True)
    result = await service.search(db_session, QUESTION, config=config(), today=TODAY)
    assert result.degraded == ["vector"]
    assert texts(result) == ["Hostel fees are payable termly."]


async def test_vector_only_raises_when_embedding_fails(db_session: AsyncSession) -> None:
    service, _ = retriever(fail_embedding=True)
    with pytest.raises(RetrievalError):
        await service.search(db_session, QUESTION, config=config("vector"), today=TODAY)


async def test_top_k_limits_results(db_session: AsyncSession) -> None:
    for number in range(5):
        await add_chunk(db_session, f"hostel fees {number}", axis=number)
    service, _ = retriever()
    result = await service.search(db_session, QUESTION, config=config(top_k=2), today=TODAY)
    assert len(result.chunks) == 2


async def test_newer_document_wins_an_equal_match(db_session: AsyncSession) -> None:
    await add_chunk(db_session, "old hostel fees notice", doc_date=dt.date(2023, 1, 1))
    await add_chunk(db_session, "new hostel fees notice", doc_date=dt.date(2026, 10, 1))
    service, _ = retriever()
    result = await service.search(db_session, QUESTION, config=config("keyword"), today=TODAY)
    assert texts(result)[0] == "new hostel fees notice"


async def test_repeated_question_is_embedded_once(db_session: AsyncSession) -> None:
    await seed_modes(db_session)
    service, embedder = retriever()
    await service.search(db_session, QUESTION, config=config("vector"), today=TODAY)
    await service.search(db_session, QUESTION, config=config("hybrid"), today=TODAY)
    assert embedder.query_calls == 1


async def test_no_matches_returns_empty(db_session: AsyncSession) -> None:
    service, _ = retriever()
    result = await service.search(db_session, QUESTION, config=config(), today=TODAY)
    assert result.chunks == []


async def test_timings_are_recorded(db_session: AsyncSession) -> None:
    await seed_modes(db_session)
    service, _ = retriever(reranker=FakeReranker())
    result = await service.search(db_session, QUESTION, config=config(rerank=True), today=TODAY)
    assert {"embed", "vector", "keyword", "rerank"} <= set(result.timings_ms)
