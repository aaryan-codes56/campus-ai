"""The two underlying searches, both run inside Postgres."""

from typing import Any

from sqlalchemy import Select, String, cast, func, select, text
from sqlalchemy.dialects.postgresql import TSQUERY
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.retrieval.types import Candidate, SearchFilters

_COLUMNS = (
    Chunk.id,
    Chunk.document_id,
    Chunk.text,
    Chunk.heading_path,
    Chunk.page_start,
    Chunk.page_end,
    Document.title,
    Document.doc_type,
    Document.doc_date,
    Document.source_file,
    Document.url,
)


def _apply_filters(
    statement: Select[*tuple[Any, ...]], filters: SearchFilters
) -> Select[*tuple[Any, ...]]:
    if filters.doc_types:
        statement = statement.where(Document.doc_type.in_(filters.doc_types))
    if filters.date_from is not None:
        statement = statement.where(Document.doc_date >= filters.date_from)
    if filters.date_to is not None:
        statement = statement.where(Document.doc_date <= filters.date_to)
    return statement


def _to_candidate(row: Row[Any]) -> Candidate:
    return Candidate(
        chunk_id=row.id,
        document_id=row.document_id,
        text=row.text,
        heading_path=row.heading_path,
        page_start=row.page_start,
        page_end=row.page_end,
        title=row.title,
        doc_type=row.doc_type,
        doc_date=row.doc_date,
        source_file=row.source_file,
        url=row.url,
    )


async def vector_search(
    session: AsyncSession,
    vector: list[float],
    *,
    model: str,
    filters: SearchFilters,
    limit: int,
) -> list[Candidate]:
    """Find chunks whose meaning is closest to the query, by cosine distance."""
    if filters.active:
        # Without this, an approximate index can return too few rows once filtered.
        await session.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
    statement = (
        select(*_COLUMNS)
        .join(Document, Chunk.document_id == Document.id)
        # Vectors from different models are not comparable.
        .where(Chunk.embedding.is_not(None), Chunk.embedding_model == model)
        .order_by(Chunk.embedding.cosine_distance(vector))
        .limit(limit)
    )
    rows = await session.execute(_apply_filters(statement, filters))
    return [_to_candidate(row) for row in rows]


def _any_word_query(query: str) -> Any:
    """Match chunks containing any word of the question.

    plainto_tsquery joins words with AND, and a natural question rarely has every
    word in one chunk. Swapping to OR finds candidates, and ranking sorts them out.
    """
    plain = cast(func.plainto_tsquery("english", query), String)
    return cast(func.replace(plain, " & ", " | "), TSQUERY)


async def keyword_search(
    session: AsyncSession,
    query: str,
    *,
    filters: SearchFilters,
    limit: int,
) -> list[Candidate]:
    """Find chunks sharing words with the query, with heading matches weighted higher."""
    tsquery = _any_word_query(query)
    # Normalisation 1 divides by log(length), so long chunks do not win by size alone.
    rank = func.ts_rank(Chunk.search_vector, tsquery, 1)
    statement = (
        select(*_COLUMNS)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.search_vector.op("@@")(tsquery))
        .order_by(rank.desc(), Chunk.id)
        .limit(limit)
    )
    rows = await session.execute(_apply_filters(statement, filters))
    return [_to_candidate(row) for row in rows]
