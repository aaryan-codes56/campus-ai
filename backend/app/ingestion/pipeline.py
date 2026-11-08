"""Ingestion: file -> parsed structure -> cleaned chunks -> database -> embeddings."""

import asyncio
import datetime as dt
import hashlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.embeddings.base import Embedder
from app.ingestion.chunking import Chunk as DraftChunk
from app.ingestion.chunking import build_embedding_text, chunk_blocks, estimate_tokens
from app.ingestion.cleaning import (
    extract_printed_date,
    extract_ref_no,
    fingerprint,
    mask_contact_details,
)
from app.ingestion.manifest import DocumentMeta, normalize_name
from app.ingestion.parsers.office import parse_docx, parse_html, parse_text
from app.ingestion.parsers.pdf import parse_pdf
from app.ingestion.types import Block, ParsedDocument

logger = logging.getLogger(__name__)

# Bump when parsing or chunking changes, so existing documents are re-processed.
PIPELINE_VERSION = 1

PARSERS: dict[str, Callable[[Path], ParsedDocument]] = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".html": parse_html,
    ".htm": parse_html,
    ".txt": parse_text,
}
SUPPORTED_SUFFIXES = frozenset(PARSERS)
MAX_TEXTS_PER_BATCH = 128

FileStatus = Literal["created", "updated", "unchanged", "duplicate", "failed"]


@dataclass
class FileResult:
    source_file: str
    status: FileStatus
    pages: int | None = None
    chunks: int = 0
    masked: int = 0
    low_text_pages: list[int] = field(default_factory=list)
    detail: str = ""


@dataclass
class PreparedDocument:
    parsed: ParsedDocument
    chunks: list[DraftChunk]
    masked: int
    text_hash: str
    printed_date: dt.date | None
    ref_no: str | None


def list_source_files(raw_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_document(path: Path, *, mask: bool) -> PreparedDocument:
    """Parse, clean and chunk one file. Touches neither the database nor the network."""
    parsed = PARSERS[path.suffix.lower()](path)
    # The hash is taken before masking, so masking settings cannot hide a duplicate.
    text_hash = fingerprint(" ".join(block.text for block in parsed.blocks))
    opening = " ".join(block.text for block in parsed.blocks[:12])

    masked = 0
    blocks: list[Block] = []
    for block in parsed.blocks:
        if mask:
            text, count = mask_contact_details(block.text)
            masked += count
            blocks.append(replace(block, text=text))
        else:
            blocks.append(block)

    return PreparedDocument(
        parsed=parsed,
        chunks=chunk_blocks(blocks),
        masked=masked,
        text_hash=text_hash,
        printed_date=extract_printed_date(opening),
        ref_no=extract_ref_no(opening),
    )


def _meta_changed(document: Document, meta: DocumentMeta | None) -> bool:
    if meta is None:
        return False
    return (
        document.title != meta.title
        or document.doc_type != meta.doc_type
        or document.url != meta.url
        or (meta.doc_date is not None and document.doc_date != meta.doc_date)
    )


def _resolve_date(
    meta: DocumentMeta | None, prepared: PreparedDocument, existing: Document | None
) -> tuple[dt.date | None, str | None]:
    if meta is not None and meta.doc_date is not None:
        return meta.doc_date, "manifest"
    if prepared.printed_date is not None:
        return prepared.printed_date, "printed"
    if prepared.parsed.metadata_date is not None:
        return prepared.parsed.metadata_date, "metadata"
    if existing is not None:
        return existing.doc_date, existing.date_source
    return None, None


async def ingest_file(
    session: AsyncSession,
    path: Path,
    *,
    meta: DocumentMeta | None,
    mask: bool,
    force: bool = False,
    source_file: str | None = None,
) -> FileResult:
    """Store one file and commit. Re-running on an unchanged file does nothing."""
    name = normalize_name(source_file or path.name)
    file_hash = await asyncio.to_thread(_hash_file, path)
    existing = await session.scalar(select(Document).where(Document.source_file == name))

    if (
        existing is not None
        and not force
        and existing.file_hash == file_hash
        and existing.pipeline_version == PIPELINE_VERSION
        and not _meta_changed(existing, meta)
    ):
        count = await session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == existing.id)
        )
        return FileResult(name, "unchanged", pages=existing.page_count, chunks=count or 0)

    prepared = await asyncio.to_thread(prepare_document, path, mask=mask)
    pages = prepared.parsed.page_count
    if not prepared.chunks:
        return FileResult(name, "failed", pages=pages, detail="no extractable text")

    duplicate_of = await session.scalar(
        select(Document.source_file).where(
            Document.text_hash == prepared.text_hash, Document.source_file != name
        )
    )
    if duplicate_of is not None:
        return FileResult(name, "duplicate", pages=pages, detail=f"same text as {duplicate_of}")

    # A file with no manifest entry keeps the metadata it was uploaded with.
    title = meta.title if meta else (existing.title if existing else Path(name).stem)
    doc_type = meta.doc_type if meta else (existing.doc_type if existing else "notice")
    url = meta.url if meta else (existing.url if existing else None)
    doc_date, date_source = _resolve_date(meta, prepared, existing)

    reusable: dict[str, tuple[list[float], str]] = {}
    if existing is not None:
        rows = await session.execute(
            select(Chunk.content_hash, Chunk.embedding, Chunk.embedding_model).where(
                Chunk.document_id == existing.id, Chunk.embedding.is_not(None)
            )
        )
        reusable = {row.content_hash: (row.embedding, row.embedding_model) for row in rows}
        await session.execute(delete(Chunk).where(Chunk.document_id == existing.id))

    document = existing or Document(source_file=name)
    document.title = title
    document.doc_type = doc_type
    document.doc_date = doc_date
    document.date_source = date_source
    document.ref_no = prepared.ref_no
    document.url = url
    document.page_count = pages
    document.file_hash = file_hash
    document.text_hash = prepared.text_hash
    document.pipeline_version = PIPELINE_VERSION
    session.add(document)
    await session.flush()

    for draft in prepared.chunks:
        embedding_text = build_embedding_text(title, draft.heading_path, draft.text)
        content_hash = hashlib.sha256(embedding_text.encode()).hexdigest()
        vector, model = reusable.get(content_hash, (None, None))
        session.add(
            Chunk(
                document_id=document.id,
                chunk_index=draft.index,
                text=draft.text,
                heading_path=draft.heading_path,
                search_title=title,
                page_start=draft.page_start,
                page_end=draft.page_end,
                token_estimate=draft.token_estimate,
                content_hash=content_hash,
                embedding=vector,
                embedding_model=model,
            )
        )
    await session.commit()

    return FileResult(
        name,
        "updated" if existing is not None else "created",
        pages=pages,
        chunks=len(prepared.chunks),
        masked=prepared.masked,
        low_text_pages=prepared.parsed.low_text_pages,
    )


async def ingest_directory(
    session: AsyncSession,
    raw_dir: Path,
    manifest: dict[str, DocumentMeta],
    *,
    mask: bool,
    force: bool = False,
    prune: bool = False,
) -> tuple[list[FileResult], list[str]]:
    """Ingest every supported file. One bad file is reported and never stops the rest."""
    paths = await asyncio.to_thread(list_source_files, raw_dir)
    results: list[FileResult] = []
    for path in paths:
        name = normalize_name(path.name)
        try:
            result = await ingest_file(
                session, path, meta=manifest.get(name), mask=mask, force=force
            )
        except Exception as error:  # noqa: BLE001 - isolating per-file failures is the point
            await session.rollback()
            logger.exception("Failed to ingest %s", name)
            result = FileResult(name, "failed", detail=type(error).__name__)
        results.append(result)

    pruned: list[str] = []
    # Guard: an empty or wrongly configured folder must never wipe the database.
    if prune and paths:
        names = [normalize_name(path.name) for path in paths]
        deleted = await session.execute(
            delete(Document)
            .where(Document.source_file.not_in(names))
            .returning(Document.source_file)
        )
        pruned = list(deleted.scalars())
        await session.commit()
    return results, pruned


@dataclass
class EmbedReport:
    embedded: int = 0
    batches: int = 0


async def count_pending(session: AsyncSession, model: str) -> int:
    count = await session.scalar(
        select(func.count()).select_from(Chunk).where(_needs_embedding(model))
    )
    return count or 0


def _needs_embedding(model: str):  # type: ignore[no-untyped-def]  # SQLAlchemy expression
    return or_(Chunk.embedding.is_(None), Chunk.embedding_model != model)


async def embed_pending(
    session: AsyncSession,
    embedder: Embedder,
    *,
    rpm: int,
    tpm: int,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_progress: Callable[[int, int], None] | None = None,
) -> EmbedReport:
    """Embed chunks that lack a current vector, committing after every batch.

    Committing per batch makes this resumable: if it stops halfway, the next run
    only pays for what is left.
    """
    rows = (
        await session.execute(
            select(Chunk.id, Chunk.text, Chunk.heading_path, Document.title)
            .join(Document, Chunk.document_id == Document.id)
            .where(_needs_embedding(embedder.model))
            .order_by(Chunk.document_id, Chunk.chunk_index)
        )
    ).all()

    # Leave headroom because token counts are estimates.
    token_budget = int(tpm * 0.8)
    batches: list[list[tuple[object, str]]] = []
    current: list[tuple[object, str]] = []
    current_tokens = 0
    for row in rows:
        text = build_embedding_text(row.title, row.heading_path, row.text)
        tokens = estimate_tokens(text)
        if current and (
            current_tokens + tokens > token_budget or len(current) >= MAX_TEXTS_PER_BATCH
        ):
            batches.append(current)
            current, current_tokens = [], 0
        current.append((row.id, text))
        current_tokens += tokens
    if current:
        batches.append(current)

    report = EmbedReport()
    for number, batch in enumerate(batches):
        if number > 0:
            batch_tokens = sum(estimate_tokens(text) for _, text in batch)
            await sleep(max(60 / rpm, 60 * batch_tokens / tpm))
        vectors = await embedder.embed_documents([text for _, text in batch])
        for (chunk_id, _), vector in zip(batch, vectors, strict=True):
            await session.execute(
                update(Chunk)
                .where(Chunk.id == chunk_id)
                .values(embedding=vector, embedding_model=embedder.model)
            )
        await session.commit()
        report.embedded += len(batch)
        report.batches += 1
        if on_progress is not None:
            on_progress(report.embedded, len(rows))
    return report
