"""ORM models for source documents and their searchable chunks."""

import uuid
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Computed,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy import func as sql_func
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

# Changing this means changing the column too, so it needs a migration.
EMBEDDING_DIMENSION = 1024
DOC_TYPES = ("notice", "syllabus", "calendar", "policy", "paper")

# Weight A (title, headings) counts for more than weight B (body text).
SEARCH_VECTOR_SQL = (
    "setweight(to_tsvector('english', search_title), 'A') || "
    "setweight(to_tsvector('english', heading_path), 'A') || "
    "setweight(to_tsvector('english', text), 'B')"
)


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "doc_type IN ('notice', 'syllabus', 'calendar', 'policy', 'paper')", name="doc_type"
        ),
        UniqueConstraint("source_file"),
        UniqueConstraint("text_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source_file: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    doc_type: Mapped[str] = mapped_column(Text)
    doc_date: Mapped[date | None] = mapped_column(Date)
    # Where doc_date came from, so a wrong date can be traced: manifest, printed or metadata.
    date_source: Mapped[str | None] = mapped_column(Text)
    ref_no: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    file_hash: Mapped[str] = mapped_column(Text)
    text_hash: Mapped[str] = mapped_column(Text)
    pipeline_version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sql_func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sql_func.now(), onupdate=sql_func.now()
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index"),
        Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    heading_path: Mapped[str] = mapped_column(Text, default="")
    # A copy of the document title, because a generated column cannot read another table.
    search_title: Mapped[str] = mapped_column(Text, default="", server_default="")
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(SEARCH_VECTOR_SQL, persisted=True),
        deferred=True,
    )
    # Null for formats without reliable pages (DOCX, HTML, TXT); those cite a section.
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    token_estimate: Mapped[int] = mapped_column(Integer)
    # Hash of the exact text that gets embedded, so unchanged chunks reuse their vector.
    content_hash: Mapped[str] = mapped_column(Text)
    # Null until embedded, which makes interrupted embedding runs resumable.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSION))
    embedding_model: Mapped[str | None] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="chunks")
