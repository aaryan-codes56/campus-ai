"""Add full-text search and vector indexes.

Revision ID: 0003
Revises: 0002
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE chunks ADD COLUMN search_title text NOT NULL DEFAULT ''")
    op.execute(
        "UPDATE chunks SET search_title = documents.title "
        "FROM documents WHERE documents.id = chunks.document_id"
    )
    op.execute(
        """
        ALTER TABLE chunks ADD COLUMN search_vector tsvector
        GENERATED ALWAYS AS (
            setweight(to_tsvector('english', search_title), 'A') ||
            setweight(to_tsvector('english', heading_path), 'A') ||
            setweight(to_tsvector('english', text), 'B')
        ) STORED
        """
    )
    op.execute("CREATE INDEX ix_chunks_search_vector ON chunks USING gin (search_vector)")
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX ix_chunks_embedding_hnsw")
    op.execute("DROP INDEX ix_chunks_search_vector")
    op.execute("ALTER TABLE chunks DROP COLUMN search_vector")
    op.execute("ALTER TABLE chunks DROP COLUMN search_title")
