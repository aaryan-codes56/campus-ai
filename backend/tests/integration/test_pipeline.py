"""The whole pipeline against real Postgres, with only the embedding provider faked."""

import hashlib
from collections.abc import Awaitable, Callable
from datetime import date
from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.admin import get_optional_embedder
from app.core.config import Settings, get_settings
from app.db.models import Chunk, Document
from app.db.session import get_session, get_sessionmaker
from app.embeddings.base import EmbeddingError
from app.ingestion.manifest import DocumentMeta
from app.ingestion.pipeline import embed_pending, ingest_directory, ingest_file
from app.main import app
from tests.helpers import BOLD, FakeEmbedder, body_lines, write_pdf

pytestmark = pytest.mark.integration

META = DocumentMeta(title="Test Policy", doc_type="policy")


def policy_pdf(path: Path, marker: str = "Original") -> Path:
    items = [
        (72, 60, "Ref No: RU/RO/PO/2026-27/1 Date: 05.03.2026", 11, "helv"),
        (72, 100, "1. Purpose", 11, BOLD),
        *body_lines(120, 10, f"{marker} policy text about eligibility."),
        (72, 300, "Contact 9876543210 or someone@example.edu for help.", 11, "helv"),
    ]
    return write_pdf(path, [items])


async def count(session: AsyncSession, model: type) -> int:
    return await session.scalar(select(func.count()).select_from(model)) or 0


async def no_wait(_: float) -> None:
    return None


async def test_ingest_stores_metadata_and_pages(db_session: AsyncSession, tmp_path: Path) -> None:
    path = policy_pdf(tmp_path / "policy.pdf")
    result = await ingest_file(db_session, path, meta=META, mask=True)

    assert result.status == "created"
    document = await db_session.scalar(select(Document))
    assert document is not None
    assert (document.title, document.doc_type) == ("Test Policy", "policy")
    assert (document.doc_date, document.date_source) == (date(2026, 3, 5), "printed")
    assert document.ref_no == "RU/RO/PO/2026-27/1"
    chunk = await db_session.scalar(select(Chunk).where(Chunk.heading_path == "1. Purpose"))
    assert chunk is not None
    assert (chunk.page_start, chunk.page_end) == (1, 1)


async def test_contact_details_never_reach_the_database(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    result = await ingest_file(db_session, policy_pdf(tmp_path / "p.pdf"), meta=META, mask=True)
    assert result.masked == 2
    stored = " ".join((await db_session.scalars(select(Chunk.text))).all())
    assert "9876543210" not in stored and "someone@example.edu" not in stored


async def test_reingesting_unchanged_file_changes_nothing(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    path = policy_pdf(tmp_path / "p.pdf")
    await ingest_file(db_session, path, meta=META, mask=True)
    ids_before = set((await db_session.scalars(select(Chunk.id))).all())

    result = await ingest_file(db_session, path, meta=META, mask=True)

    assert result.status == "unchanged"
    assert set((await db_session.scalars(select(Chunk.id))).all()) == ids_before


async def test_changed_file_replaces_its_chunks(db_session: AsyncSession, tmp_path: Path) -> None:
    path = policy_pdf(tmp_path / "p.pdf", "Original")
    await ingest_file(db_session, path, meta=META, mask=True)

    policy_pdf(path, "Revised")
    result = await ingest_file(db_session, path, meta=META, mask=True)

    assert result.status == "updated"
    assert await count(db_session, Document) == 1
    stored = " ".join((await db_session.scalars(select(Chunk.text))).all())
    assert "Revised" in stored and "Original" not in stored


async def test_same_text_under_another_name_is_a_duplicate(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    await ingest_file(db_session, policy_pdf(tmp_path / "one.pdf"), meta=META, mask=True)
    result = await ingest_file(db_session, policy_pdf(tmp_path / "two.pdf"), meta=META, mask=True)
    assert result.status == "duplicate"
    assert await count(db_session, Document) == 1


async def test_force_reprocess_reuses_existing_vectors(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    path = policy_pdf(tmp_path / "p.pdf")
    await ingest_file(db_session, path, meta=META, mask=True)
    embedder = FakeEmbedder()
    await embed_pending(db_session, embedder, rpm=100, tpm=100_000, sleep=no_wait)
    calls = len(embedder.calls)

    await ingest_file(db_session, path, meta=META, mask=True, force=True)
    report = await embed_pending(db_session, embedder, rpm=100, tpm=100_000, sleep=no_wait)

    assert report.embedded == 0
    assert len(embedder.calls) == calls


async def test_new_embedding_model_triggers_reembedding(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    await ingest_file(db_session, policy_pdf(tmp_path / "p.pdf"), meta=META, mask=True)
    await embed_pending(db_session, FakeEmbedder("old"), rpm=100, tpm=100_000, sleep=no_wait)
    report = await embed_pending(
        db_session, FakeEmbedder("new"), rpm=100, tpm=100_000, sleep=no_wait
    )
    assert report.embedded == await count(db_session, Chunk)


async def test_interrupted_embedding_resumes_where_it_stopped(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    pages = [body_lines(100, 40, f"Page {number} sentence about rules.") for number in range(6)]
    await ingest_file(db_session, write_pdf(tmp_path / "long.pdf", pages), meta=META, mask=True)
    total = await count(db_session, Chunk)
    assert total > 2

    failing = FakeEmbedder(fail_on_call=2)
    with pytest.raises(EmbeddingError):
        await embed_pending(db_session, failing, rpm=100, tpm=500, sleep=no_wait)
    await db_session.rollback()
    done = len(failing.calls[0])

    resumed = FakeEmbedder()
    report = await embed_pending(db_session, resumed, rpm=100, tpm=100_000, sleep=no_wait)
    assert report.embedded == total - done


async def test_one_bad_file_does_not_stop_the_rest(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    policy_pdf(tmp_path / "good.pdf")
    (tmp_path / "bad.pdf").write_bytes(b"this is not a pdf")
    results, _ = await ingest_directory(db_session, tmp_path, {"good.pdf": META}, mask=True)
    assert {result.source_file: result.status for result in results} == {
        "bad.pdf": "failed",
        "good.pdf": "created",
    }


async def test_prune_removes_only_files_that_are_gone(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    keep = policy_pdf(tmp_path / "keep.pdf", "Keep")
    remove = policy_pdf(tmp_path / "remove.pdf", "Remove")
    await ingest_directory(db_session, tmp_path, {}, mask=True)
    remove.unlink()

    _, pruned = await ingest_directory(db_session, tmp_path, {}, mask=True, prune=True)

    assert pruned == ["remove.pdf"]
    assert await db_session.scalar(select(Document.source_file)) == keep.name
    assert await count(db_session, Document) == 1


async def test_prune_with_empty_folder_deletes_nothing(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    await ingest_file(db_session, policy_pdf(tmp_path / "p.pdf"), meta=META, mask=True)
    empty = tmp_path / "empty"
    empty.mkdir()
    _, pruned = await ingest_directory(db_session, empty, {}, mask=True, prune=True)
    assert pruned == []
    assert await count(db_session, Document) == 1


@pytest.fixture
def upload_app(
    tmp_path: Path, session_factory: async_sessionmaker[AsyncSession]
) -> Callable[[], Awaitable[None]]:
    async def session():  # type: ignore[no-untyped-def]
        async with session_factory() as db:
            yield db

    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        database_url="postgresql+asyncpg://u:p@localhost/db",
        admin_api_key="secret-key",
        raw_data_dir=tmp_path / "raw",
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session] = session
    app.dependency_overrides[get_sessionmaker] = lambda: session_factory
    app.dependency_overrides[get_optional_embedder] = lambda: FakeEmbedder()

    async def noop() -> None:
        return None

    return noop


async def test_upload_stores_file_and_embeds(
    client: AsyncClient,
    upload_app: Callable[[], Awaitable[None]],
    tmp_path: Path,
    db_session: AsyncSession,
) -> None:
    source = policy_pdf(tmp_path / "source.pdf")
    response = await client.post(
        "/admin/documents",
        data={"doc_type": "notice", "title": "Uploaded Notice"},
        files={"file": ("../../uploaded.pdf", source.read_bytes())},
        headers={"X-Admin-Key": "secret-key"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "created"
    assert response.json()["embedding"] == "scheduled"
    raw = tmp_path / "raw"
    assert [path.name for path in raw.iterdir()] == ["uploaded.pdf"]
    assert hashlib.sha256((raw / "uploaded.pdf").read_bytes()).hexdigest()
    document = await db_session.scalar(select(Document))
    assert document is not None and document.title == "Uploaded Notice"
    pending = await db_session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.embedding.is_(None))
    )
    assert pending == 0


async def test_failed_upload_leaves_no_file_behind(
    client: AsyncClient, upload_app: Callable[[], Awaitable[None]], tmp_path: Path
) -> None:
    response = await client.post(
        "/admin/documents",
        data={"doc_type": "notice"},
        files={"file": ("broken.pdf", b"%PDF-1.7 but actually broken")},
        headers={"X-Admin-Key": "secret-key"},
    )
    assert response.status_code == 422
    assert list((tmp_path / "raw").iterdir()) == []
