"""Upload rejections, all of which happen before any parsing or database work."""

from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_session
from app.main import app

KEY = {"X-Admin-Key": "secret-key"}
FORM = {"doc_type": "notice"}


@pytest.fixture(autouse=True)
def configured() -> None:
    async def session() -> AsyncIterator[AsyncSession]:
        yield cast(AsyncSession, AsyncMock())

    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        database_url="postgresql+asyncpg://u:p@localhost/db",
        admin_api_key="secret-key",
        max_upload_mb=1,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session] = session


async def test_missing_key_is_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/admin/documents", data=FORM, files={"file": ("a.pdf", b"%PDF-1.7")}
    )
    assert response.status_code == 401


async def test_wrong_key_is_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/admin/documents",
        data=FORM,
        files={"file": ("a.pdf", b"%PDF-1.7")},
        headers={"X-Admin-Key": "nope"},
    )
    assert response.status_code == 401


async def test_upload_is_disabled_without_configured_key(client: AsyncClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,  # type: ignore[call-arg]
        database_url="postgresql+asyncpg://u:p@localhost/db",
        admin_api_key="",
    )
    response = await client.post(
        "/admin/documents", data=FORM, files={"file": ("a.pdf", b"%PDF-1.7")}, headers=KEY
    )
    assert response.status_code == 503


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        ("virus.exe", b"MZ", 415),
        ("fake.pdf", b"not a pdf", 415),
        ("fake.docx", b"not a zip", 415),
        ("binary.txt", b"abc\x00def", 415),
        ("empty.txt", b"", 415),
        (".hidden.pdf", b"%PDF-1.7", 422),
    ],
)
async def test_bad_files_are_rejected(
    client: AsyncClient, name: str, content: bytes, expected: int
) -> None:
    response = await client.post(
        "/admin/documents", data=FORM, files={"file": (name, content)}, headers=KEY
    )
    assert response.status_code == expected


async def test_oversized_file_is_rejected(client: AsyncClient) -> None:
    big = b"%PDF-" + b"x" * (1024 * 1024 + 1)
    response = await client.post(
        "/admin/documents", data=FORM, files={"file": ("big.pdf", big)}, headers=KEY
    )
    assert response.status_code == 413


async def test_unknown_doc_type_is_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/admin/documents",
        data={"doc_type": "memo"},
        files={"file": ("a.pdf", b"%PDF-1.7")},
        headers=KEY,
    )
    assert response.status_code == 422
