"""Health endpoints, with the database replaced by a small stand-in session."""

from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import AsyncMock, MagicMock

from httpx import AsyncClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.main import app


def use_session(session: AsyncMock) -> None:
    async def override() -> AsyncIterator[AsyncSession]:
        yield cast(AsyncSession, session)

    app.dependency_overrides[get_session] = override


def session_returning(version: str | None) -> AsyncMock:
    result = MagicMock()
    result.scalar_one_or_none.return_value = version
    session = AsyncMock()
    session.execute.return_value = result
    return session


async def test_health_needs_no_database(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ready_when_pgvector_installed(client: AsyncClient) -> None:
    use_session(session_returning("0.8.1"))
    response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "database": True, "pgvector_version": "0.8.1"}


async def test_not_ready_when_pgvector_missing(client: AsyncClient) -> None:
    use_session(session_returning(None))
    response = await client.get("/health/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "database": True, "pgvector_version": None}


async def test_not_ready_when_database_down(client: AsyncClient) -> None:
    session = AsyncMock()
    session.execute.side_effect = OperationalError("SELECT 1", {}, Exception("down"))
    use_session(session)
    response = await client.get("/health/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "database": False, "pgvector_version": None}


async def test_not_ready_when_connection_refused(client: AsyncClient) -> None:
    session = AsyncMock()
    session.execute.side_effect = ConnectionRefusedError()
    use_session(session)
    response = await client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["database"] is False
