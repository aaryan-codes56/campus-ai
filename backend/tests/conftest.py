"""Shared test setup. Settings need a database URL before the app is imported."""

import os
from collections.abc import AsyncIterator
from pathlib import Path

# Environment variables outrank .env, so only supply a dummy URL when there is no
# .env to read. Otherwise integration tests would lose the real database URL.
if not (Path(__file__).parents[2] / ".env").exists():
    os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:1/test")
os.environ.setdefault("APP_ENV", "test")

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client
    app.dependency_overrides.clear()
