"""Runs against the real Compose database: pytest -m integration."""

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.integration


async def test_readiness_against_real_database(client: AsyncClient) -> None:
    response = await client.get("/health/ready")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ready"
    assert body["pgvector_version"]
