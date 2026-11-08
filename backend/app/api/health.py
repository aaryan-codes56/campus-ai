"""Liveness and readiness probes.

/health only proves the process is up, so it stays cheap and is used to wake a
sleeping server. /health/ready also proves the database and pgvector work.
"""

import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    database: bool
    pgvector_version: str | None


@router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/health/ready")
async def readiness(
    response: Response, session: Annotated[AsyncSession, Depends(get_session)]
) -> ReadinessResponse:
    try:
        result = await session.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
        version = result.scalar_one_or_none()
    except (SQLAlchemyError, OSError):
        # OSError covers a refused connection, which the driver raises unwrapped.
        logger.exception("Readiness check could not reach the database")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(status="not_ready", database=False, pgvector_version=None)

    if version is None:
        logger.error("Database reachable but pgvector is missing; run the migrations")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(status="not_ready", database=True, pgvector_version=None)

    return ReadinessResponse(status="ready", database=True, pgvector_version=str(version))
