"""Streaming answers over Server-Sent Events."""

import dataclasses
import datetime as dt
import json
import logging
import uuid
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_session
from app.generation.service import (
    AnswerEvent,
    AnswerGenerator,
    CitationsEvent,
    DoneEvent,
    ErrorEvent,
    TokenEvent,
)
from app.ingestion.manifest import DocType
from app.llm.factory import build_llm
from app.retrieval.service import RetrievalError, Retriever, build_retriever
from app.retrieval.types import SearchFilters

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])

ERROR_MESSAGES = {
    "rate_limited": "The assistant is getting a lot of questions right now. "
    "Please try again in a minute.",
    "unavailable": "The assistant is temporarily unavailable. Please try again shortly.",
    "generation_failed": "Something went wrong while writing the answer. Please try again.",
}


class ChatFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_types: list[DocType] = Field(default_factory=list, max_length=5)
    date_from: dt.date | None = None
    date_to: dt.date | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=1000)
    filters: ChatFilters = Field(default_factory=ChatFilters)


@lru_cache
def get_retriever() -> Retriever:
    return build_retriever(get_settings())


@lru_cache
def get_generator() -> AnswerGenerator | None:
    settings = get_settings()
    llm = build_llm(settings)
    if llm is None:
        return None
    return AnswerGenerator(
        llm,
        min_rerank_score=settings.refusal_min_rerank_score,
        max_output_tokens=settings.llm_max_output_tokens,
    )


def _json_default(value: Any) -> str:
    if isinstance(value, uuid.UUID | dt.date):
        return str(value)
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=_json_default)}\n\n"


def _encode(event: AnswerEvent) -> str:
    if isinstance(event, TokenEvent):
        return _sse("token", {"text": event.text})
    if isinstance(event, CitationsEvent):
        return _sse("citations", {"citations": [dataclasses.asdict(c) for c in event.citations]})
    if isinstance(event, DoneEvent):
        return _sse("done", dataclasses.asdict(event))
    assert isinstance(event, ErrorEvent)
    return _sse("error", {"code": event.code, "message": ERROR_MESSAGES[event.code]})


async def _event_stream(events: AsyncIterator[AnswerEvent]) -> AsyncIterator[str]:
    try:
        async for event in events:
            yield _encode(event)
    except Exception:
        # Headers have already gone out, so failures must travel inside the stream.
        logger.exception("Answer stream failed")
        yield _encode(ErrorEvent("generation_failed"))


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    retriever: Annotated[Retriever, Depends(get_retriever)],
    generator: Annotated[AnswerGenerator | None, Depends(get_generator)],
) -> StreamingResponse:
    if generator is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The LLM is not configured")

    filters = SearchFilters(
        doc_types=tuple(body.filters.doc_types),
        date_from=body.filters.date_from,
        date_to=body.filters.date_to,
    )
    # Retrieve before streaming, so a failure here can still be a normal HTTP error.
    try:
        retrieval = await retriever.search(session, body.question, filters=filters)
    except RetrievalError as error:
        logger.error("Retrieval failed: %s", error)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Search is temporarily unavailable"
        ) from error
    # Give the database connection back: an answer can stream for many seconds.
    await session.rollback()

    return StreamingResponse(
        _event_stream(generator.stream(body.question, retrieval)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
