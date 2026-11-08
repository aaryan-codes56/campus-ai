"""Turn retrieved chunks into a streamed, cited answer, or a clear refusal."""

import asyncio
import datetime as dt
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from app.generation.citations import CitationFilter
from app.generation.prompts import REFUSAL_MESSAGE, SYSTEM_PROMPT, build_user_prompt, is_refusal
from app.llm.base import (
    LLMError,
    LLMProvider,
    LLMRateLimitError,
    LLMUnavailableError,
    LLMUsage,
    TextDelta,
)
from app.retrieval.types import RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2
SNIPPET_CHARS = 240

AnswerStatus = Literal["answered", "refused", "uncited"]
RefusalReason = Literal["no_sources", "low_relevance", "model"]
ErrorCode = Literal["rate_limited", "unavailable", "generation_failed"]


@dataclass(frozen=True)
class Citation:
    number: int  # the [n] used in the answer
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    title: str
    source_file: str
    doc_type: str
    doc_date: dt.date | None
    page_start: int | None
    page_end: int | None
    section: str
    url: str | None
    snippet: str


@dataclass(frozen=True)
class TokenEvent:
    text: str


@dataclass(frozen=True)
class CitationsEvent:
    citations: list[Citation]


@dataclass(frozen=True)
class DoneEvent:
    status: AnswerStatus
    refusal_reason: RefusalReason | None = None
    model: str | None = None
    usage: LLMUsage | None = None
    truncated: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ErrorEvent:
    code: ErrorCode


AnswerEvent = TokenEvent | CitationsEvent | DoneEvent | ErrorEvent


def _citation(number: int, chunk: RetrievedChunk) -> Citation:
    item = chunk.candidate
    return Citation(
        number=number,
        chunk_id=item.chunk_id,
        document_id=item.document_id,
        title=item.title,
        source_file=item.source_file,
        doc_type=item.doc_type,
        doc_date=item.doc_date,
        page_start=item.page_start,
        page_end=item.page_end,
        section=item.heading_path,
        url=item.url,
        snippet=item.text[:SNIPPET_CHARS],
    )


class AnswerGenerator:
    def __init__(
        self,
        llm: LLMProvider,
        *,
        min_rerank_score: float,
        max_output_tokens: int,
        retry_delay: float = 2.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._llm = llm
        self._min_rerank_score = min_rerank_score
        self._max_output_tokens = max_output_tokens
        self._retry_delay = retry_delay
        self._sleep = sleep

    def _gate(self, retrieval: RetrievalResult) -> RefusalReason | None:
        """Refuse without calling the LLM when retrieval found nothing usable.

        Cheaper and faster, and the model cannot be tempted to answer from weak text.
        If re-ranking failed there is no reliable score, so the model decides.
        """
        if not retrieval.chunks:
            return "no_sources"
        scores = [chunk.rerank_score for chunk in retrieval.chunks]
        known = [score for score in scores if score is not None]
        if known and max(known) < self._min_rerank_score:
            return "low_relevance"
        return None

    async def stream(
        self, question: str, retrieval: RetrievalResult, *, today: dt.date | None = None
    ) -> AsyncIterator[AnswerEvent]:
        timings = dict(retrieval.timings_ms)

        reason = self._gate(retrieval)
        if reason is not None:
            yield TokenEvent(REFUSAL_MESSAGE)
            yield CitationsEvent([])
            yield DoneEvent(
                status="refused",
                refusal_reason=reason,
                timings_ms=timings,
                degraded=retrieval.degraded,
            )
            return

        prompt = build_user_prompt(question, retrieval.chunks, today=today or dt.date.today())
        citations = CitationFilter(len(retrieval.chunks))
        answer = ""
        usage: LLMUsage | None = None
        truncated = False
        started = time.perf_counter()
        first_token_ms: float | None = None
        received_text = False

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                async for event in self._llm.stream(
                    system=SYSTEM_PROMPT,
                    prompt=prompt,
                    max_output_tokens=self._max_output_tokens,
                ):
                    if isinstance(event, TextDelta):
                        received_text = True
                        visible = citations.feed(event.text)
                        if visible:
                            if first_token_ms is None:
                                first_token_ms = _elapsed_ms(started)
                            answer += visible
                            yield TokenEvent(visible)
                    else:
                        usage, truncated = event.usage, event.truncated
                break
            except (LLMRateLimitError, LLMUnavailableError) as error:
                # Only safe before any text: a retry mid-answer would show text twice.
                if received_text or attempt == MAX_ATTEMPTS:
                    logger.warning("LLM failed after %d attempts: %s", attempt, error)
                    yield ErrorEvent(
                        "rate_limited" if isinstance(error, LLMRateLimitError) else "unavailable"
                    )
                    return
                await self._sleep(self._retry_delay)
            except LLMError as error:
                logger.error("LLM request failed: %s", error)
                yield ErrorEvent("generation_failed")
                return

        remainder = citations.finish()
        if remainder:
            answer += remainder
            yield TokenEvent(remainder)
        if not answer.strip():
            logger.error("LLM returned no text")
            yield ErrorEvent("generation_failed")
            return

        timings["first_token"] = first_token_ms or _elapsed_ms(started)
        timings["generation"] = _elapsed_ms(started)

        refused = is_refusal(answer) and not citations.cited
        status: AnswerStatus = (
            "refused" if refused else "answered" if citations.cited else "uncited"
        )
        yield CitationsEvent(
            []
            if refused
            else [_citation(number, retrieval.chunks[number - 1]) for number in citations.cited]
        )
        yield DoneEvent(
            status=status,
            refusal_reason="model" if refused else None,
            model=self._llm.model,
            usage=usage,
            truncated=truncated,
            timings_ms=timings,
            degraded=retrieval.degraded,
        )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
