"""The streaming endpoint, with retrieval and the LLM faked."""

import json
from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.chat import get_generator, get_retriever
from app.db.session import get_session
from app.generation.service import AnswerGenerator
from app.llm.base import LLMError
from app.main import app
from app.retrieval.service import RetrievalError
from app.retrieval.types import RetrievalResult, SearchFilters
from tests.generation.fakes import FakeLLM, make_chunk, make_retrieval


class FakeRetriever:
    def __init__(self, result: RetrievalResult | None = None, error: bool = False) -> None:
        self.result = result or make_retrieval(make_chunk())
        self.error = error
        self.filters: SearchFilters | None = None

    async def search(
        self, session: AsyncSession, query: str, *, filters: SearchFilters
    ) -> RetrievalResult:
        self.filters = filters
        if self.error:
            raise RetrievalError("down")
        return self.result


@pytest.fixture
def db_session() -> AsyncMock:
    session = AsyncMock()

    async def override() -> AsyncIterator[AsyncSession]:
        yield cast(AsyncSession, session)

    app.dependency_overrides[get_session] = override
    return session


def configure(llm: FakeLLM | None, retriever: FakeRetriever | None = None) -> FakeRetriever:
    fake = retriever or FakeRetriever()
    generator = AnswerGenerator(llm, min_rerank_score=0.3, max_output_tokens=100) if llm else None
    app.dependency_overrides[get_retriever] = lambda: fake
    app.dependency_overrides[get_generator] = lambda: generator
    return fake


def parse_events(body: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.split("\n")
        events.append(
            (lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: ")))
        )
    return events


async def post(client: AsyncClient, **body: Any) -> Any:
    return await client.post("/chat/stream", json=body)


async def test_stream_sends_tokens_citations_then_done(
    client: AsyncClient, db_session: AsyncMock
) -> None:
    configure(FakeLLM(["Due in ", "August [1]."]))
    response = await post(client, question="When are fees due?")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_events(response.text)
    assert [name for name, _ in events] == ["token", "token", "citations", "done"]
    assert "".join(data["text"] for name, data in events if name == "token") == "Due in August [1]."
    citation = events[2][1]["citations"][0]
    assert citation["number"] == 1 and citation["page_start"] == 3
    assert citation["doc_date"] == "2026-05-01"
    assert events[3][1]["status"] == "answered"


async def test_connection_is_released_before_streaming(
    client: AsyncClient, db_session: AsyncMock
) -> None:
    configure(FakeLLM(["Due [1]."]))
    await post(client, question="When?")
    db_session.rollback.assert_awaited_once()


async def test_refusal_is_streamed(client: AsyncClient, db_session: AsyncMock) -> None:
    configure(FakeLLM(), FakeRetriever(make_retrieval()))
    events = parse_events((await post(client, question="Unknown?")).text)
    assert events[-1][1]["status"] == "refused"


async def test_llm_failure_is_reported_inside_the_stream(
    client: AsyncClient, db_session: AsyncMock
) -> None:
    configure(FakeLLM([LLMError("nope")]))
    response = await post(client, question="When?")
    assert response.status_code == 200
    name, data = parse_events(response.text)[-1]
    assert name == "error" and data["code"] == "generation_failed"
    assert "nope" not in data["message"]


async def test_filters_reach_retrieval(client: AsyncClient, db_session: AsyncMock) -> None:
    retriever = configure(FakeLLM(["Due [1]."]))
    await post(
        client,
        question="When?",
        filters={"doc_types": ["calendar"], "date_from": "2026-01-01"},
    )
    assert retriever.filters is not None
    assert retriever.filters.doc_types == ("calendar",)
    assert retriever.filters.date_from is not None


async def test_retrieval_failure_is_a_503(client: AsyncClient, db_session: AsyncMock) -> None:
    configure(FakeLLM(), FakeRetriever(error=True))
    assert (await post(client, question="When?")).status_code == 503


async def test_missing_llm_is_a_503(client: AsyncClient, db_session: AsyncMock) -> None:
    configure(None)
    assert (await post(client, question="When?")).status_code == 503


@pytest.mark.parametrize(
    "body",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 1001},
        {},
        {"question": "ok", "extra": 1},
        {"question": "ok", "filters": {"doc_types": ["memo"]}},
    ],
)
async def test_invalid_requests_are_rejected(
    client: AsyncClient, db_session: AsyncMock, body: dict[str, Any]
) -> None:
    configure(FakeLLM())
    assert (await client.post("/chat/stream", json=body)).status_code == 422
