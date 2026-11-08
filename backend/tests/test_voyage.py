"""The Voyage client against a simulated API."""

import json
from typing import Any

import httpx
import pytest

from app.embeddings.base import EmbeddingError
from app.embeddings.voyage import VoyageEmbedder


def make_embedder(handler, sleeps: list[float] | None = None) -> VoyageEmbedder:  # type: ignore[no-untyped-def]
    recorded = sleeps if sleeps is not None else []

    async def sleep(seconds: float) -> None:
        recorded.append(seconds)

    return VoyageEmbedder(
        "key", "voyage-test", 2, transport=httpx.MockTransport(handler), sleep=sleep
    )


def success(indices: list[int]) -> httpx.Response:
    data = [{"index": index, "embedding": [float(index), 0.0]} for index in indices]
    return httpx.Response(200, json={"data": data})


async def test_vectors_are_returned_in_input_order() -> None:
    embedder = make_embedder(lambda request: success([1, 0]))
    assert await embedder.embed_documents(["a", "b"]) == [[0.0, 0.0], [1.0, 0.0]]


async def test_request_marks_input_type_and_refuses_truncation() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        seen["auth"] = request.headers["authorization"]
        return success([0])

    await make_embedder(handler).embed_query("question")
    assert seen["input_type"] == "query"
    assert seen["truncation"] is False
    assert seen["output_dimension"] == 2
    assert seen["auth"] == "Bearer key"


async def test_rate_limit_waits_as_told_then_succeeds() -> None:
    responses = [httpx.Response(429, headers={"retry-after": "7"}), success([0])]
    sleeps: list[float] = []
    embedder = make_embedder(lambda request: responses.pop(0), sleeps)
    assert await embedder.embed_documents(["a"]) == [[0.0, 0.0]]
    assert sleeps == [7.0]


async def test_gives_up_after_repeated_server_errors() -> None:
    embedder = make_embedder(lambda request: httpx.Response(500))
    with pytest.raises(EmbeddingError):
        await embedder.embed_documents(["a"])


async def test_bad_request_is_not_retried() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, text="bad key")

    with pytest.raises(EmbeddingError, match="401"):
        await make_embedder(handler).embed_documents(["a"])
    assert len(calls) == 1


async def test_wrong_vector_count_is_an_error() -> None:
    embedder = make_embedder(lambda request: success([0]))
    with pytest.raises(EmbeddingError):
        await embedder.embed_documents(["a", "b"])


async def test_wrong_dimension_is_an_error() -> None:
    body = {"data": [{"index": 0, "embedding": [1.0, 2.0, 3.0]}]}
    embedder = make_embedder(lambda request: httpx.Response(200, json=body))
    with pytest.raises(EmbeddingError):
        await embedder.embed_documents(["a"])
