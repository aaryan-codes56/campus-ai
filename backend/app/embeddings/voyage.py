"""Voyage AI embeddings and re-ranking over REST.

Uses httpx rather than Voyage's SDK to keep heavy dependencies off a 512 MB server.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import httpx

from app.embeddings.base import EmbeddingError, ProviderError, RerankError, RerankResult

logger = logging.getLogger(__name__)

EMBEDDINGS_URL = "https://api.voyageai.com/v1/embeddings"
RERANK_URL = "https://api.voyageai.com/v1/rerank"
MAX_BACKOFF_SECONDS = 120.0

SleepFn = Callable[[float], Awaitable[None]]


class _VoyageClient:
    def __init__(
        self,
        api_key: str,
        *,
        transport: httpx.AsyncBaseTransport | None,
        max_retries: int,
        base_delay: float,
        sleep: SleepFn,
    ) -> None:
        self._api_key = api_key
        self._transport = transport
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._sleep = sleep

    async def post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with httpx.AsyncClient(transport=self._transport, timeout=60.0) as client:
            for attempt in range(self._max_retries + 1):
                delay = min(self._base_delay * 2**attempt, MAX_BACKOFF_SECONDS)
                try:
                    response = await client.post(url, json=payload, headers=headers)
                except httpx.TransportError as error:
                    if attempt == self._max_retries:
                        raise ProviderError(f"Could not reach Voyage: {error!r}") from error
                    await self._sleep(delay)
                    continue

                if response.status_code == 429 or response.status_code >= 500:
                    if attempt == self._max_retries:
                        break
                    logger.warning("Voyage returned %s; retrying", response.status_code)
                    await self._sleep(_retry_after(response) or delay)
                    continue

                if response.status_code >= 400:
                    raise ProviderError(
                        f"Voyage rejected the request ({response.status_code}): "
                        f"{response.text[:200]}"
                    )
                body: dict[str, Any] = response.json()
                return body
        raise ProviderError(f"Voyage still failing after {self._max_retries} retries")


class VoyageEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str,
        dimension: int,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 5,
        base_delay: float = 20.0,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        self._client = _VoyageClient(
            api_key,
            transport=transport,
            max_retries=max_retries,
            base_delay=base_delay,
            sleep=sleep,
        )
        self._model = model
        self._dimension = dimension

    @property
    def model(self) -> str:
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, "document")

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], "query"))[0]

    async def _embed(
        self, texts: list[str], input_type: Literal["document", "query"]
    ) -> list[list[float]]:
        payload = {
            "input": texts,
            "model": self._model,
            "input_type": input_type,
            "output_dimension": self._dimension,
            # Fail loudly instead of silently embedding only part of a text.
            "truncation": False,
        }
        try:
            body = await self._client.post(EMBEDDINGS_URL, payload)
        except ProviderError as error:
            raise EmbeddingError(str(error)) from error
        return self._parse(body, len(texts))

    def _parse(self, body: dict[str, Any], expected: int) -> list[list[float]]:
        # Sort by index: pairing vectors with the wrong text would corrupt search silently.
        items = sorted(body["data"], key=lambda item: item["index"])
        vectors: list[list[float]] = [item["embedding"] for item in items]
        if len(vectors) != expected:
            raise EmbeddingError(f"Expected {expected} embeddings, received {len(vectors)}")
        if any(len(vector) != self._dimension for vector in vectors):
            raise EmbeddingError(f"Expected vectors of length {self._dimension}")
        return vectors


class VoyageReranker:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 5,
        base_delay: float = 20.0,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        self._client = _VoyageClient(
            api_key,
            transport=transport,
            max_retries=max_retries,
            base_delay=base_delay,
            sleep=sleep,
        )
        self._model = model

    async def rerank(self, query: str, documents: list[str]) -> list[RerankResult]:
        if not documents:
            return []
        payload = {
            "query": query,
            "documents": documents,
            "model": self._model,
            "truncation": False,
        }
        try:
            body = await self._client.post(RERANK_URL, payload)
        except ProviderError as error:
            raise RerankError(str(error)) from error

        results = [
            RerankResult(index=item["index"], score=float(item["relevance_score"]))
            for item in body["data"]
        ]
        if any(not 0 <= result.index < len(documents) for result in results):
            raise RerankError("Voyage returned an index outside the document list")
        return sorted(results, key=lambda result: result.score, reverse=True)


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers["retry-after"])
    except (KeyError, ValueError):
        return None
