"""LLM adapters, using stand-ins for the vendor SDK clients."""

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from google.genai import errors, types

from app.core.config import Settings
from app.llm.base import (
    LLMError,
    LLMEvent,
    LLMRateLimitError,
    LLMUnavailableError,
    LLMUsage,
    StreamEnd,
    TextDelta,
)
from app.llm.claude import ClaudeProvider
from app.llm.factory import build_llm
from app.llm.gemini import GeminiProvider


async def drain(stream: AsyncIterator[LLMEvent]) -> list[LLMEvent]:
    return [event async for event in stream]


def gemini_chunk(
    text: str | None,
    finish: types.FinishReason | None = None,
    usage: tuple[int, int, int] | None = None,
) -> SimpleNamespace:
    metadata = (
        SimpleNamespace(
            prompt_token_count=usage[0],
            candidates_token_count=usage[1],
            thoughts_token_count=usage[2],
        )
        if usage
        else None
    )
    return SimpleNamespace(
        text=text, usage_metadata=metadata, candidates=[SimpleNamespace(finish_reason=finish)]
    )


def gemini_provider(chunks: list[Any] | None = None, error: Exception | None = None) -> Any:
    async def generate() -> AsyncIterator[Any]:
        for chunk in chunks or []:
            yield chunk

    async def generate_content_stream(**kwargs: Any) -> AsyncIterator[Any]:
        client.last_kwargs = kwargs
        if error is not None:
            raise error
        return generate()

    client: Any = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content_stream=generate_content_stream)
        ),
        last_kwargs={},
    )
    return GeminiProvider("key", "gemini-test", thinking_level="low", client=client), client


async def test_gemini_streams_text_and_counts_thinking_as_output() -> None:
    provider, client = gemini_provider(
        [gemini_chunk("Hello "), gemini_chunk(None), gemini_chunk("there", usage=(10, 5, 3))]
    )
    events = await drain(provider.stream(system="S", prompt="P", max_output_tokens=50))
    assert events == [
        TextDelta("Hello "),
        TextDelta("there"),
        StreamEnd(LLMUsage(10, 8), truncated=False),
    ]
    config = client.last_kwargs["config"]
    assert config.system_instruction == "S" and config.max_output_tokens == 50
    assert client.last_kwargs["contents"] == "P"


async def test_gemini_reports_truncation() -> None:
    provider, _ = gemini_provider([gemini_chunk("Hi", finish=types.FinishReason.MAX_TOKENS)])
    events = await drain(provider.stream(system="S", prompt="P", max_output_tokens=5))
    assert events[-1] == StreamEnd(None, truncated=True)


def gemini_thinking_config(level: str) -> Any:
    provider = GeminiProvider("key", "gemini-test", thinking_level=level, client=object())
    return provider._config("S", 10).thinking_config


def test_gemini_thinking_level_is_optional() -> None:
    assert gemini_thinking_config("") is None
    assert gemini_thinking_config("low") is not None


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (errors.ClientError(429, {"error": {"message": "quota"}}), LLMRateLimitError),
        (errors.ClientError(403, {"error": {"message": "no"}}), LLMError),
        (errors.ServerError(503, {"error": {"message": "busy"}}), LLMUnavailableError),
    ],
)
async def test_gemini_errors_are_translated(error: Exception, expected: type[Exception]) -> None:
    provider, _ = gemini_provider(error=error)
    with pytest.raises(expected) as caught:
        await drain(provider.stream(system="S", prompt="P", max_output_tokens=5))
    assert type(caught.value) is expected


class FakeClaudeStream:
    def __init__(self, texts: list[str], stop_reason: str = "end_turn") -> None:
        self._texts = texts
        self._stop_reason = stop_reason

    async def __aenter__(self) -> "FakeClaudeStream":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    @property
    async def text_stream(self) -> AsyncIterator[str]:
        for text in self._texts:
            yield text

    async def get_final_message(self) -> Any:
        return SimpleNamespace(
            stop_reason=self._stop_reason,
            usage=SimpleNamespace(input_tokens=7, output_tokens=3),
        )


def claude_provider(stream: Any = None, error: Exception | None = None) -> Any:
    def start(**kwargs: Any) -> Any:
        client.last_kwargs = kwargs
        if error is not None:
            raise error
        return stream

    client: Any = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(stream=start)), last_kwargs={}
    )
    return ClaudeProvider("key", "claude-test", effort="low", client=client), client


async def test_claude_streams_text_and_usage() -> None:
    provider, client = claude_provider(FakeClaudeStream(["Hi ", "there"]))
    events = await drain(provider.stream(system="S", prompt="P", max_output_tokens=50))
    assert events == [TextDelta("Hi "), TextDelta("there"), StreamEnd(LLMUsage(7, 3), False)]
    assert client.last_kwargs["system"] == "S"
    assert client.last_kwargs["output_config"] == {"effort": "low"}
    assert "temperature" not in client.last_kwargs


async def test_claude_reports_truncation() -> None:
    provider, _ = claude_provider(FakeClaudeStream(["Hi"], stop_reason="max_tokens"))
    events = await drain(provider.stream(system="S", prompt="P", max_output_tokens=5))
    assert events[-1] == StreamEnd(LLMUsage(7, 3), truncated=True)


async def test_claude_refusal_is_an_error() -> None:
    provider, _ = claude_provider(FakeClaudeStream([], stop_reason="refusal"))
    with pytest.raises(LLMError):
        await drain(provider.stream(system="S", prompt="P", max_output_tokens=5))


def claude_status_error(cls: type[anthropic.APIStatusError], code: int) -> Exception:
    request = httpx2.Request("POST", "https://example.test")
    return cls("failed", response=httpx2.Response(code, request=request), body=None)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (claude_status_error(anthropic.RateLimitError, 429), LLMRateLimitError),
        (claude_status_error(anthropic.InternalServerError, 500), LLMUnavailableError),
        (claude_status_error(anthropic.BadRequestError, 400), LLMError),
        (
            anthropic.APIConnectionError(request=httpx2.Request("POST", "https://example.test")),
            LLMUnavailableError,
        ),
    ],
)
async def test_claude_errors_are_translated(error: Exception, expected: type[Exception]) -> None:
    provider, _ = claude_provider(error=error)
    with pytest.raises(expected) as caught:
        await drain(provider.stream(system="S", prompt="P", max_output_tokens=5))
    assert type(caught.value) is expected


def settings(**overrides: Any) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        database_url="postgresql+asyncpg://u:p@localhost/db",
        **overrides,
    )


def test_factory_selects_the_configured_provider() -> None:
    gemini = build_llm(settings(llm_provider="gemini", gemini_api_key="k"))
    claude = build_llm(settings(llm_provider="anthropic", anthropic_api_key="k"))
    assert isinstance(gemini, GeminiProvider) and isinstance(claude, ClaudeProvider)


def test_factory_returns_none_without_a_key() -> None:
    assert build_llm(settings(llm_provider="gemini", gemini_api_key="")) is None
    assert build_llm(settings(llm_provider="anthropic", anthropic_api_key="")) is None
