"""Anthropic Claude provider."""

from collections.abc import AsyncIterator
from typing import Any

import anthropic

from app.llm.base import (
    LLMError,
    LLMEvent,
    LLMRateLimitError,
    LLMUnavailableError,
    LLMUsage,
    StreamEnd,
    TextDelta,
)

# If a safety classifier declines a request, the API retries it on a fallback model.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        effort: str = "low",
        client: Any | None = None,
    ) -> None:
        self._client: Any = client or anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._effort = effort

    @property
    def model(self) -> str:
        return self._model

    async def stream(
        self, *, system: str, prompt: str, max_output_tokens: int
    ) -> AsyncIterator[LLMEvent]:
        try:
            async with self._client.beta.messages.stream(
                model=self._model,
                max_tokens=max_output_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": self._effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                async for text in stream.text_stream:
                    yield TextDelta(text)
                message = await stream.get_final_message()
        except anthropic.RateLimitError as error:
            raise LLMRateLimitError("Claude rate limit reached") from error
        except anthropic.APIStatusError as error:
            if error.status_code >= 500:
                raise LLMUnavailableError(f"Claude unavailable ({error.status_code})") from error
            raise LLMError(f"Claude rejected the request ({error.status_code})") from error
        except anthropic.APIConnectionError as error:
            raise LLMUnavailableError("Could not reach Claude") from error

        if message.stop_reason == "refusal":
            raise LLMError("Claude declined the request")
        yield StreamEnd(
            usage=LLMUsage(message.usage.input_tokens, message.usage.output_tokens),
            truncated=message.stop_reason == "max_tokens",
        )
