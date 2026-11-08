"""Google Gemini provider."""

import logging
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import errors, types

from app.llm.base import (
    LLMError,
    LLMEvent,
    LLMRateLimitError,
    LLMUnavailableError,
    LLMUsage,
    StreamEnd,
    TextDelta,
)

logger = logging.getLogger(__name__)


class GeminiProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        thinking_level: str = "",
        client: Any | None = None,
    ) -> None:
        self._client = client or genai.Client(api_key=api_key)
        self._model = model
        self._thinking_level = thinking_level

    @property
    def model(self) -> str:
        return self._model

    def _config(self, system: str, max_output_tokens: int) -> types.GenerateContentConfig:
        thinking = (
            types.ThinkingConfig(thinking_level=self._thinking_level)  # type: ignore[arg-type]
            if self._thinking_level
            else None
        )
        return types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_output_tokens,
            temperature=0.1,  # factual answers should vary as little as possible
            thinking_config=thinking,
        )

    async def stream(
        self, *, system: str, prompt: str, max_output_tokens: int
    ) -> AsyncIterator[LLMEvent]:
        usage: LLMUsage | None = None
        truncated = False
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=prompt,
                config=self._config(system, max_output_tokens),
            )
            async for chunk in stream:
                if chunk.usage_metadata is not None:
                    usage = _to_usage(chunk.usage_metadata)
                for candidate in chunk.candidates or []:
                    if candidate.finish_reason == types.FinishReason.MAX_TOKENS:
                        truncated = True
                if chunk.text:
                    yield TextDelta(chunk.text)
        except errors.ClientError as error:
            if error.code == 429:
                raise LLMRateLimitError("Gemini quota exceeded") from error
            logger.error("Gemini rejected the request: %s %s", error.code, error.message)
            raise LLMError(f"Gemini rejected the request ({error.code})") from error
        except errors.ServerError as error:
            raise LLMUnavailableError(f"Gemini is unavailable ({error.code})") from error
        except errors.APIError as error:
            raise LLMError(f"Gemini failed ({error.code})") from error
        yield StreamEnd(usage=usage, truncated=truncated)


def _to_usage(metadata: types.GenerateContentResponseUsageMetadata) -> LLMUsage:
    # Thinking tokens are billed as output, so they belong in the cost figure.
    output = (metadata.candidates_token_count or 0) + (metadata.thoughts_token_count or 0)
    return LLMUsage(input_tokens=metadata.prompt_token_count or 0, output_tokens=output)
