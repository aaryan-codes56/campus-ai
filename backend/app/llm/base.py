"""The small interface answer generation uses, so providers can be swapped by config."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


class LLMError(RuntimeError):
    """The provider refused or failed in a way that retrying will not fix."""


class LLMRateLimitError(LLMError):
    """The provider's quota is used up for now."""


class LLMUnavailableError(LLMError):
    """The provider is temporarily unreachable or overloaded."""


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class StreamEnd:
    usage: LLMUsage | None
    truncated: bool  # True when the output hit the length limit


LLMEvent = TextDelta | StreamEnd


class LLMProvider(Protocol):
    @property
    def model(self) -> str: ...

    def stream(
        self, *, system: str, prompt: str, max_output_tokens: int
    ) -> AsyncIterator[LLMEvent]: ...
