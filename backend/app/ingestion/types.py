"""Shared structures passed between the parsers and the chunker."""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

BlockKind = Literal["heading", "text", "table"]


@dataclass(frozen=True)
class Block:
    """One structural piece of a document, in reading order."""

    kind: BlockKind
    text: str
    page: int | None = None
    level: int = 0  # headings only: 1 is the outermost


@dataclass
class ParsedDocument:
    blocks: list[Block]
    page_count: int | None = None
    # Pages with almost no text, usually image slides. Reported, never silently lost.
    low_text_pages: list[int] = field(default_factory=list)
    metadata_date: date | None = None
