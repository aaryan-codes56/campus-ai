"""Structure-aware chunking: split on sections first, then by size, with overlap.

Sizes are estimated as characters / 4, a standard approximation for English. It
avoids depending on any one provider's tokenizer, and limits are set low enough
that the estimate's error cannot cause a problem.
"""

import math
import re
from dataclasses import dataclass, field

from app.ingestion.cleaning import fingerprint
from app.ingestion.types import Block

SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9\"“(\[-])")
HEADING_SEPARATOR = " > "
MIN_ALNUM_CHARS = 3


@dataclass(frozen=True)
class ChunkConfig:
    target_tokens: int = 450
    max_tokens: int = 600
    overlap_tokens: int = 70
    min_section_tokens: int = 80


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    heading_path: str
    page_start: int | None
    page_end: int | None
    token_estimate: int


@dataclass(frozen=True)
class _Unit:
    text: str
    page: int | None
    paragraph: int = 0  # sentences of one paragraph rejoin with a space, not a break


@dataclass
class _Section:
    path: tuple[str, ...]
    blocks: list[Block] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(estimate_tokens(block.text) for block in self.blocks)

    @property
    def has_table(self) -> bool:
        return any(block.kind == "table" for block in self.blocks)


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def build_embedding_text(title: str, heading_path: str, text: str) -> str:
    """Prefix a chunk with where it came from, so it is still findable out of context."""
    header = f"{title}\n{heading_path}" if heading_path else title
    return f"{header}\n\n{text}"


def _build_sections(blocks: list[Block]) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    current = _Section(path=())
    for block in blocks:
        if block.kind != "heading":
            # Skip stray glyphs: lone bullets and letters from decorative slide text.
            if sum(char.isalnum() for char in block.text) >= MIN_ALNUM_CHARS:
                current.blocks.append(block)
            continue
        if current.blocks:
            sections.append(current)
        while stack and stack[-1][0] >= block.level:
            stack.pop()
        stack.append((block.level, block.text))
        current = _Section(path=tuple(text for _, text in stack))
    if current.blocks:
        sections.append(current)
    return sections


def _can_merge(previous: _Section, section: _Section, config: ChunkConfig) -> bool:
    if previous.has_table or section.has_table:
        return False
    if previous.tokens + section.tokens > config.target_tokens:
        return False
    previous_small = previous.tokens < config.min_section_tokens
    section_small = section.tokens < config.min_section_tokens
    if previous_small and section_small:
        return True  # two fragments are worth more together, even across headings
    # Text before the first heading has no parent, so it joins whatever follows.
    related = previous.path[:1] == section.path[:1] or not previous.path
    return related and (previous_small or section_small)


def _merge_small_sections(sections: list[_Section], config: ChunkConfig) -> list[_Section]:
    """Fold very short sections into a neighbour, keeping the heading as text."""
    merged: list[_Section] = []
    for section in sections:
        previous = merged[-1] if merged else None
        if previous is not None and _can_merge(previous, section, config):
            if not previous.path:
                # Preamble takes on the first real heading, which retrieval needs.
                previous.path = section.path
                previous.blocks.extend(section.blocks)
                continue
            label = section.path[-1] if section.path else ""
            if label:
                previous.blocks.append(Block("text", label, section.blocks[0].page))
            previous.blocks.extend(section.blocks)
        else:
            merged.append(section)
    return merged


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """Last resort for text with no sentence breaks: split between words."""
    pieces: list[str] = []
    current: list[str] = []
    for word in text.split():
        if current and estimate_tokens(" ".join([*current, word])) > max_tokens:
            pieces.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _split_paragraph(text: str, target: int, maximum: int) -> list[str]:
    if estimate_tokens(text) <= target:
        return [text]
    pieces: list[str] = []
    for sentence in SENTENCE_END.split(text):
        if estimate_tokens(sentence) <= maximum:
            pieces.append(sentence)
        else:
            pieces.extend(_hard_split(sentence, maximum))
    return pieces


def _overlap_tail(text: str, overlap_tokens: int) -> str:
    """Take whole trailing sentences that fit, so overlap never starts mid-sentence."""
    kept: list[str] = []
    used = 0
    for sentence in reversed(SENTENCE_END.split(text)):
        cost = estimate_tokens(sentence)
        if used + cost > overlap_tokens:
            break
        kept.insert(0, sentence)
        used += cost
    return " ".join(kept)


def _pack_units(units: list[_Unit], config: ChunkConfig) -> list[list[_Unit]]:
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    tokens = 0
    fresh = False  # True once `current` holds something beyond carried-over overlap
    for unit in units:
        cost = estimate_tokens(unit.text)
        if current and fresh and tokens + cost > config.target_tokens:
            groups.append(current)
            last = current[-1]
            tail = _overlap_tail(last.text, config.overlap_tokens)
            carried = estimate_tokens(tail)
            if tail and carried + cost <= config.max_tokens:
                current, tokens = [_Unit(tail, last.page, last.paragraph)], carried
            else:
                current, tokens = [], 0
            fresh = False
        current.append(unit)
        tokens += cost
        fresh = True
    if current and fresh:
        groups.append(current)
    return groups


def _split_table(text: str, config: ChunkConfig) -> list[str]:
    """Split between rows and repeat the header, so every piece stays readable."""
    lines = text.splitlines()
    header, rows = lines[:2], lines[2:]
    header_tokens = estimate_tokens("\n".join(header))
    room = config.max_tokens - header_tokens

    expanded: list[str] = []
    for row in rows:
        expanded.extend([row] if estimate_tokens(row) <= room else _hard_split(row, room))

    pieces: list[str] = []
    current: list[str] = []
    tokens = header_tokens
    for row in expanded:
        cost = estimate_tokens(row) + 1
        if current and tokens + cost > config.target_tokens:
            pieces.append("\n".join([*header, *current]))
            current, tokens = [], header_tokens
        current.append(row)
        tokens += cost
    if current:
        pieces.append("\n".join([*header, *current]))
    return pieces


def _join_units(group: list[_Unit]) -> str:
    text = group[0].text
    for previous, unit in zip(group, group[1:], strict=False):
        text += (" " if unit.paragraph == previous.paragraph else "\n\n") + unit.text
    return text


def _drafts_from_units(
    units: list[_Unit], path: str, config: ChunkConfig
) -> list[tuple[str, str, list[int]]]:
    drafts = []
    for group in _pack_units(units, config):
        pages = [unit.page for unit in group if unit.page is not None]
        drafts.append((path, _join_units(group), pages))
    return drafts


def chunk_blocks(blocks: list[Block], config: ChunkConfig | None = None) -> list[Chunk]:
    config = config or ChunkConfig()
    drafts: list[tuple[str, str, list[int]]] = []

    for section in _merge_small_sections(_build_sections(blocks), config):
        path = HEADING_SEPARATOR.join(section.path)
        units: list[_Unit] = []
        for paragraph, block in enumerate(section.blocks):
            if block.kind == "table":
                drafts.extend(_drafts_from_units(units, path, config))
                units = []
                pages = [block.page] if block.page is not None else []
                drafts.extend((path, piece, pages) for piece in _split_table(block.text, config))
            else:
                pieces = _split_paragraph(block.text, config.target_tokens, config.max_tokens)
                units.extend(_Unit(piece, block.page, paragraph) for piece in pieces)
        drafts.extend(_drafts_from_units(units, path, config))

    chunks: list[Chunk] = []
    seen: set[str] = set()
    for path, text, pages in drafts:
        key = fingerprint(text)
        if key in seen:
            continue
        seen.add(key)
        chunks.append(
            Chunk(
                index=len(chunks),
                text=text,
                heading_path=path,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                token_estimate=estimate_tokens(text),
            )
        )
    return chunks
