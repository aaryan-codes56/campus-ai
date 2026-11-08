"""The chunker's guarantees: size limits, section boundaries, pages and overlap."""

from app.ingestion.chunking import (
    ChunkConfig,
    build_embedding_text,
    chunk_blocks,
    estimate_tokens,
)
from app.ingestion.types import Block

CONFIG = ChunkConfig(target_tokens=60, max_tokens=80, overlap_tokens=15, min_section_tokens=10)


def sentence(number: int) -> str:
    return f"Sentence number {number} explains one rule of the university policy clearly."


def long_text(count: int, start: int = 0) -> str:
    return " ".join(sentence(start + index) for index in range(count))


def test_no_chunk_exceeds_max_tokens() -> None:
    blocks = [Block("heading", "Rules", 1, 1), Block("text", long_text(40), 1)]
    chunks = chunk_blocks(blocks, CONFIG)
    assert len(chunks) > 1
    assert all(chunk.token_estimate <= CONFIG.max_tokens for chunk in chunks)


def test_all_source_text_is_kept() -> None:
    blocks = [Block("text", long_text(25), 1)]
    combined = " ".join(chunk.text for chunk in chunk_blocks(blocks, CONFIG))
    assert all(sentence(index) in combined for index in range(25))


def test_consecutive_chunks_overlap_by_whole_sentences() -> None:
    config = ChunkConfig(target_tokens=60, max_tokens=80, overlap_tokens=25)
    chunks = chunk_blocks([Block("text", long_text(30), 1)], config)
    tail = chunks[0].text[chunks[0].text.rfind("Sentence number") :]
    assert chunks[1].text.startswith(tail)


def test_sections_do_not_mix() -> None:
    blocks = [
        Block("heading", "1. Fees", 1, 1),
        Block("text", long_text(3), 1),
        Block("heading", "2. Hostel", 2, 1),
        Block("text", long_text(3, start=100), 2),
    ]
    chunks = chunk_blocks(blocks, CONFIG)
    assert [chunk.heading_path for chunk in chunks] == ["1. Fees", "2. Hostel"]
    assert sentence(100) not in chunks[0].text


def test_heading_path_tracks_nesting() -> None:
    blocks = [
        Block("heading", "4. Definitions", 1, 1),
        Block("heading", "4.1 Achiever", 1, 2),
        Block("text", long_text(3), 1),
        Block("heading", "5. Categories", 2, 1),
        Block("text", long_text(3, start=50), 2),
    ]
    paths = [chunk.heading_path for chunk in chunk_blocks(blocks, CONFIG)]
    assert paths == ["4. Definitions > 4.1 Achiever", "5. Categories"]


def test_pages_span_the_blocks_used() -> None:
    blocks = [Block("text", sentence(1), 3), Block("text", sentence(2), 4)]
    chunk = chunk_blocks(blocks, CONFIG)[0]
    assert (chunk.page_start, chunk.page_end) == (3, 4)


def test_formats_without_pages_keep_none() -> None:
    chunk = chunk_blocks([Block("text", sentence(1))], CONFIG)[0]
    assert chunk.page_start is None and chunk.page_end is None


def test_large_table_repeats_header_in_every_piece() -> None:
    header = "| Event | Date |\n| --- | --- |"
    rows = "\n".join(f"| Examination for batch number {i} | 0{i % 9 + 1}-Dec |" for i in range(40))
    chunks = chunk_blocks([Block("table", f"{header}\n{rows}", 2)], CONFIG)
    assert len(chunks) > 1
    assert all(chunk.text.startswith(header) for chunk in chunks)
    assert all(chunk.token_estimate <= CONFIG.max_tokens for chunk in chunks)


def test_table_rows_are_never_lost() -> None:
    header = "| Event | Date |\n| --- | --- |"
    rows = [f"| Event number {i} padded out with words | Day {i} |" for i in range(30)]
    combined = "\n".join(
        chunk.text
        for chunk in chunk_blocks([Block("table", "\n".join([header, *rows]), 1)], CONFIG)
    )
    assert all(row in combined for row in rows)


def test_tiny_section_folds_into_next_and_keeps_its_heading() -> None:
    blocks = [
        Block("heading", "Stage 1", 1, 1),
        Block("heading", "Who it is for", 1, 3),
        Block("text", "Students.", 1),
        Block("heading", "What you get", 1, 3),
        Block("text", long_text(2), 1),
    ]
    chunks = chunk_blocks(blocks, CONFIG)
    assert len(chunks) == 1
    assert "What you get" in chunks[0].text and "Students." in chunks[0].text


def test_stray_glyphs_are_ignored() -> None:
    blocks = [Block("text", "- ", 1), Block("text", "th", 1), Block("text", sentence(1), 1)]
    chunks = chunk_blocks(blocks, CONFIG)
    assert len(chunks) == 1 and chunks[0].text == sentence(1)


def test_preamble_joins_the_first_section() -> None:
    blocks = [
        Block("text", "Date: 01.01.2026", 1),
        Block("heading", "Office Order", 1, 1),
        Block("text", sentence(1), 1),
    ]
    chunks = chunk_blocks(blocks, CONFIG)
    assert len(chunks) == 1
    assert "Date: 01.01.2026" in chunks[0].text
    assert chunks[0].heading_path == "Office Order"


def test_neighbouring_fragments_merge_across_headings() -> None:
    blocks = [
        Block("heading", "Tokyo", 3, 1),
        Block("text", "Robotics visit", 3),
        Block("heading", "Osaka", 4, 1),
        Block("text", "Castle tour", 4),
    ]
    chunks = chunk_blocks(blocks, CONFIG)
    assert len(chunks) == 1 and "Osaka" in chunks[0].text


def test_identical_chunks_in_one_document_are_removed() -> None:
    blocks = [
        Block("heading", "A", 1, 1),
        Block("text", long_text(3), 1),
        Block("heading", "B", 2, 1),
        Block("text", long_text(3), 2),
    ]
    chunks = chunk_blocks(blocks, CONFIG)
    assert len(chunks) == 1
    assert [chunk.index for chunk in chunks] == [0]


def test_text_with_no_sentence_breaks_is_still_split() -> None:
    chunks = chunk_blocks([Block("text", "word " * 900, 1)], CONFIG)
    assert all(chunk.token_estimate <= CONFIG.max_tokens for chunk in chunks)


def test_empty_input_gives_no_chunks() -> None:
    assert chunk_blocks([], CONFIG) == []


def test_embedding_text_includes_title_and_section() -> None:
    text = build_embedding_text("Leave Policy", "2. Eligibility", "Body")
    assert text == "Leave Policy\n2. Eligibility\n\nBody"
    assert build_embedding_text("Leave Policy", "", "Body") == "Leave Policy\n\nBody"


def test_estimate_tokens_rounds_up() -> None:
    assert estimate_tokens("abcde") == 2
