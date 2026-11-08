"""PDF parsing: turn positioned text into headings, paragraphs and tables.

A PDF stores text as positioned fragments, with no notion of "heading" or
"paragraph". Structure has to be inferred from font size, weight and layout.
"""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.ingestion.cleaning import BULLET_START, normalize_bullet, normalize_text, parse_pdf_date
from app.ingestion.types import Block, ParsedDocument

LOW_TEXT_CHARS = 80
EDGE_ZONE_POINTS = 90
RELIABLE_PAGE_CHARS = 300
CALENDAR_MIN_NUMBER_CELLS = 30
ROW_TOLERANCE_POINTS = 3.0

PAGE_NUMBER = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.IGNORECASE)
NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+\S")
NOT_A_HEADING = re.compile(
    r"^(dr\.|prof\.|registrar\b|copy to\b|ref\b|ref\.|date[d]?\b|\[amended\]|name$|designation$)",
    re.IGNORECASE,
)

EMAIL_LINE_WRAP = re.compile(r"(?<=[@.])\s+|\s+(?=[@.])")
SIGNATURE = re.compile(r"^((prof|dr)\b.*|registrar)$", re.IGNORECASE)
GRID_NUMBER = re.compile(r"^\d{1,2}$")
GRID_NOISE = re.compile(
    r"^((sun|mon|tue|wed|thu|fri|sat)(\s+(sun|mon|tue|wed|thu|fri|sat))*"
    r"|[A-Z][a-z]+ \d{4}|academic days\s*-?\s*\d*|index\s*:?|date"
    r"|exams|holiday|academic events|non academic events)$",
    re.IGNORECASE,
)
WEEKDAY_CELL = re.compile(
    r"^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"(\s*-\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Line:
    text: str
    size: float
    bold: bool
    x0: float
    y0: float
    block_no: int


def _table_markdown(rows: list[list[str | None]]) -> str | None:
    cleaned = [
        [normalize_text((cell or "").replace("\n", " ")).replace("|", "/") for cell in row]
        for row in rows
    ]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return None
    # Table detection can invent a column from a stray line; drop any that are fully empty.
    filled = [index for index in range(len(cleaned[0])) if any(row[index] for row in cleaned)]
    cleaned = [[row[index] for index in filled] for row in cleaned]
    if len(cleaned) < 2 or len(cleaned[0]) < 2:
        return None
    header, *body = cleaned
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def _read_lines(page: pymupdf.Page, exclude: list[pymupdf.Rect]) -> list[_Line]:
    lines: list[_Line] = []
    for block_no, block in enumerate(page.get_text("dict")["blocks"]):
        for line in block.get("lines", []):
            spans = line["spans"]
            text = normalize_text("".join(span["text"] for span in spans))
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            center = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
            if any(center in rect for rect in exclude):
                continue
            main = max(spans, key=lambda span: len(span["text"]))
            bold = bool(main["flags"] & 16) or "bold" in main["font"].lower()
            lines.append(_Line(text, round(main["size"], 1), bold, x0, y0, block_no))
    return sorted(lines, key=lambda item: (item.block_no, item.y0))


def _dominant_size(lines: list[_Line]) -> float | None:
    sizes: Counter[float] = Counter()
    for line in lines:
        sizes[line.size] += len(line.text)
    return sizes.most_common(1)[0][0] if sizes else None


def _repeated_edge_lines(pages: list[list[_Line]], heights: list[float]) -> set[str]:
    """Find headers and footers: lines recurring near the page edge on most pages."""
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for lines, height in zip(pages, heights, strict=True):
        counts.update(
            {
                _edge_key(line.text)
                for line in lines
                if line.y0 < EDGE_ZONE_POINTS or line.y0 > height - EDGE_ZONE_POINTS
            }
        )
    threshold = max(3, len(pages) / 2)
    return {key for key, count in counts.items() if count >= threshold and len(key) >= 4}


def _edge_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower())


def _heading_level(line: _Line, body_size: float) -> int | None:
    text = line.text
    if len(text) > 90 or len(text.split()) > 14 or BULLET_START.match(text):
        return None
    if NOT_A_HEADING.match(text) or text[0].islower() or text.endswith((",", ";", ".")):
        return None
    if line.size > body_size + 0.5:
        return 1
    if not line.bold or ":" in text[:-1]:
        return None
    numbered = NUMBERED_HEADING.match(text)
    return min(1 + numbered.group(1).count("."), 3) if numbered else 3


def _join(buffer: list[str]) -> str:
    joined = ""
    for part in buffer:
        if joined.endswith("-") and part[:1].islower():
            joined = joined[:-1] + part  # word split across a line break
        else:
            joined = f"{joined} {part}".strip()
    return normalize_bullet(joined)


def _flowing_blocks(
    lines: list[_Line],
    page_no: int,
    body_size: float,
    tables: list[tuple[pymupdf.Rect, str]],
) -> list[Block]:
    blocks: list[Block] = []
    buffer: list[str] = []
    current_block: int | None = None
    pending_tables = sorted(tables, key=lambda item: item[0].y0)

    def flush() -> None:
        if buffer:
            blocks.append(Block("text", _join(buffer), page_no))
            buffer.clear()

    for line in lines:
        # A table belongs before the first line of text that sits below its top edge.
        while pending_tables and line.y0 > pending_tables[0][0].y0:
            flush()
            blocks.append(Block("table", pending_tables.pop(0)[1], page_no))
        level = _heading_level(line, body_size)
        if level is not None:
            flush()
            blocks.append(Block("heading", line.text, page_no, level))
            current_block = line.block_no
            continue
        only_a_bullet = len(buffer) == 1 and BULLET_START.fullmatch(buffer[0]) is not None
        if not only_a_bullet and (line.block_no != current_block or BULLET_START.match(line.text)):
            flush()
        buffer.append(line.text)
        current_block = line.block_no
    flush()
    blocks.extend(Block("table", markdown, page_no) for _, markdown in pending_tables)
    return blocks


def _is_calendar_page(lines: list[_Line]) -> bool:
    return sum(GRID_NUMBER.match(line.text) is not None for line in lines) >= (
        CALENDAR_MIN_NUMBER_CELLS
    )


def _group_rows(lines: list[_Line]) -> list[list[_Line]]:
    rows: list[list[_Line]] = []
    for line in sorted(lines, key=lambda item: item.y0):
        if rows and abs(rows[-1][0].y0 - line.y0) <= ROW_TOLERANCE_POINTS:
            rows[-1].append(line)
        else:
            rows.append([line])
    return [sorted(row, key=lambda item: item.x0) for row in rows]


def _calendar_blocks(lines: list[_Line], page_no: int) -> list[Block]:
    """Keep a calendar page's event table and drop its month grids.

    The grids are bare day numbers whose meaning was in cell colour, so as text
    they are noise. The event table holds the same dates in readable form.
    """
    blocks: list[Block] = []
    rows: list[str] = []

    def flush() -> None:
        if rows:
            table = ["| Event | Date | Day |", "| --- | --- | --- |", *rows]
            blocks.append(Block("table", "\n".join(table), page_no))
            rows.clear()

    for row in _group_rows(lines):
        cells = [line.text for line in row if not GRID_NUMBER.match(line.text)]
        cells = [cell for cell in cells if not GRID_NOISE.match(cell)]
        if not cells:
            continue
        if len(cells) >= 3 and WEEKDAY_CELL.match(cells[-1]):
            event = " ".join(cells[:-2]).replace("|", "/")
            rows.append(f"| {event} | {cells[-2]} | {cells[-1]} |")
        elif "SEMESTER" in " ".join(cells).upper():
            flush()
            blocks.append(Block("heading", " ".join(cells), page_no, 1))
        elif not SIGNATURE.match(" ".join(cells)):
            flush()
            blocks.append(Block("text", " ".join(cells), page_no))
    flush()
    return blocks


def _cell_text(page: pymupdf.Page, bbox: tuple[float, float, float, float] | None) -> str:
    """Read a cell's text directly from the page.

    table.extract() can reorder punctuation in Word-exported PDFs, turning
    "a.b@x.edu" into "ab@xedu . .", which would also get past contact masking.
    """
    if bbox is None:
        return ""
    text = normalize_text(page.get_text("text", clip=pymupdf.Rect(bbox)).replace("\n", " "))
    return EMAIL_LINE_WRAP.sub("", text) if "@" in text else text


def _find_tables(page: pymupdf.Page) -> list[tuple[pymupdf.Rect, str]]:
    found = []
    for table in page.find_tables().tables:
        rows = [[_cell_text(page, cell) for cell in row.cells] for row in table.rows]
        markdown = _table_markdown(rows)  # type: ignore[arg-type]
        if markdown is not None:
            found.append((pymupdf.Rect(table.bbox), markdown))
    return found


def parse_pdf(path: Path) -> ParsedDocument:
    with pymupdf.open(path) as doc:
        pages = list(doc)
        tables = [_find_tables(page) for page in pages]
        all_lines = [
            _read_lines(page, [rect for rect, _ in page_tables])
            for page, page_tables in zip(pages, tables, strict=True)
        ]
        repeated = _repeated_edge_lines(all_lines, [page.rect.height for page in pages])
        document_size = _dominant_size([line for lines in all_lines for line in lines]) or 0.0

        blocks: list[Block] = []
        low_text_pages: list[int] = []
        for page_no, (lines, page_tables, page) in enumerate(
            zip(all_lines, tables, pages, strict=True), start=1
        ):
            if len(page.get_text("text").strip()) < LOW_TEXT_CHARS:
                low_text_pages.append(page_no)
            # Before the page-number filter below, which would also remove the day numbers.
            is_calendar = _is_calendar_page(lines)
            height = page.rect.height
            kept = [
                line
                for line in lines
                if not PAGE_NUMBER.match(line.text)
                and not (
                    _edge_key(line.text) in repeated
                    and (line.y0 < EDGE_ZONE_POINTS or line.y0 > height - EDGE_ZONE_POINTS)
                )
            ]
            if is_calendar:
                blocks.extend(_calendar_blocks(kept, page_no))
                continue

            # Trust the page's own body size when it has enough text. A cover page
            # in 12pt must not look like all headings beside a 7pt document.
            chars = sum(len(line.text) for line in kept)
            page_size = _dominant_size(kept) if chars >= RELIABLE_PAGE_CHARS else None
            blocks.extend(_flowing_blocks(kept, page_no, page_size or document_size, page_tables))

        return ParsedDocument(
            blocks=blocks,
            page_count=doc.page_count,
            low_text_pages=low_text_pages,
            metadata_date=parse_pdf_date((doc.metadata or {}).get("creationDate")),
        )
