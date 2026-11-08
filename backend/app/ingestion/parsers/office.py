"""Parsers for DOCX, HTML and plain text. None of these have reliable page numbers."""

from pathlib import Path

from bs4 import BeautifulSoup, Tag
from docx import Document as open_docx
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.ingestion.cleaning import normalize_bullet, normalize_text
from app.ingestion.types import Block, ParsedDocument


def _markdown_table(rows: list[list[str]]) -> Block | None:
    rows = [row for row in rows if any(row)]
    if len(rows) < 2 or len(rows[0]) < 2:
        return None
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return Block("table", "\n".join(lines))


def _cell(text: str) -> str:
    return normalize_text(text.replace("\n", " ")).replace("|", "/")


def parse_docx(path: Path) -> ParsedDocument:
    document = open_docx(str(path))
    blocks: list[Block] = []
    for item in document.iter_inner_content():
        if isinstance(item, Table):
            table = _markdown_table([[_cell(c.text) for c in row.cells] for row in item.rows])
            if table is not None:
                blocks.append(table)
        elif isinstance(item, Paragraph):
            text = normalize_text(item.text)
            if not text:
                continue
            style = item.style.name if item.style is not None and item.style.name else ""
            if style.startswith("Heading"):
                suffix = style.removeprefix("Heading").strip()
                blocks.append(Block("heading", text, None, int(suffix) if suffix.isdigit() else 1))
            elif style == "Title":
                blocks.append(Block("heading", text, None, 1))
            else:
                blocks.append(Block("text", text))
    return ParsedDocument(blocks=blocks)


def parse_html(path: Path) -> ParsedDocument:
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    for element in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
        element.decompose()

    blocks: list[Block] = []
    root = soup.body or soup
    wanted = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"]
    for element in root.find_all(wanted):
        if not isinstance(element, Tag):
            continue
        if element.name != "table" and element.find_parent("table") is not None:
            continue
        if element.name == "table":
            rows = [
                [_cell(cell.get_text(" ")) for cell in row.find_all(["th", "td"])]
                for row in element.find_all("tr")
            ]
            table = _markdown_table(rows)
            if table is not None:
                blocks.append(table)
            continue
        text = normalize_text(element.get_text(" "))
        if not text:
            continue
        if element.name.startswith("h"):
            blocks.append(Block("heading", text, None, int(element.name[1])))
        elif element.name == "li":
            blocks.append(Block("text", normalize_bullet(f"• {text}")))
        else:
            blocks.append(Block("text", text))
    return ParsedDocument(blocks=blocks)


def parse_text(path: Path) -> ParsedDocument:
    raw = path.read_bytes()
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = raw.decode("latin-1")
    paragraphs = [normalize_text(" ".join(part.split())) for part in content.split("\n\n")]
    return ParsedDocument(blocks=[Block("text", text) for text in paragraphs if text])
