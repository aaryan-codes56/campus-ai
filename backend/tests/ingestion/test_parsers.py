"""Parsers, run against real files generated during the test."""

from pathlib import Path

import pymupdf
from docx import Document as new_docx

from app.ingestion.parsers.office import parse_docx, parse_html, parse_text
from app.ingestion.parsers.pdf import parse_pdf
from tests.helpers import BODY, BOLD, PdfItem, body_lines, write_pdf


def test_bold_numbered_line_becomes_heading(tmp_path: Path) -> None:
    items = [(72, 80, "2.1 Eligibility Criteria", 11, BOLD), *body_lines(100, 8)]
    blocks = parse_pdf(write_pdf(tmp_path / "a.pdf", [items])).blocks
    assert (blocks[0].kind, blocks[0].text, blocks[0].level) == (
        "heading",
        "2.1 Eligibility Criteria",
        2,
    )


def test_larger_font_becomes_top_level_heading(tmp_path: Path) -> None:
    items = [(72, 80, "Stage 1 - Idea", 14, BODY), *body_lines(100, 40)]
    blocks = parse_pdf(write_pdf(tmp_path / "a.pdf", [items])).blocks
    assert (blocks[0].kind, blocks[0].level) == ("heading", 1)


def test_bold_sentence_fragment_is_not_a_heading(tmp_path: Path) -> None:
    items = [
        (72, 80, "the VIP Lounge, A-Block.", 11, BOLD),
        (72, 94, "Application Review: as per the schedule", 11, BOLD),
        (72, 108, "Dr. Some Signatory", 11, BOLD),
        *body_lines(130, 8),
    ]
    blocks = parse_pdf(write_pdf(tmp_path / "a.pdf", [items])).blocks
    assert all(block.kind != "heading" for block in blocks)


def test_cover_page_in_normal_font_is_not_all_headings(tmp_path: Path) -> None:
    cover = [(72, 100 + i * 16, f"Cover line {i} of the order", 12, BODY) for i in range(30)]
    small_body = [
        (72, 80 + i * 10, "Very small body text here on this page.", 7, BODY) for i in range(60)
    ]
    blocks = parse_pdf(write_pdf(tmp_path / "a.pdf", [cover, small_body, small_body])).blocks
    assert all(block.kind != "heading" for block in blocks if block.page == 1)


def test_repeated_headers_and_page_numbers_are_removed(tmp_path: Path) -> None:
    pages: list[list[PdfItem]] = [
        [
            (72, 40, "RISHIHOOD UNIVERSITY", 9, BODY),
            *body_lines(120, 8, f"Unique content for page {number}."),
            (72, 800, f"Page {number} of 3", 9, BODY),
        ]
        for number in range(1, 4)
    ]
    text = " ".join(block.text for block in parse_pdf(write_pdf(tmp_path / "a.pdf", pages)).blocks)
    assert "RISHIHOOD" not in text and "Page 2 of 3" not in text
    assert "Unique content for page 2." in text


def test_blocks_carry_their_real_page_numbers(tmp_path: Path) -> None:
    pages = [body_lines(100, 8, "First page words."), body_lines(100, 8, "Second page words.")]
    blocks = parse_pdf(write_pdf(tmp_path / "a.pdf", pages)).blocks
    assert {block.page for block in blocks if "Second" in block.text} == {2}


def test_low_text_pages_are_reported(tmp_path: Path) -> None:
    pages: list[list[PdfItem]] = [body_lines(100, 8), [(72, 100, "Slide", 11, BODY)]]
    assert parse_pdf(write_pdf(tmp_path / "a.pdf", pages)).low_text_pages == [2]


def test_ruled_table_becomes_markdown_in_position(tmp_path: Path) -> None:
    path = tmp_path / "table.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 80), "Text above the table.", fontsize=11)
    for line_y in (100, 120, 140):
        page.draw_line((72, line_y), (372, line_y))
    for line_x in (72, 222, 372):
        page.draw_line((line_x, 100), (line_x, 140))
    page.insert_text((80, 115), "Name", fontsize=10)
    page.insert_text((230, 115), "Role", fontsize=10)
    page.insert_text((80, 135), "Asha", fontsize=10)
    page.insert_text((230, 135), "Chair", fontsize=10)
    page.insert_text((72, 180), "Text below the table.", fontsize=11)
    document.save(path)
    document.close()

    blocks = parse_pdf(path).blocks
    assert [block.kind for block in blocks] == ["text", "table", "text"]
    assert "| Asha | Chair |" in blocks[1].text


def test_table_cells_keep_their_punctuation(tmp_path: Path) -> None:
    path = tmp_path / "people.pdf"
    document = pymupdf.open()
    page = document.new_page()
    for line_y in (100, 120, 140):
        page.draw_line((72, line_y), (472, line_y))
    for line_x in (72, 222, 472):
        page.draw_line((line_x, 100), (line_x, 140))
    page.insert_text((80, 115), "Name", fontsize=10)
    page.insert_text((230, 115), "Email", fontsize=10)
    page.insert_text((80, 135), "Dr. Asha Rao", fontsize=10)
    page.insert_text((230, 135), "asha.r@uni.edu.in", fontsize=10)
    document.save(path)
    document.close()

    table = next(block for block in parse_pdf(path).blocks if block.kind == "table")
    assert "| Dr. Asha Rao | asha.r@uni.edu.in |" in table.text


def test_calendar_page_keeps_events_and_drops_grid(tmp_path: Path) -> None:
    grid = [
        (60 + (day % 7) * 20, 100 + (day // 7) * 14, str(day + 1), 7, BODY) for day in range(31)
    ]
    items = [
        (72, 70, "ACADEMIC CALENDAR (2026-27) : Odd Semester", 10, BOLD),
        (60, 90, "Sun Mon Tue Wed Thu", 7, BOLD),
        *grid,
        (60, 220, "Academic Days - 21", 7, BOLD),
        (72, 300, "Mid Term Exam", 7, BODY),
        (350, 300, "12-16 Oct-26", 7, BODY),
        (450, 300, "Monday-Friday", 7, BODY),
        (72, 314, "Gandhi Jayanti", 7, BODY),
        (350, 314, "2-Oct-26", 7, BODY),
        (450, 314, "Friday", 7, BODY),
    ]
    blocks = parse_pdf(write_pdf(tmp_path / "c.pdf", [items])).blocks
    tables = [block for block in blocks if block.kind == "table"]
    assert len(tables) == 1
    assert "| Mid Term Exam | 12-16 Oct-26 | Monday-Friday |" in tables[0].text
    assert "| Gandhi Jayanti | 2-Oct-26 | Friday |" in tables[0].text
    assert blocks[0].kind == "heading"
    assert not any("Academic Days" in block.text or block.text == "31" for block in blocks)


def test_calendar_signature_is_dropped(tmp_path: Path) -> None:
    grid: list[PdfItem] = [
        (60 + (d % 7) * 20, 100 + (d // 7) * 14, str(d + 1), 7, BODY) for d in range(31)
    ]
    items = [
        *grid,
        (72, 400, "Prof. (Dr) Some Signatory", 8, BOLD),
        (72, 412, "Registrar", 8, BOLD),
    ]
    assert parse_pdf(write_pdf(tmp_path / "c.pdf", [items])).blocks == []


def test_docx_keeps_headings_paragraphs_and_tables(tmp_path: Path) -> None:
    document = new_docx()
    document.add_heading("Fees", level=2)
    document.add_paragraph("Fees are due in August.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Item", "Amount"
    table.cell(1, 0).text, table.cell(1, 1).text = "Hostel", "100"
    path = tmp_path / "a.docx"
    document.save(str(path))

    blocks = parse_docx(path).blocks
    assert [(block.kind, block.level) for block in blocks] == [
        ("heading", 2),
        ("text", 0),
        ("table", 0),
    ]
    assert blocks[0].page is None
    assert "| Hostel | 100 |" in blocks[2].text


def test_html_ignores_scripts_and_navigation(tmp_path: Path) -> None:
    path = tmp_path / "a.html"
    path.write_text(
        "<html><body><nav>Menu</nav><script>alert(1)</script><h2>Rules</h2>"
        "<p>Be kind.</p><ul><li>Item one</li></ul>"
        "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        "</body></html>"
    )
    blocks = parse_html(path).blocks
    assert [block.kind for block in blocks] == ["heading", "text", "text", "table"]
    assert blocks[2].text == "- Item one"
    assert not any("alert" in block.text or "Menu" in block.text for block in blocks)


def test_text_splits_paragraphs(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("First para\nstill first.\n\nSecond para.\n")
    assert [block.text for block in parse_text(path).blocks] == [
        "First para still first.",
        "Second para.",
    ]


def test_text_falls_back_for_non_utf8(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_bytes("Caf\xe9 policy".encode("latin-1"))
    assert parse_text(path).blocks[0].text == "Café policy"
