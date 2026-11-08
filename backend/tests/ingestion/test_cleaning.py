"""Text cleaning, masking and metadata extraction."""

from datetime import date
from pathlib import Path

import pytest

from app.ingestion.cleaning import (
    extract_printed_date,
    extract_ref_no,
    fingerprint,
    mask_contact_details,
    normalize_text,
    parse_pdf_date,
)
from app.ingestion.manifest import load_manifest


def test_normalize_removes_invisible_characters() -> None:
    assert normalize_text("Why NST​  is here​ ") == "Why NST is here"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Date: 30.04.2026", date(2026, 4, 30)),
        ("Dated: 21.08.2025", date(2025, 8, 21)),
        ("Date: 16/09/2026", date(2026, 9, 16)),
        ("Date: 01-09-2026", date(2026, 9, 1)),
        ("Date: 31.02.2026", None),
        ("No date in this text", None),
    ],
)
def test_printed_dates_are_day_first(text: str, expected: date | None) -> None:
    assert extract_printed_date(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ref No: RU/RO/AC/2025-26/564 Date", "RU/RO/AC/2025-26/564"),
        ("Ref. No.: RU/RO/SGRC/2025-26/370", "RU/RO/SGRC/2025-26/370"),
        ("Ref: RU/RO/OO/2026-27/679", "RU/RO/OO/2026-27/679"),
        ("Nothing here", None),
    ],
)
def test_reference_numbers(text: str, expected: str | None) -> None:
    assert extract_ref_no(text) == expected


def test_pdf_metadata_date() -> None:
    assert parse_pdf_date("D:20261024074624+00'00'") == date(2026, 10, 24)
    assert parse_pdf_date("") is None
    assert parse_pdf_date(None) is None


@pytest.mark.parametrize(
    "contact",
    ["person.name@nst.rishihood.edu.in", "9876543210", "+91 98765 43210", "+91-9876543210"],
)
def test_contact_details_are_masked(contact: str) -> None:
    masked, count = mask_contact_details(f"Contact {contact} today")
    assert contact not in masked
    assert count == 1


@pytest.mark.parametrize("text", ["RU/RO/OO/2026-27/679", "Date: 30.04.2026", "Fee of 1234567"])
def test_masking_leaves_other_numbers_alone(text: str) -> None:
    assert mask_contact_details(text) == (text, 0)


def test_fingerprint_ignores_case_and_whitespace() -> None:
    assert fingerprint("Hello   World\n") == fingerprint("hello world")
    assert fingerprint("Hello") != fingerprint("Goodbye")


def test_manifest_loads_and_normalizes(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text('"a.pdf":\n  title: A\n  doc_type: policy\n  date: 2026-01-02\n')
    meta = load_manifest(path)["a.pdf"]
    assert (meta.title, meta.doc_type, meta.doc_date) == ("A", "policy", date(2026, 1, 2))


def test_manifest_rejects_unknown_doc_type(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text('"a.pdf":\n  title: A\n  doc_type: memo\n')
    with pytest.raises(ValueError):
        load_manifest(path)


def test_missing_manifest_is_empty(tmp_path: Path) -> None:
    assert load_manifest(tmp_path / "nope.yaml") == {}


@pytest.mark.parametrize("damaged", ["someone@rishihoodeduin", "ab@nst", "a.b@x.edu.in."])
def test_damaged_emails_are_still_masked(damaged: str) -> None:
    masked, count = mask_contact_details(f"| Name | {damaged} |")
    assert "@" not in masked and count == 1


def test_student_representative_name_is_removed_from_table() -> None:
    table = (
        "| Name | Designation |\n| --- | --- |\n"
        "| Staff Member | Member |\n| Some Learner | Student Representative |"
    )
    masked, count = mask_contact_details(table)
    assert "Some Learner" not in masked and "[name removed]" in masked
    assert "Staff Member" in masked
    assert count == 1


def test_student_role_in_prose_is_left_alone() -> None:
    text = "The committee includes a student representative."
    assert mask_contact_details(text) == (text, 0)
