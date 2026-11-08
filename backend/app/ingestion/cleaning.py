"""Text normalisation, contact-detail masking and metadata extraction."""

import hashlib
import re
import unicodedata
from datetime import date

ZERO_WIDTH = re.compile("[​‌‍⁠﻿­]")
SPACES = re.compile(r"[ \t ]+")
BULLET_START = re.compile(r"^[●•▪◦■○·]\s*")

# Deliberately broad: anything shaped like x@y is masked, even if extraction damaged it.
EMAIL = re.compile(r"[^\s|<>()\[\]]+@[^\s|<>()\[\]]+")
PHONE = re.compile(r"(?<![\d/-])(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?![\d/-])")

STUDENT_ROLE = re.compile(r"\bstudent\s+rep", re.IGNORECASE)

PRINTED_DATE = re.compile(r"\bDated?\s*:?\s*(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b", re.IGNORECASE)
REF_NO = re.compile(r"\bRef(?:\.|erence)?\s*(?:No\.?)?\s*:\s*([A-Za-z0-9/.-]*[A-Za-z0-9])")
PDF_DATE = re.compile(r"^D:(\d{4})(\d{2})(\d{2})")


def normalize_text(text: str) -> str:
    """Remove invisible characters and collapse runs of spaces."""
    text = ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text))
    return SPACES.sub(" ", text).strip()


def normalize_bullet(text: str) -> str:
    return BULLET_START.sub("- ", text)


def _mask_student_rows(text: str) -> tuple[str, int]:
    """Remove the name from table rows that list a student representative."""
    if not text.startswith("|"):
        return text, 0
    lines = text.split("\n")
    masked = 0
    for index, line in enumerate(lines):
        if STUDENT_ROLE.search(line):
            cells = line.split("|")
            cells[1] = " [name removed] "
            lines[index] = "|".join(cells)
            masked += 1
    return "\n".join(lines), masked


def mask_contact_details(text: str) -> tuple[str, int]:
    """Remove emails, phone numbers and student names, returning the text and the count."""
    text, names = _mask_student_rows(text)
    text, emails = EMAIL.subn("[email removed]", text)
    text, phones = PHONE.subn("[phone removed]", text)
    return text, names + emails + phones


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def extract_printed_date(text: str) -> date | None:
    """Find a date printed as 'Date: DD.MM.YYYY'. Indian documents are day-first."""
    match = PRINTED_DATE.search(text)
    if match is None:
        return None
    day, month, year = (int(group) for group in match.groups())
    return _safe_date(year, month, day)


def extract_ref_no(text: str) -> str | None:
    match = REF_NO.search(text)
    return match.group(1) if match else None


def parse_pdf_date(value: str | None) -> date | None:
    match = PDF_DATE.match(value or "")
    if match is None:
        return None
    year, month, day = (int(group) for group in match.groups())
    return _safe_date(year, month, day)


def fingerprint(text: str) -> str:
    """Hash text ignoring case and whitespace, so reformatted copies still match."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()
