"""Hand-maintained document metadata from data/manifest.yaml."""

import datetime as dt
import unicodedata
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

DocType = Literal["notice", "syllabus", "calendar", "policy", "paper"]


class DocumentMeta(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    title: str
    doc_type: DocType
    doc_date: dt.date | None = Field(default=None, alias="date")
    url: str | None = None


def normalize_name(name: str) -> str:
    # macOS and YAML can encode the same accented or curly character differently.
    return unicodedata.normalize("NFC", name)


def load_manifest(path: Path) -> dict[str, DocumentMeta]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {normalize_name(name): DocumentMeta.model_validate(item) for name, item in data.items()}
