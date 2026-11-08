"""Admin document upload.

Protected by a shared key for now. Milestone 5 replaces it with signed-in admin users.
"""

import asyncio
import datetime as dt
import logging
import os
import secrets
import uuid
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Form,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.db.session import get_session, get_sessionmaker
from app.embeddings.base import Embedder, EmbeddingError
from app.embeddings.factory import build_embedder
from app.ingestion.manifest import DocType, DocumentMeta
from app.ingestion.pipeline import SUPPORTED_SUFFIXES, embed_pending, ingest_file

logger = logging.getLogger(__name__)

MAX_FILENAME_LENGTH = 200


def require_admin_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_admin_key: Annotated[str | None, Header()] = None,
) -> None:
    if not settings.admin_api_key:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Uploads are not configured")
    # compare_digest takes the same time wherever the strings differ, so response
    # timing cannot be used to guess the key one character at a time.
    if x_admin_key is None or not secrets.compare_digest(
        x_admin_key.encode(), settings.admin_api_key.encode()
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid admin key")


def get_optional_embedder(settings: Annotated[Settings, Depends(get_settings)]) -> Embedder | None:
    return build_embedder(settings)


router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin_key)])


class UploadResponse(BaseModel):
    source_file: str
    status: Literal["created", "updated", "unchanged"]
    chunks: int
    embedding: Literal["scheduled", "not_configured"]


def _safe_filename(raw_name: str | None) -> str:
    # Keep only the final part, so "../../etc/passwd" cannot escape the folder.
    name = PurePosixPath((raw_name or "").replace("\\", "/")).name
    if not name or name.startswith(".") or len(name) > MAX_FILENAME_LENGTH:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Invalid file name")
    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Unsupported file type")
    return name


def _check_content(name: str, content: bytes) -> None:
    """Check the bytes match the extension. File names are claims, not facts."""
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        valid = content.startswith(b"%PDF-")
    elif suffix == ".docx":
        valid = content.startswith(b"PK\x03\x04")
    else:
        valid = b"\x00" not in content
    if not valid or not content:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "File content does not match")


async def _embed_in_background(
    session_factory: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    rpm: int,
    tpm: int,
) -> None:
    try:
        async with session_factory() as session:
            await embed_pending(session, embedder, rpm=rpm, tpm=tpm)
    except EmbeddingError:
        logger.exception("Background embedding failed; the next ingest run will resume it")


@router.post("/documents", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile,
    doc_type: Annotated[DocType, Form()],
    background: BackgroundTasks,
    session: Annotated[AsyncSession, Depends(get_session)],
    session_factory: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[Embedder | None, Depends(get_optional_embedder)],
    title: Annotated[str | None, Form(max_length=300)] = None,
    doc_date: Annotated[dt.date | None, Form()] = None,
    url: Annotated[str | None, Form(max_length=2000)] = None,
) -> UploadResponse:
    name = _safe_filename(file.filename)
    limit = settings.max_upload_mb * 1024 * 1024
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "File is too large")
    _check_content(name, content)

    raw_dir = settings.raw_data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Written to a temporary name first: a failed upload never replaces a good file.
    temporary = raw_dir / f".upload-{uuid.uuid4().hex}{Path(name).suffix.lower()}"
    await asyncio.to_thread(temporary.write_bytes, content)
    try:
        meta = DocumentMeta(
            title=title or Path(name).stem, doc_type=doc_type, date=doc_date, url=url
        )
        try:
            result = await ingest_file(
                session,
                temporary,
                meta=meta,
                mask=settings.mask_contact_details,
                source_file=name,
            )
        except Exception as error:
            await session.rollback()
            logger.exception("Upload could not be parsed: %s", name)
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, "The file could not be read"
            ) from error

        if result.status == "duplicate":
            raise HTTPException(status.HTTP_409_CONFLICT, result.detail)
        if result.status == "failed":
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, result.detail)
        os.replace(temporary, raw_dir / name)
    finally:
        temporary.unlink(missing_ok=True)

    if embedder is not None:
        background.add_task(
            _embed_in_background,
            session_factory,
            embedder,
            settings.voyage_rpm,
            settings.voyage_tpm,
        )
    return UploadResponse(
        source_file=result.source_file,
        status=result.status,
        chunks=result.chunks,
        embedding="scheduled" if embedder is not None else "not_configured",
    )
