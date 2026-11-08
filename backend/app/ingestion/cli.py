"""Command line ingestion: uv run python -m app.ingestion.cli --help"""

import argparse
import asyncio
import sys

from app.core.config import Settings, get_settings
from app.db.session import get_engine, get_sessionmaker
from app.embeddings.base import EmbeddingError
from app.embeddings.factory import build_embedder
from app.ingestion.manifest import load_manifest, normalize_name
from app.ingestion.pipeline import (
    FileResult,
    count_pending,
    embed_pending,
    ingest_directory,
    list_source_files,
    prepare_document,
)


def _print_row(result: FileResult) -> None:
    pages = "-" if result.pages is None else result.pages
    notes = [result.detail] if result.detail else []
    if result.masked:
        notes.append(f"masked {result.masked} contact details")
    if result.low_text_pages:
        notes.append(f"little text on pages {result.low_text_pages}")
    suffix = f"  ({'; '.join(notes)})" if notes else ""
    print(
        f"{result.status:<10}{pages:>4} pages{result.chunks:>5} chunks  "
        f"{result.source_file[:48]}{suffix}"
    )


def _dry_run(settings: Settings) -> int:
    manifest = load_manifest(settings.manifest_path)
    paths = list_source_files(settings.raw_data_dir)
    total = 0
    for path in paths:
        prepared = prepare_document(path, mask=settings.mask_contact_details)
        total += len(prepared.chunks)
        missing = "" if normalize_name(path.name) in manifest else "  (not in manifest)"
        print(f"{len(prepared.chunks):>5} chunks  {path.name[:60]}{missing}")
    print(f"\nTOTAL: {total} chunks from {len(paths)} files. Nothing was saved.")
    return 0


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if not settings.raw_data_dir.is_dir():
        print(f"Source folder not found: {settings.raw_data_dir}", file=sys.stderr)
        return 1
    if args.dry_run:
        return _dry_run(settings)

    manifest = load_manifest(settings.manifest_path)
    try:
        async with get_sessionmaker()() as session:
            results, pruned = await ingest_directory(
                session,
                settings.raw_data_dir,
                manifest,
                mask=settings.mask_contact_details,
                force=args.force,
                prune=args.prune,
            )
            for result in results:
                _print_row(result)
            for name in pruned:
                print(f"{'pruned':<10}{name}")
            failed = sum(result.status == "failed" for result in results)
            print(
                f"\n{len(results)} files, {sum(r.chunks for r in results)} chunks, {failed} failed"
            )

            if args.no_embed:
                return 1 if failed else 0
            embedder = build_embedder(settings)
            if embedder is None:
                print("Embedding skipped: VOYAGE_API_KEY is not set.")
                return 1 if failed else 0

            pending = await count_pending(session, embedder.model)
            print(f"Embedding {pending} chunks with {embedder.model}...")
            try:
                report = await embed_pending(
                    session,
                    embedder,
                    rpm=settings.voyage_rpm,
                    tpm=settings.voyage_tpm,
                    on_progress=lambda done, total: print(f"  {done}/{total}", flush=True),
                )
            except EmbeddingError as error:
                print(f"Embedding stopped: {error}\nRe-run to resume.", file=sys.stderr)
                return 1
            print(f"Embedded {report.embedded} chunks in {report.batches} batches.")
            return 1 if failed else 0
    finally:
        await get_engine().dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the documents in data/raw.")
    parser.add_argument("--force", action="store_true", help="re-process unchanged files")
    parser.add_argument("--prune", action="store_true", help="delete documents no longer on disk")
    parser.add_argument("--no-embed", action="store_true", help="store chunks without embedding")
    parser.add_argument("--dry-run", action="store_true", help="parse and chunk, save nothing")
    sys.exit(asyncio.run(_run(parser.parse_args())))


if __name__ == "__main__":
    main()
