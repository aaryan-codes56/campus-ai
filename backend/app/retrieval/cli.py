"""Try retrieval from the terminal: uv run python -m app.retrieval.cli "your question" """

import argparse
import asyncio
import datetime as dt
from dataclasses import replace

from app.core.config import get_settings
from app.db.session import get_engine, get_sessionmaker
from app.retrieval.service import RetrievalError, build_retriever, config_from_settings
from app.retrieval.types import RetrievalConfig, RetrievalResult, SearchFilters


def _print_result(label: str, result: RetrievalResult) -> None:
    timings = ", ".join(f"{stage} {ms:.0f}ms" for stage, ms in result.timings_ms.items())
    degraded = f"  DEGRADED: {', '.join(result.degraded)}" if result.degraded else ""
    print(f"\n--- {label}  ({timings}){degraded}")
    for position, chunk in enumerate(result.chunks, start=1):
        item = chunk.candidate
        page = f"p{item.page_start}" if item.page_start else "-"
        section = item.heading_path.split(" > ")[-1] if item.heading_path else "(no section)"
        print(f"{position}. {chunk.score:.4f} {page:<4} {item.title[:30]:<30} | {section[:55]}")


def _configs(base: RetrievalConfig, args: argparse.Namespace) -> list[tuple[str, RetrievalConfig]]:
    if args.compare:
        return [
            ("vector", replace(base, mode="vector", rerank=False)),
            ("keyword", replace(base, mode="keyword", rerank=False)),
            ("hybrid", replace(base, mode="hybrid", rerank=False)),
            ("hybrid + re-rank", replace(base, mode="hybrid", rerank=True)),
        ]
    chosen = replace(base, mode=args.mode or base.mode, rerank=base.rerank and not args.no_rerank)
    return [(f"{chosen.mode}{' + re-rank' if chosen.rerank else ''}", chosen)]


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    # Patient mode: from a terminal it is fine to wait out rate limits.
    retriever = build_retriever(settings, interactive=False)
    base = replace(config_from_settings(settings), top_k=args.top_k)
    filters = SearchFilters(
        doc_types=tuple(args.doc_type or ()), date_from=args.date_from, date_to=args.date_to
    )
    try:
        async with get_sessionmaker()() as session:
            for label, config in _configs(base, args):
                result = await retriever.search(session, args.query, filters=filters, config=config)
                _print_result(label, result)
    except RetrievalError as error:
        print(f"Retrieval failed: {error}")
        return 1
    finally:
        await get_engine().dispose()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the ingested documents.")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["vector", "keyword", "hybrid"])
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--compare", action="store_true", help="run all four configurations")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--doc-type", action="append", help="repeat for several types")
    parser.add_argument("--date-from", type=dt.date.fromisoformat)
    parser.add_argument("--date-to", type=dt.date.fromisoformat)
    raise SystemExit(asyncio.run(_run(parser.parse_args())))


if __name__ == "__main__":
    main()
