"""Pure ranking maths: rank fusion and recency. No I/O, so it is easy to test."""

import datetime as dt
import uuid
from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[uuid.UUID]], k: int = 60
) -> list[tuple[uuid.UUID, float]]:
    """Combine ranked lists: each item scores the sum of 1 / (k + rank) across lists.

    Only positions are used, because vector distances and text-match scores are on
    different scales and cannot be added directly. A larger k flattens the curve,
    so a single very high rank dominates less.
    """
    scores: dict[uuid.UUID, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    # sorted() is stable, so tied items keep the order they were first seen in.
    return sorted(scores.items(), key=lambda entry: entry[1], reverse=True)


def apply_recency(
    score: float,
    doc_date: dt.date | None,
    *,
    today: dt.date,
    boost: float,
    half_life_days: int,
) -> float:
    """Nudge a score upward for newer documents.

    The boost halves every half_life_days and is capped at `boost`, so it can settle
    a close call between conflicting documents but cannot beat a much better match.
    Undated documents get no boost.
    """
    if doc_date is None or boost == 0:
        return score
    age_days = max((today - doc_date).days, 0)
    decay: float = 0.5 ** (age_days / half_life_days)
    return score * (1.0 + boost * decay)
