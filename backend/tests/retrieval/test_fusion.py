"""Rank fusion and recency maths."""

import uuid
from datetime import date

import pytest

from app.retrieval.fusion import apply_recency, reciprocal_rank_fusion

A, B, C = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
TODAY = date(2026, 11, 9)


def test_scores_follow_the_formula() -> None:
    fused = dict(reciprocal_rank_fusion([[A, B]], k=60))
    assert fused[A] == pytest.approx(1 / 61)
    assert fused[B] == pytest.approx(1 / 62)


def test_item_in_both_lists_beats_top_of_one_list() -> None:
    fused = reciprocal_rank_fusion([[A, B], [C, B]], k=60)
    assert fused[0][0] == B


def test_each_item_appears_once() -> None:
    fused = reciprocal_rank_fusion([[A, B], [B, A]], k=60)
    assert sorted(item for item, _ in fused) == sorted([A, B])


def test_empty_list_is_ignored() -> None:
    assert reciprocal_rank_fusion([[A], []], k=60) == reciprocal_rank_fusion([[A]], k=60)
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_ties_keep_first_seen_order() -> None:
    assert [item for item, _ in reciprocal_rank_fusion([[A], [B]], k=60)] == [A, B]


def test_larger_k_narrows_the_gap_between_ranks() -> None:
    def gap(k: int) -> float:
        fused = dict(reciprocal_rank_fusion([[A, B]], k=k))
        return fused[A] / fused[B]

    assert gap(60) < gap(1)


def recency(doc_date: date | None, boost: float = 0.05, score: float = 1.0) -> float:
    return apply_recency(score, doc_date, today=TODAY, boost=boost, half_life_days=365)


def test_newer_document_scores_higher() -> None:
    assert recency(date(2026, 11, 1)) > recency(date(2024, 11, 1)) > 1.0


def test_boost_halves_after_one_half_life() -> None:
    assert recency(date(2025, 11, 9)) == pytest.approx(1.025)


def test_undated_document_gets_no_boost() -> None:
    assert recency(None) == 1.0


def test_future_date_is_capped_at_full_boost() -> None:
    assert recency(date(2030, 1, 1)) == pytest.approx(1.05)


def test_zero_boost_changes_nothing() -> None:
    assert recency(date(2026, 11, 1), boost=0) == 1.0


def test_recency_cannot_overturn_a_much_better_match() -> None:
    old_but_right = recency(date(2020, 1, 1), score=0.9)
    new_but_weak = recency(TODAY, score=0.6)
    assert old_but_right > new_but_weak


def test_recency_settles_a_close_call() -> None:
    assert recency(TODAY, score=0.80) > recency(date(2024, 1, 1), score=0.81)
