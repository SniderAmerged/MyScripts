"""Unit tests for the pure comparison logic (no DB access)."""

from __future__ import annotations

from datetime import date

import pytest

from rank_drop_checker import (
    RankPoint,
    evaluate_rank_change,
    parse_time_frame,
)

BASE_DATE = date(2026, 5, 21)
CUR_DATE = date(2026, 5, 28)


def _eval(baseline_rank: int | None, current_rank: int | None, **kwargs):
    base = RankPoint(baseline_rank, BASE_DATE) if baseline_rank is not None else None
    curr = RankPoint(current_rank, CUR_DATE) if current_rank is not None else None
    return evaluate_rank_change(
        "B000TEST",
        2,
        "Test Category",
        base,
        curr,
        min_pct=kwargs.get("min_pct", 100.0),
        min_positions=kwargs.get("min_positions", 100),
    )


def test_worsened_past_both_thresholds_is_flagged():
    finding = _eval(100, 300)  # +200 positions, +200%
    assert finding is not None
    assert finding.abs_change == 200
    assert finding.pct_change == pytest.approx(200.0)
    assert finding.category == "Test Category"


def test_worsened_pct_only_not_flagged():
    # 10 -> 30: +200% but only +20 positions (< 100) -> AND fails.
    assert _eval(10, 30) is None


def test_worsened_positions_only_not_flagged():
    # 1000 -> 1150: +150 positions but only +15% (< 100%) -> AND fails.
    assert _eval(1000, 1150) is None


def test_improved_rank_not_flagged():
    # Rank number decreased = improved performance.
    assert _eval(500, 100) is None


def test_unchanged_not_flagged():
    assert _eval(500, 500) is None


def test_missing_baseline_skipped():
    assert _eval(None, 300) is None


def test_missing_current_skipped():
    assert _eval(100, None) is None


def test_exact_threshold_boundary_is_flagged():
    # 100 -> 200: exactly +100% and +100 positions -> inclusive, flagged.
    assert _eval(100, 200) is not None


def test_custom_thresholds_or_like_loosening():
    # Loosen so a small move qualifies.
    assert _eval(50, 70, min_pct=10.0, min_positions=10) is not None


@pytest.mark.parametrize(
    "value,expected",
    [("T-7", 7), ("t-1", 1), ("T-30", 30), ("14", 14), (" T-2 ", 2)],
)
def test_parse_time_frame_valid(value, expected):
    assert parse_time_frame(value) == expected


@pytest.mark.parametrize("value", ["T-0", "0", "-5", "abc", "T-x"])
def test_parse_time_frame_invalid(value):
    with pytest.raises(Exception):
        parse_time_frame(value)
