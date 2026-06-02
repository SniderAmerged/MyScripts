"""Unit tests for the pure move-detection logic (no DB access)."""

from __future__ import annotations

import math
from datetime import date, datetime
from statistics import median

import pytest

from rank_drop_checker import (
    Observation,
    compute_cohort_moves,
    detect_anomalous_drop,
    largest_qualifying_move,
    parse_time_frame,
)


def _obs(day: int, rank: int) -> Observation:
    d = date(2026, 5, day)
    return Observation(report_date=d, rank=rank, captured_at=datetime(2026, 5, day, 21))


def _detect(ranks: list[tuple[int, int]], **kwargs):
    """ranks = list of (day, rank); returns the largest qualifying move."""
    obs = [_obs(day, rank) for day, rank in ranks]
    return largest_qualifying_move(
        "B000TEST",
        2,
        "Test Category",
        obs,
        min_pct=kwargs.get("min_pct", 100.0),
        min_positions=kwargs.get("min_positions", 100),
    )


def test_qualifies_on_positions_only():
    # +150 positions in one step (+15%): OR logic flags on positions alone.
    move = _detect([(21, 1000), (22, 1150)])
    assert move is not None
    assert move.abs_change == 150
    assert move.move_date == date(2026, 5, 22)
    assert move.category == "Test Category"


def test_qualifies_on_pct_only():
    # 10 -> 30 in one step: +200% but only +20 positions. OR flags on pct.
    move = _detect([(21, 10), (22, 30)])
    assert move is not None
    assert move.pct_change == pytest.approx(200.0)


def test_small_move_not_flagged():
    # +15 positions, +15%: neither threshold met.
    assert _detect([(21, 100), (22, 115)]) is None


def test_improved_rank_not_flagged():
    assert _detect([(21, 500), (22, 100)]) is None


def test_picks_largest_qualifying_step_and_its_date():
    # Two qualifying steps; the bigger jump (day 23) must win and set the date.
    move = _detect([(21, 100), (22, 250), (23, 900)])
    assert move is not None
    assert move.abs_change == 650
    assert move.move_date == date(2026, 5, 23)
    assert move.prev.rank == 250


def test_move_is_day_over_day_not_cumulative():
    # Gradual climb 100->130->160->190: cumulative is large but each step is
    # +30 (+<100%) -> no single step qualifies.
    assert _detect([(21, 100), (22, 130), (23, 160), (24, 190)]) is None


def test_single_observation_yields_no_move():
    assert _detect([(21, 100)]) is None


def test_empty_series_yields_no_move():
    assert _detect([]) is None


def test_unsorted_input_is_handled():
    # Same data as the positions test but out of order on input.
    move = _detect([(22, 1150), (21, 1000)])
    assert move is not None
    assert move.move_date == date(2026, 5, 22)


def test_move_datetime_carried_from_observation():
    move = _detect([(21, 10), (22, 30)])
    assert move is not None
    assert move.move_datetime == datetime(2026, 5, 22, 21)


WINDOW_START = date(2026, 5, 25)


def _series(ranks_by_day: list[tuple[int, int]]) -> list[Observation]:
    """ranks_by_day = list of (day-of-may, rank)."""
    return [_obs(day, rank) for day, rank in ranks_by_day]


def _detect_anomaly(observations, **kwargs):
    return detect_anomalous_drop(
        "B000TEST",
        1,
        "Test Category",
        observations,
        WINDOW_START,
        z_threshold=kwargs.get("z_threshold", 3.5),
        min_sustain=kwargs.get("min_sustain", 2),
        min_floor_pct=kwargs.get("min_floor_pct", 20.0),
        cohort_move=kwargs.get("cohort_move"),
    )


def test_anomaly_insufficient_history_returns_none():
    # Only a few days of history (< MIN_BASELINE_CHANGES) -> cannot judge.
    obs = _series([(20, 100), (21, 101), (22, 100), (26, 400)])
    assert _detect_anomaly(obs) is None


def test_anomaly_flags_break_from_stable_history():
    # 14 stable days (~100), then a sustained jump to ~500 inside the window.
    stable = [(d, 100 + (d % 3)) for d in range(1, 15)]  # days 1..14, ~100
    spike = [(25, 100), (26, 520), (27, 540), (28, 530)]  # window: big break
    finding = _detect_anomaly(_series(stable + spike))
    assert finding is not None
    assert finding.move.move_date == date(2026, 5, 26)
    assert finding.z_score >= 3.5
    assert finding.sustained_days >= 2


def test_anomaly_ignores_normal_volatility():
    # A series that habitually swings ~3x (cycling 100/300/150); an in-window
    # 1.5x move is smaller than its usual churn, so it is NOT anomalous.
    cycle = [100, 300, 150]
    noisy = [(d, cycle[(d - 1) % 3]) for d in range(1, 25)]
    noisy += [(25, 100), (26, 150), (27, 150)]  # ~1.5x — within normal
    assert _detect_anomaly(_series(noisy)) is None


def test_anomaly_requires_sustained_drop():
    # Stable history, then a one-day spike that immediately recovers.
    stable = [(d, 100 + (d % 3)) for d in range(1, 15)]
    blip = [(25, 100), (26, 600), (27, 101), (28, 100)]  # bounces right back
    assert _detect_anomaly(_series(stable + blip), min_sustain=2) is None


def test_cohort_denoise_removes_market_wide_move():
    # Stable history, then a big break inside the window.
    stable = [(d, 100 + (d % 3)) for d in range(1, 15)]
    spike = [(25, 100), (26, 520), (27, 540), (28, 530)]
    obs = _series(stable + spike)
    assert _detect_anomaly(obs) is not None  # flags without de-noising
    # If the whole cohort made the same move on 05-26, the excess is ~0.
    cohort_move = {date(2026, 5, 26): math.log(520 / 100)}
    assert _detect_anomaly(obs, cohort_move=cohort_move) is None


def test_cohort_denoise_keeps_product_specific_move():
    # Same break, but the cohort barely moved that day -> excess stays large.
    stable = [(d, 100 + (d % 3)) for d in range(1, 15)]
    spike = [(25, 100), (26, 520), (27, 540), (28, 530)]
    obs = _series(stable + spike)
    cohort_move = {date(2026, 5, 26): math.log(1.05)}  # peers ~flat
    finding = _detect_anomaly(obs, cohort_move=cohort_move)
    assert finding is not None
    assert finding.move.move_date == date(2026, 5, 26)


def test_compute_cohort_moves_medians_and_min_size():
    s1 = _series([(25, 100), (26, 500)])
    s2 = _series([(25, 100), (26, 510)])
    series = {("A1", 2, "h1"): s1, ("A2", 2, "h2"): s2}
    categories = {("A1", 2, "h1"): "Tools", ("A2", 2, "h2"): "Tools"}
    # min_cohort_size=2 -> 05-26 has 2 members, gets a median move.
    moves = compute_cohort_moves(series, categories, min_cohort_size=2)
    assert moves[(2, "Tools")][date(2026, 5, 26)] == pytest.approx(
        median([math.log(5.0), math.log(5.1)])
    )
    # min_cohort_size=3 -> too few members, nothing recorded.
    assert (
        compute_cohort_moves(series, categories, min_cohort_size=3)[(2, "Tools")] == {}
    )


def test_anomaly_improvement_not_flagged():
    stable = [(d, 500 + (d % 3)) for d in range(1, 15)]
    improve = [(25, 500), (26, 50), (27, 48)]  # rank got much better
    assert _detect_anomaly(_series(stable + improve)) is None


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
