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
    detect_uniformity_break,
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


# --- uniformity mode -------------------------------------------------------

UNI_WINDOW_START = date(2026, 5, 25)


def _uni_series(uniform_days, window_days):
    """Build {date: {asin: rank}}. uniform_days: list of (day, rank) applied to
    all of x,y,z,w. window_days: list of (day, {asin: rank})."""
    per_date: dict = {}
    for day, rank in uniform_days:
        per_date[date(2026, 5, day)] = {a: rank for a in ("x", "y", "z", "w")}
    for day, ranks in window_days:
        per_date[date(2026, 5, day)] = dict(ranks)
    return per_date


def _detect_uni(per_date, **kwargs):
    return detect_uniformity_break(
        "AP",
        "Tools & Home Improvement",
        per_date,
        UNI_WINDOW_START,
        rank_type=kwargs.get("rank_type", 2),
        min_children=kwargs.get("min_children", 3),
        uniform_ratio=kwargs.get("uniform_ratio", 1.5),
        divergence_ratio=kwargs.get("divergence_ratio", 3.0),
        child_deviation_factor=kwargs.get("child_deviation_factor", 2.0),
        anomalous_fraction=kwargs.get("anomalous_fraction", 2 / 3),
        relative_divergence=kwargs.get("relative_divergence"),
    )


def test_uniformity_fanout_flags_anomalous():
    # All four children at 12 for days 20..25, then fan out on day 26.
    per_date = _uni_series(
        [(d, 12) for d in range(20, 26)],
        [(26, {"x": 31, "y": 25, "z": 102, "w": 155})],
    )
    event = _detect_uni(per_date, rank_type=1)
    assert event is not None
    assert event.severity == "anomalous"
    assert event.event_date == date(2026, 5, 26)
    assert event.n_children == 4
    assert event.n_diverged == 4  # all >= 2x of prior level 12
    assert event.prior_uniform_rank == 12
    assert event.rank_type == 1  # carries the rank type it was scanned in


def test_uniformity_single_outlier_is_suspect():
    # Only w jumps; x,y,z stay uniform -> minority -> suspect.
    per_date = _uni_series(
        [(d, 12) for d in range(20, 26)],
        [(26, {"x": 12, "y": 13, "z": 12, "w": 155})],
    )
    event = _detect_uni(per_date)
    assert event is not None
    assert event.severity == "suspect"
    assert event.n_diverged == 1


def test_uniformity_anomalous_fraction_boundary():
    # Two of four diverge = 0.5. Default 2/3 -> suspect; pass 0.5 -> anomalous.
    window = [(26, {"x": 12, "y": 13, "z": 80, "w": 100})]
    per_date = _uni_series([(d, 12) for d in range(20, 26)], window)
    assert _detect_uni(per_date).severity == "suspect"
    assert _detect_uni(per_date, anomalous_fraction=0.5).severity == "anomalous"


def test_uniformity_always_dispersed_parent_not_flagged():
    # Children are never uniform (spread > uniform_ratio in baseline) -> None.
    per_date = {
        date(2026, 5, d): {"x": 10, "y": 50, "z": 200, "w": 600} for d in range(20, 27)
    }
    assert _detect_uni(per_date) is None


def test_uniformity_improvement_only_not_flagged():
    # Spread grows but because children improved (ranks dropped) -> no diverged.
    per_date = _uni_series(
        [(d, 100) for d in range(20, 26)],
        [(26, {"x": 100, "y": 95, "z": 20, "w": 8})],
    )
    assert _detect_uni(per_date) is None


def test_uniformity_too_few_children_not_flagged():
    per_date = {date(2026, 5, d): {"x": 12, "y": 12} for d in range(20, 26)}
    per_date[date(2026, 5, 26)] = {"x": 12, "y": 200}
    assert _detect_uni(per_date, min_children=3) is None


def test_uniformity_multiday_picks_strongest_and_lists_others():
    # Diverge on day 26 (spread ~13x), recover, diverge again day 29 (spread ~5x).
    per_date = _uni_series(
        [(d, 12) for d in range(18, 26)],
        [
            (26, {"x": 31, "y": 25, "z": 102, "w": 155}),  # spread 155/25=6.2
            (27, {"x": 12, "y": 12, "z": 12, "w": 12}),  # back to uniform
            (28, {"x": 12, "y": 12, "z": 12, "w": 12}),
            (29, {"x": 40, "y": 30, "z": 38, "w": 36}),  # spread 40/30=1.33 < 3 (no)
            (30, {"x": 12, "y": 12, "z": 12, "w": 12}),
            (31, {"x": 200, "y": 250, "z": 30, "w": 40}),  # spread 250/30=8.3 strongest
        ],
    )
    event = _detect_uni(per_date)
    assert event is not None
    assert event.event_date == date(2026, 5, 31)  # strongest spread
    assert date(2026, 5, 26) in event.other_event_dates
    # Most recent occurrence across strongest + others.
    assert event.last_event_date == date(2026, 5, 31)


def test_uniformity_last_event_date_differs_from_strongest():
    # Strongest fan-out on day 26, a smaller later one on day 30.
    per_date = _uni_series(
        [(d, 12) for d in range(18, 26)],
        [
            (26, {"x": 31, "y": 25, "z": 102, "w": 400}),  # spread 16x (strongest)
            (27, {"x": 12, "y": 12, "z": 12, "w": 12}),
            (28, {"x": 12, "y": 12, "z": 12, "w": 12}),
            (29, {"x": 12, "y": 12, "z": 12, "w": 12}),
            (30, {"x": 12, "y": 12, "z": 12, "w": 60}),  # spread 5x (later, weaker)
        ],
    )
    event = _detect_uni(per_date)
    assert event is not None
    assert event.event_date == date(2026, 5, 26)  # strongest
    assert event.last_event_date == date(2026, 5, 30)  # most recent


def test_uniformity_relative_divergence_flags_tight_family():
    # Family glued at 12 for days 18..25, then a modest 1.6x spread on day 26.
    per_date = _uni_series(
        [(d, 12) for d in range(18, 26)],
        [(26, {"x": 12, "y": 12, "z": 12, "w": 19})],  # spread 19/12 = 1.58x
    )
    # Absolute bar (3.0) misses it.
    assert _detect_uni(per_date, child_deviation_factor=1.5) is None
    # Relative bar: 1.5 x the family's normal spread (~1.0) = 1.5 -> 1.58 trips it.
    event = _detect_uni(per_date, relative_divergence=1.5, child_deviation_factor=1.5)
    assert event is not None
    assert event.severity == "suspect"  # only w diverged (1 of 4)


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
