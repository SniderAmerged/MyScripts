"""Unit tests for the pure corank detection logic (no DB access)."""

from __future__ import annotations

from datetime import date

import pytest

from rank_drop_checker import (
    _build_corank_groups,
    annotate_with_parent,
    detect_corank_breaks,
    parse_time_frame,
)

# --- corank detection ------------------------------------------------------

CORANK_WINDOW_START = date(2026, 5, 20)


def _detect_corank(per_date, **kwargs):
    return detect_corank_breaks(
        "Cut Resistant Gloves",
        per_date,
        CORANK_WINDOW_START,
        rank_type=kwargs.get("rank_type", 2),
        min_group_size=kwargs.get("min_group_size", 3),
        uniform_ratio=kwargs.get("uniform_ratio", 1.5),
        divergence_ratio=kwargs.get("divergence_ratio", 3.0),
        child_deviation_factor=kwargs.get("child_deviation_factor", 2.0),
        anomalous_fraction=kwargs.get("anomalous_fraction", 2 / 3),
    )


def test_build_corank_groups_clusters_shared_ranks():
    # Four ~13s cluster; the two ~900s are a separate, too-small group.
    ranks = {"a": 13, "b": 13, "c": 13, "d": 17, "e": 900, "f": 905}
    groups = _build_corank_groups(ranks, uniform_ratio=1.5, min_size=3)
    assert len(groups) == 1
    assert set(groups[0]) == {"a", "b", "c", "d"}  # 17/13 = 1.31 <= 1.5


def test_build_corank_groups_exact_rank_only():
    # uniform_ratio 1.0 -> only identical ranks cluster (parent-pure).
    ranks = {"a": 13, "b": 13, "c": 13, "d": 17}
    groups = _build_corank_groups(ranks, uniform_ratio=1.0, min_size=3)
    assert len(groups) == 1
    assert set(groups[0]) == {"a", "b", "c"}  # 17 excluded


def test_build_corank_groups_respects_min_size():
    ranks = {"a": 10, "b": 11}
    assert _build_corank_groups(ranks, uniform_ratio=1.5, min_size=3) == []


def test_corank_fanout_flags_anomalous():
    # Day 20: four ASINs share rank ~14. Day 21: all four fan out worse.
    per_date = {
        date(2026, 5, 20): {"x": 14, "y": 14, "z": 14, "w": 14},
        date(2026, 5, 21): {"x": 47, "y": 60, "z": 102, "w": 155},
    }
    events = _detect_corank(per_date)
    assert len(events) == 1
    e = events[0]
    assert e.severity == "anomalous"  # all 4 >= 2x of prior level 14
    assert e.group_date == date(2026, 5, 20)
    assert e.break_date == date(2026, 5, 21)
    assert e.prior_shared_rank == 14
    assert e.n_members == 4
    assert e.n_diverged == 4
    assert e.pseudo_group_id == "w@14"  # min member asin + shared level


def test_corank_fraction_controls_severity():
    per_date = {
        date(2026, 5, 20): {"x": 14, "y": 14, "z": 14, "w": 14},
        date(2026, 5, 21): {"x": 60, "y": 70, "z": 80, "w": 14},
    }
    # 3 of 4 diverge = 0.75 -> anomalous at default 2/3.
    assert _detect_corank(per_date)[0].severity == "anomalous"
    # One outlier among many -> suspect.
    per_date2 = {
        date(2026, 5, 20): {"x": 14, "y": 14, "z": 14, "w": 14},
        date(2026, 5, 21): {"x": 14, "y": 14, "z": 14, "w": 120},
    }
    assert _detect_corank(per_date2)[0].severity == "suspect"


def test_corank_improvement_only_not_flagged():
    # Spread grows but only because a member improved (rank dropped) -> skip.
    per_date = {
        date(2026, 5, 20): {"x": 100, "y": 100, "z": 100, "w": 100},
        date(2026, 5, 21): {"x": 100, "y": 100, "z": 100, "w": 8},
    }
    assert _detect_corank(per_date) == []


def test_corank_no_break_when_still_uniform():
    per_date = {
        date(2026, 5, 20): {"x": 14, "y": 14, "z": 15, "w": 16},
        date(2026, 5, 21): {"x": 15, "y": 15, "z": 16, "w": 17},
    }
    assert _detect_corank(per_date) == []


def test_corank_ignores_pairs_before_window():
    # The break is before window_start -> not reported.
    per_date = {
        date(2026, 5, 18): {"x": 14, "y": 14, "z": 14, "w": 14},
        date(2026, 5, 19): {"x": 14, "y": 14, "z": 80, "w": 120},
    }
    assert _detect_corank(per_date) == []


def test_corank_multiday_keeps_strongest_and_lists_others():
    # Same group breaks on day 21 (mild) and day 23 (stronger).
    per_date = {
        date(2026, 5, 20): {"x": 14, "y": 14, "z": 14, "w": 14},
        date(2026, 5, 21): {"x": 14, "y": 14, "z": 50, "w": 60},
        date(2026, 5, 22): {"x": 14, "y": 14, "z": 14, "w": 14},
        date(2026, 5, 23): {"x": 14, "y": 200, "z": 300, "w": 400},
    }
    events = _detect_corank(per_date)
    assert len(events) == 1
    e = events[0]
    assert e.break_date == date(2026, 5, 23)  # strongest spread
    assert date(2026, 5, 21) in e.other_break_dates


# --- optional parent-verify check ------------------------------------------


def test_annotate_with_parent_consistent_family():
    per_date = {
        date(2026, 5, 20): {"x": 14, "y": 14, "z": 14, "w": 14},
        date(2026, 5, 21): {"x": 47, "y": 60, "z": 102, "w": 155},
    }
    [event] = _detect_corank(per_date)
    parent_of = {"x": "P1", "y": "P1", "z": "P1", "w": "P1"}
    [annotated] = annotate_with_parent([event], parent_of)
    assert annotated.dominant_parent == "P1"
    assert annotated.parent_consistency == 1.0
    assert all(m.parent_asin == "P1" for m in annotated.members)


def test_annotate_with_parent_mixed_and_missing():
    per_date = {
        date(2026, 5, 20): {"x": 14, "y": 14, "z": 14, "w": 14},
        date(2026, 5, 21): {"x": 47, "y": 60, "z": 102, "w": 155},
    }
    [event] = _detect_corank(per_date)
    # 2 share P1, one a different parent, one untracked (absent from map).
    parent_of = {"x": "P1", "y": "P1", "z": "P2"}
    [annotated] = annotate_with_parent([event], parent_of)
    assert annotated.dominant_parent == "P1"
    assert annotated.parent_consistency == 0.5  # 2 of 4 members
    members = {m.asin: m.parent_asin for m in annotated.members}
    assert members["w"] == ""  # untracked -> empty


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
