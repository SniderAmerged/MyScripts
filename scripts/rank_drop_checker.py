"""Detect ASINs whose Amazon best-seller rank dropped within a recent window.

Given an ``amerge_id``, this resolves the account's ASINs (Registry DB), then
scans each ASIN's daily rank history (ROAS DB ``asin_ranks``) over the last
``T-N`` days for a *significant day-over-day move*: a worsening (rank number
increased) of at least ``--min-pct`` percent OR at least ``--min-positions``
positions between two consecutive observations. For each ASIN/category it
reports the single largest qualifying move and the date/datetime it happened,
and exports all findings to a CSV file.

Usage:
    uv run python scripts/rank_drop_checker.py \\
        --amerge-id "US:US:1771995219729009" --time-frame T-7

Connection strings come from ``REGISTRY_DB_DSN`` and ``ROAS_DB_DSN`` (.env).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from statistics import median

import psycopg
from dotenv import load_dotenv

# Directory for generated CSV exports.
EXPORT_DIR = "export_output"

# asin_ranks.type -> human label (from live-data inspection).
RANK_TYPE_LABELS: dict[int, str] = {1: "subcategory", 2: "main"}

# --- Anomaly-mode defaults -------------------------------------------------
# Trailing days used to learn each series' "normal" daily volatility.
DEFAULT_BASELINE_DAYS = 45
# Robust-z cutoff: how many MADs above the product's normal a move must be.
DEFAULT_Z_THRESHOLD = 3.5
# The worsened rank must persist at least this many observations (anti-blip).
DEFAULT_MIN_SUSTAIN = 2
# Floor on the volatility scale, as a % move, so ultra-stable series don't flag
# on trivially small wiggles (sigma is floored at ln(1 + pct/100)).
DEFAULT_MIN_FLOOR_PCT = 20.0
# Minimum baseline day-over-day changes needed to trust the volatility estimate.
MIN_BASELINE_CHANGES = 10
# Minimum series in a cohort before its median move is used for de-noising.
DEFAULT_MIN_COHORT_SIZE = 3

# --- Uniformity-mode defaults ----------------------------------------------
# Min child ASINs a parent needs for an intra-parent variance test to be valid.
DEFAULT_MIN_CHILDREN = 3
# A group is "uniform" when max_rank <= this ratio * min_rank (tight cluster).
DEFAULT_UNIFORM_RATIO = 1.5
# A day is "dispersed" when max_rank >= this ratio * min_rank (fanned out).
DEFAULT_DIVERGENCE_RATIO = 3.0
# A child has "diverged" when its rank >= this factor * the prior uniform level.
DEFAULT_CHILD_DEVIATION_FACTOR = 2.0
# Fraction of children that must diverge to call the event anomalous (else suspect).
DEFAULT_ANOMALOUS_FRACTION = 2 / 3


@dataclass(frozen=True)
class Observation:
    """A single daily rank observation for one ASIN/category series."""

    report_date: date
    rank: int
    captured_at: datetime  # asin_ranks.date_add — when the row was recorded


@dataclass(frozen=True)
class Move:
    """A significant day-over-day worsening of an ASIN/category rank series."""

    asin: str
    rank_type: int
    category: str
    prev: Observation  # the earlier observation
    curr: Observation  # the observation the move landed on

    @property
    def abs_change(self) -> int:
        """Positions the rank moved (positive = worsened)."""
        return self.curr.rank - self.prev.rank

    @property
    def pct_change(self) -> float:
        """Percentage worsening relative to the previous rank."""
        return self.abs_change / self.prev.rank * 100.0

    @property
    def move_date(self) -> date:
        """The report_date on which the move happened."""
        return self.curr.report_date

    @property
    def move_datetime(self) -> datetime:
        """The timestamp the move's observation was captured."""
        return self.curr.captured_at


def parse_time_frame(value: str) -> int:
    """Parse ``T-7`` (or a bare ``7``) into a positive day count."""
    cleaned = value.strip().upper()
    if cleaned.startswith("T-"):
        cleaned = cleaned[2:]
    elif cleaned.startswith("T"):
        cleaned = cleaned[1:]
    try:
        days = int(cleaned)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid time frame {value!r}; expected e.g. 'T-7' or '7'"
        ) from exc
    if days <= 0:
        raise argparse.ArgumentTypeError("time frame must be a positive number of days")
    return days


def largest_qualifying_move(
    asin: str,
    rank_type: int,
    category: str,
    observations: list[Observation],
    *,
    min_pct: float,
    min_positions: int,
) -> Move | None:
    """Return the biggest qualifying day-over-day move in a series, else None.

    Pure function (no DB) so it can be unit-tested. Walks consecutive
    observations (sorted by date); a step qualifies when the rank *worsened*
    (number increased) by >= ``min_pct`` percent OR >= ``min_positions``
    positions. Returns the qualifying step with the largest absolute worsening.
    """
    ordered = sorted(observations, key=lambda o: o.report_date)
    best: Move | None = None
    for prev, curr in zip(ordered, ordered[1:]):
        if prev.rank <= 0:
            continue
        abs_change = curr.rank - prev.rank
        if abs_change <= 0:  # unchanged or improved
            continue
        pct_change = abs_change / prev.rank * 100.0
        if pct_change >= min_pct or abs_change >= min_positions:
            if best is None or abs_change > best.abs_change:
                best = Move(asin, rank_type, category, prev, curr)
    return best


@dataclass(frozen=True)
class AnomalyFinding:
    """A rank move that is anomalous versus a series' own normal volatility."""

    move: Move
    z_score: float  # robust z of the move's log-change vs baseline
    baseline_median_rank: float  # the series' typical rank before the move
    robust_sigma: float  # volatility scale used (floored MAD, in log space)
    sustained_days: int  # consecutive observations the rank stayed worse


def detect_anomalous_drop(
    asin: str,
    rank_type: int,
    category: str,
    observations: list[Observation],
    window_start: date,
    *,
    z_threshold: float,
    min_sustain: int,
    min_floor_pct: float,
    cohort_move: dict[date, float] | None = None,
) -> AnomalyFinding | None:
    """Return the most anomalous sustained worsening in the detection window.

    Pure function (no DB) so it can be unit-tested. Learns the series' normal
    daily *log-rank* change from observations before ``window_start`` (robust
    median + MAD), then within the window flags days whose worsening exceeds
    ``z_threshold`` robust deviations AND stays worse for ``min_sustain``
    observations. Returns the highest-z qualifying move, or None when there is
    too little history or nothing is anomalous.

    When ``cohort_move`` is given (date -> the cohort's median log-move that
    day), it is subtracted from every log-change so scoring is on the series'
    *excess* move over its peers — a product that merely rode a category-wide
    shift nets ~0 and drops out. The reported ranks and the sustained check use
    the raw ranks; only the z-score is cohort-adjusted.
    """
    ordered = sorted(observations, key=lambda o: o.report_date)
    if len(ordered) < 2:
        return None

    def adjustment(d: date) -> float:
        # Clamp at 0: de-noising only discounts a *shared worsening*; it never
        # amplifies a move just because peers improved (which would flag trivial
        # drops). So a product is only ever credited for moving with the herd.
        return max(0.0, cohort_move.get(d, 0.0)) if cohort_move else 0.0

    baseline_changes: list[float] = []
    baseline_ranks: list[int] = []
    for prev, curr in zip(ordered, ordered[1:]):
        if curr.report_date < window_start and prev.rank > 0 and curr.rank > 0:
            raw = math.log(curr.rank) - math.log(prev.rank)
            baseline_changes.append(raw - adjustment(curr.report_date))
    baseline_ranks = [o.rank for o in ordered if o.report_date < window_start]
    if len(baseline_changes) < MIN_BASELINE_CHANGES:
        return None

    center = median(baseline_changes)
    mad = median([abs(c - center) for c in baseline_changes])
    floor = math.log(1 + min_floor_pct / 100.0)
    sigma = max(1.4826 * mad, floor)
    base_median_rank = float(median(baseline_ranks))

    n = len(ordered)
    best: AnomalyFinding | None = None
    for i in range(1, n):
        prev, curr = ordered[i - 1], ordered[i]
        if curr.report_date < window_start or prev.rank <= 0 or curr.rank <= 0:
            continue
        raw_delta = math.log(curr.rank) - math.log(prev.rank)
        if raw_delta <= 0:  # actual rank unchanged or improved
            continue
        # Score the excess move over the cohort (no-op when not de-noising).
        delta = raw_delta - adjustment(curr.report_date)
        z = (delta - center) / sigma
        if z < z_threshold:
            continue
        # Sustained: count consecutive observations from the move that stay
        # "elevated" — at or above the log-midpoint (geometric mean) between
        # the pre- and post-move ranks. A bounce back near the old rank ends
        # the run, so one-day blips don't qualify.
        elevated = math.sqrt(prev.rank * curr.rank)
        run = 0
        for j in range(i, n):
            if ordered[j].rank >= elevated:
                run += 1
            else:
                break
        trailing_available = n - i  # observations from the move to series end
        # Accept if sustained long enough, or if it is too recent to confirm.
        if run >= min_sustain or trailing_available < min_sustain:
            if best is None or z > best.z_score:
                best = AnomalyFinding(
                    move=Move(asin, rank_type, category, prev, curr),
                    z_score=z,
                    baseline_median_rank=base_median_rank,
                    robust_sigma=sigma,
                    sustained_days=run,
                )
    return best


# Minimum baseline days with enough children needed to call a parent "normally
# uniform" before testing for a divergence event.
MIN_UNIFORM_BASELINE_DAYS = 3


@dataclass(frozen=True)
class ChildRank:
    """One child ASIN's standing on a parent's divergence-event day."""

    asin: str
    rank: int
    deviation_x: float  # rank / prior uniform level (>=1 worse, <1 better)
    diverged: bool


@dataclass(frozen=True)
class UniformityEvent:
    """A parent whose children's main ranks suddenly lost uniformity."""

    parent_asin: str
    rank_type: int  # 1 = subcategory, 2 = main
    category: str
    severity: str  # "anomalous" | "suspect"
    event_date: date
    prior_uniform_rank: float
    spread_ratio: float  # max_rank / min_rank on the event day
    children: list[ChildRank]
    other_event_dates: list[date]

    @property
    def n_children(self) -> int:
        return len(self.children)

    @property
    def n_diverged(self) -> int:
        return sum(c.diverged for c in self.children)

    @property
    def diverged_fraction(self) -> float:
        return self.n_diverged / self.n_children if self.children else 0.0

    @property
    def last_event_date(self) -> date:
        """Most recent date the divergence occurred (strongest or any other)."""
        return max([self.event_date, *self.other_event_dates])


@dataclass(frozen=True)
class CorankEvent:
    """A co-rank pseudo-group (ASINs sharing a rank) that fanned out next day."""

    pseudo_group_id: str  # synthetic label, e.g. "B07...@14"
    rank_type: int  # 1 = subcategory, 2 = main
    category: str
    severity: str  # "anomalous" | "suspect"
    group_date: date  # day the ASINs shared a rank (D)
    break_date: date  # day they fanned out (D+1)
    prior_shared_rank: float  # the shared rank level at group_date
    spread_ratio: float  # max/min of member ranks on the break day
    members: list[ChildRank]
    other_break_dates: list[date]

    @property
    def n_members(self) -> int:
        return len(self.members)

    @property
    def n_diverged(self) -> int:
        return sum(m.diverged for m in self.members)

    @property
    def diverged_fraction(self) -> float:
        return self.n_diverged / self.n_members if self.members else 0.0

    @property
    def last_break_date(self) -> date:
        return max([self.break_date, *self.other_break_dates])


def _spread_ratio(ranks: list[int]) -> float:
    """max/min of a group's ranks (1.0 == perfectly uniform)."""
    lo = min(ranks)
    return max(ranks) / lo if lo > 0 else float("inf")


def detect_uniformity_break(
    parent_asin: str,
    category: str,
    per_date_child_ranks: dict[date, dict[str, int]],
    window_start: date,
    *,
    rank_type: int = 2,
    min_children: int,
    uniform_ratio: float,
    divergence_ratio: float,
    child_deviation_factor: float,
    anomalous_fraction: float,
    relative_divergence: float | None = None,
) -> UniformityEvent | None:
    """Flag a parent whose children's main ranks suddenly fan out.

    Pure function (no DB). ``per_date_child_ranks`` maps a date to ``{child_asin:
    rank}`` for one (parent, category) group. The parent must be *normally
    uniform* before the window; a window day qualifies when the immediately
    prior day was uniform (``max/min <= uniform_ratio``) and that day is
    dispersed past the divergence bar. The bar is absolute (``>= divergence_ratio``)
    by default, or — when ``relative_divergence`` is given — relative to the
    family's own normal spread (``>= relative_divergence * baseline_median_spread``),
    so a normally ultra-tight family trips on a smaller absolute fan-out. Among
    the children, those whose rank is ``>= child_deviation_factor`` times the prior
    uniform level count as "diverged"; ``>= anomalous_fraction`` of them diverging
    makes the event *anomalous*, otherwise *suspect* (needs at least one). The
    strongest day (largest spread) is returned; the rest go into
    ``other_event_dates``.
    """
    dated = sorted(
        (d, ranks)
        for d, ranks in per_date_child_ranks.items()
        if len(ranks) >= min_children
    )
    if len(dated) < 2:
        return None

    # The parent must be normally uniform before the window.
    baseline_spreads = [
        _spread_ratio(list(ranks.values())) for d, ranks in dated if d < window_start
    ]
    if len(baseline_spreads) < MIN_UNIFORM_BASELINE_DAYS:
        return None
    baseline_median_spread = median(baseline_spreads)
    if baseline_median_spread > uniform_ratio:
        return None

    # The fan-out bar: absolute, or relative to this family's normal spread.
    divergence_bar = (
        relative_divergence * baseline_median_spread
        if relative_divergence is not None
        else divergence_ratio
    )

    candidates: list[UniformityEvent] = []
    for idx in range(1, len(dated)):
        day, ranks = dated[idx]
        if day < window_start:
            continue
        _, prev_ranks = dated[idx - 1]
        prev_spread = _spread_ratio(list(prev_ranks.values()))
        spread = _spread_ratio(list(ranks.values()))
        # Onset: uniform the prior day, dispersed today.
        if prev_spread > uniform_ratio or spread < divergence_bar:
            continue
        level = float(median(list(prev_ranks.values())))
        children = [
            ChildRank(
                asin=asin,
                rank=rank,
                deviation_x=rank / level if level > 0 else float("inf"),
                diverged=rank >= child_deviation_factor * level,
            )
            for asin, rank in sorted(ranks.items())
        ]
        n_diverged = sum(c.diverged for c in children)
        if n_diverged == 0:  # spread came only from improvement — not a drop
            continue
        fraction = n_diverged / len(children)
        severity = "anomalous" if fraction >= anomalous_fraction else "suspect"
        candidates.append(
            UniformityEvent(
                parent_asin=parent_asin,
                rank_type=rank_type,
                category=category,
                severity=severity,
                event_date=day,
                prior_uniform_rank=level,
                spread_ratio=spread,
                children=children,
                other_event_dates=[],
            )
        )

    if not candidates:
        return None
    strongest = max(candidates, key=lambda e: e.spread_ratio)
    others = sorted(
        e.event_date for e in candidates if e.event_date != strongest.event_date
    )
    return UniformityEvent(
        parent_asin=strongest.parent_asin,
        rank_type=strongest.rank_type,
        category=strongest.category,
        severity=strongest.severity,
        event_date=strongest.event_date,
        prior_uniform_rank=strongest.prior_uniform_rank,
        spread_ratio=strongest.spread_ratio,
        children=strongest.children,
        other_event_dates=others,
    )


def _build_corank_groups(
    ranks: dict[str, int], uniform_ratio: float, min_size: int
) -> list[list[str]]:
    """Cluster ASINs that share a rank into co-rank pseudo-groups.

    Pure helper. ASINs are sorted by rank and greedily clustered: an ASIN joins
    the current cluster while its rank stays within ``uniform_ratio`` of the
    cluster's smallest (best) rank, so every cluster keeps ``max/min <=
    uniform_ratio``. Returns clusters with at least ``min_size`` members. Because
    Amazon assigns the same BSR to a parent's child variations, a tight co-rank
    cluster within one category is effectively one variation family — no
    registry ``parent_asin`` needed.
    """
    ordered = sorted(ranks.items(), key=lambda kv: kv[1])
    groups: list[list[str]] = []
    current: list[str] = []
    cluster_min = 0
    for asin, rank in ordered:
        if current and cluster_min > 0 and rank / cluster_min <= uniform_ratio:
            current.append(asin)
        else:
            if len(current) >= min_size:
                groups.append(current)
            current = [asin]
            cluster_min = rank
    if len(current) >= min_size:
        groups.append(current)
    return groups


def detect_corank_breaks(
    category: str,
    per_date_ranks: dict[date, dict[str, int]],
    window_start: date,
    *,
    rank_type: int,
    min_group_size: int,
    uniform_ratio: float,
    divergence_ratio: float,
    child_deviation_factor: float,
    anomalous_fraction: float,
) -> list[CorankEvent]:
    """Flag co-rank pseudo-groups whose uniformity breaks the very next day.

    Pure function (no DB). ``per_date_ranks`` maps a date to ``{asin: rank}`` for
    one (category, rank_type). Baseline-free: for each consecutive day pair
    ``(D, D+1)`` whose later day falls in the window, build pseudo-groups from the
    ASINs that *share a rank* at ``D`` (``_build_corank_groups``), then look at the
    same ASINs at ``D+1``. A group breaks when its ``D+1`` ranks fan out to
    ``max/min >= divergence_ratio``. A member *diverged* if its ``D+1`` rank is
    ``>= child_deviation_factor`` times the shared level ``L`` (median rank at
    ``D``) — worsening only; if none diverged (spread came from improvement), the
    group is skipped. ``>= anomalous_fraction`` of members diverging makes it
    *anomalous*, else *suspect*. Each pseudo-group (identified by its member set)
    keeps its strongest break; other break dates go to ``other_break_dates``.
    """
    dated = sorted(per_date_ranks.items())
    by_group: dict[frozenset[str], list[CorankEvent]] = {}
    for idx in range(1, len(dated)):
        day, ranks_next = dated[idx]
        if day < window_start:
            continue
        prev_day, ranks_prev = dated[idx - 1]
        for group in _build_corank_groups(ranks_prev, uniform_ratio, min_group_size):
            present = {a: ranks_next[a] for a in group if a in ranks_next}
            if len(present) < 2:
                continue
            spread = _spread_ratio(list(present.values()))
            if spread < divergence_ratio:
                continue
            level = float(median([ranks_prev[a] for a in group]))
            members = [
                ChildRank(
                    asin=asin,
                    rank=rank,
                    deviation_x=rank / level if level > 0 else float("inf"),
                    diverged=rank >= child_deviation_factor * level,
                )
                for asin, rank in sorted(present.items())
            ]
            n_diverged = sum(m.diverged for m in members)
            if n_diverged == 0:  # fanned out only by improving — not a drop
                continue
            fraction = n_diverged / len(members)
            severity = "anomalous" if fraction >= anomalous_fraction else "suspect"
            group_id = f"{min(group)}@{round(level)}"
            by_group.setdefault(frozenset(group), []).append(
                CorankEvent(
                    pseudo_group_id=group_id,
                    rank_type=rank_type,
                    category=category,
                    severity=severity,
                    group_date=prev_day,
                    break_date=day,
                    prior_shared_rank=level,
                    spread_ratio=spread,
                    members=members,
                    other_break_dates=[],
                )
            )

    events: list[CorankEvent] = []
    for candidates in by_group.values():
        strongest = max(candidates, key=lambda e: e.spread_ratio)
        others = sorted(
            e.break_date for e in candidates if e.break_date != strongest.break_date
        )
        events.append(
            CorankEvent(
                pseudo_group_id=strongest.pseudo_group_id,
                rank_type=strongest.rank_type,
                category=strongest.category,
                severity=strongest.severity,
                group_date=strongest.group_date,
                break_date=strongest.break_date,
                prior_shared_rank=strongest.prior_shared_rank,
                spread_ratio=strongest.spread_ratio,
                members=strongest.members,
                other_break_dates=others,
            )
        )
    return events


def fetch_account_asins(
    conn: psycopg.Connection, amerge_id: str, marketplace: str | None
) -> list[str]:
    """Distinct non-empty ASINs (product_code) registered for an amerge_id."""
    sql = (
        "SELECT DISTINCT product_code FROM public.registry_api_asinregistry "
        "WHERE amerge_id = %(amerge_id)s AND product_code <> ''"
    )
    params: dict[str, object] = {"amerge_id": amerge_id}
    if marketplace:
        sql += " AND marketplace = %(marketplace)s"
        params["marketplace"] = marketplace
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [row[0] for row in cur.fetchall()]


def fetch_parent_map(
    conn: psycopg.Connection, amerge_id: str, marketplace: str | None
) -> dict[str, str]:
    """Map each child ASIN (product_code) -> its parent_asin for an account.

    Only products that have a non-empty parent are returned. If a product_code
    appears under more than one parent, the first seen wins (rare; logged-free).
    """
    sql = (
        "SELECT DISTINCT product_code, parent_asin "
        "FROM public.registry_api_asinregistry "
        "WHERE amerge_id = %(amerge_id)s AND product_code <> '' AND parent_asin <> ''"
    )
    params: dict[str, object] = {"amerge_id": amerge_id}
    if marketplace:
        sql += " AND marketplace = %(marketplace)s"
        params["marketplace"] = marketplace
    parent_of: dict[str, str] = {}
    with conn.cursor() as cur:
        cur.execute(sql, params)
        for product_code, parent_asin in cur.fetchall():
            parent_of.setdefault(product_code, parent_asin)
    return parent_of


def fetch_latest_report_date(conn: psycopg.Connection) -> date | None:
    """The most recent report_date present in asin_ranks."""
    with conn.cursor() as cur:
        cur.execute("SELECT max(report_date) FROM public.asin_ranks")
        row = cur.fetchone()
    return row[0] if row else None


def _type_filter(rank_type: str) -> tuple[str, dict[str, object]]:
    """SQL fragment + params restricting to a rank type (empty when 'both')."""
    if rank_type == "both":
        return "", {}
    return " AND type = %(rank_type)s", {"rank_type": int(rank_type)}


# A rank series is identified by (asin, type, category_hash). A product can be
# ranked in several categories of the same `type`, each its own series; the
# category hash is the 3rd ':'-segment of `asin_ranks.id` and is stable over time.
SeriesKey = tuple[str, int, str]


def fetch_series(
    conn: psycopg.Connection,
    asins: list[str],
    start_date: date,
    end_date: date,
    rank_type: str,
) -> tuple[dict[SeriesKey, list[Observation]], dict[SeriesKey, str]]:
    """Daily observations in [start_date, end_date] per (asin, type, cat_hash).

    Returns (series, categories): ``series`` maps each key to its list of
    observations; ``categories`` maps each key to its human category title.
    """
    frag, params = _type_filter(rank_type)
    sql = (
        "SELECT asin, type, split_part(id, ':', 3) AS cat_hash, title, "
        "rank, report_date, date_add "
        "FROM public.asin_ranks "
        "WHERE asin = ANY(%(asins)s) "
        "AND report_date >= %(start)s AND report_date <= %(end)s" + frag + " "
        "ORDER BY asin, type, split_part(id, ':', 3), report_date"
    )
    series: dict[SeriesKey, list[Observation]] = {}
    categories: dict[SeriesKey, str] = {}
    with conn.cursor() as cur:
        cur.execute(
            sql,
            {"asins": asins, "start": start_date, "end": end_date, **params},
        )
        for asin, rtype, cat_hash, title, rank, report_date, captured_at in cur:
            key = (asin, rtype, cat_hash)
            series.setdefault(key, []).append(
                Observation(report_date=report_date, rank=rank, captured_at=captured_at)
            )
            categories[key] = title
    return series, categories


def find_rank_drops(
    registry_conn: psycopg.Connection,
    roas_conn: psycopg.Connection,
    *,
    amerge_id: str,
    days: int,
    min_pct: float,
    min_positions: int,
    rank_type: str,
    marketplace: str | None,
) -> tuple[list[Move], dict[str, object]]:
    """Run the full pipeline; return the largest move per series + a summary."""
    asins = fetch_account_asins(registry_conn, amerge_id, marketplace)
    summary: dict[str, object] = {
        "amerge_id": amerge_id,
        "asin_count": len(asins),
        "series_scanned": 0,
    }
    if not asins:
        return [], summary

    cur_date = fetch_latest_report_date(roas_conn)
    if cur_date is None:
        return [], summary
    start_date = cur_date - timedelta(days=days)
    summary["window_start"] = start_date
    summary["window_end"] = cur_date

    series, categories = fetch_series(roas_conn, asins, start_date, cur_date, rank_type)
    summary["series_scanned"] = len(series)

    moves: list[Move] = []
    for key, observations in series.items():
        asin, rtype, _cat_hash = key
        move = largest_qualifying_move(
            asin,
            rtype,
            categories[key],
            observations,
            min_pct=min_pct,
            min_positions=min_positions,
        )
        if move is not None:
            moves.append(move)
    moves.sort(key=lambda m: m.abs_change, reverse=True)
    return moves, summary


# A cohort groups series that tend to move together when Amazon recomputes a
# category: same rank type + same category title.
CohortKey = tuple[int, str]


def compute_cohort_moves(
    series: dict[SeriesKey, list[Observation]],
    categories: dict[SeriesKey, str],
    min_cohort_size: int,
) -> dict[CohortKey, dict[date, float]]:
    """Per (rank_type, category) cohort, the median daily log-move by date.

    Only dates with at least ``min_cohort_size`` member series contribute — a
    median over too few products is not a reliable market-wide signal, and
    de-noising tiny cohorts would erase the very moves we want to keep.
    """
    raw: dict[CohortKey, dict[date, list[float]]] = {}
    for key, observations in series.items():
        cohort = (key[1], categories[key])
        ordered = sorted(observations, key=lambda o: o.report_date)
        by_date = raw.setdefault(cohort, {})
        for prev, curr in zip(ordered, ordered[1:]):
            if prev.rank > 0 and curr.rank > 0:
                change = math.log(curr.rank) - math.log(prev.rank)
                by_date.setdefault(curr.report_date, []).append(change)
    return {
        cohort: {
            d: median(vals)
            for d, vals in by_date.items()
            if len(vals) >= min_cohort_size
        }
        for cohort, by_date in raw.items()
    }


def find_rank_anomalies(
    registry_conn: psycopg.Connection,
    roas_conn: psycopg.Connection,
    *,
    amerge_id: str,
    days: int,
    baseline_days: int,
    z_threshold: float,
    min_sustain: int,
    min_floor_pct: float,
    rank_type: str,
    marketplace: str | None,
    cohort_denoise: bool = False,
    min_cohort_size: int = 3,
) -> tuple[list[AnomalyFinding], dict[str, object]]:
    """Anomaly-mode pipeline: flag drops unusual vs each series' own history."""
    asins = fetch_account_asins(registry_conn, amerge_id, marketplace)
    summary: dict[str, object] = {
        "amerge_id": amerge_id,
        "asin_count": len(asins),
        "series_scanned": 0,
    }
    if not asins:
        return [], summary

    cur_date = fetch_latest_report_date(roas_conn)
    if cur_date is None:
        return [], summary
    window_start = cur_date - timedelta(days=days)
    fetch_start = window_start - timedelta(days=baseline_days)
    summary["window_start"] = window_start
    summary["window_end"] = cur_date
    summary["baseline_start"] = fetch_start

    series, categories = fetch_series(
        roas_conn, asins, fetch_start, cur_date, rank_type
    )
    summary["series_scanned"] = len(series)
    summary["cohort_denoise"] = cohort_denoise

    cohort_moves = (
        compute_cohort_moves(series, categories, min_cohort_size)
        if cohort_denoise
        else {}
    )

    findings: list[AnomalyFinding] = []
    for key, observations in series.items():
        asin, rtype, _cat_hash = key
        cohort_move = (
            cohort_moves.get((rtype, categories[key])) if cohort_denoise else None
        )
        finding = detect_anomalous_drop(
            asin,
            rtype,
            categories[key],
            observations,
            window_start,
            z_threshold=z_threshold,
            min_sustain=min_sustain,
            min_floor_pct=min_floor_pct,
            cohort_move=cohort_move,
        )
        if finding is not None:
            findings.append(finding)
    findings.sort(key=lambda f: f.z_score, reverse=True)
    return findings, summary


# Sort key for severity: anomalous before suspect.
_SEVERITY_ORDER = {"anomalous": 0, "suspect": 1}


def find_uniformity_breaks(
    registry_conn: psycopg.Connection,
    roas_conn: psycopg.Connection,
    *,
    amerge_id: str,
    days: int,
    baseline_days: int,
    min_children: int,
    uniform_ratio: float,
    divergence_ratio: float,
    child_deviation_factor: float,
    anomalous_fraction: float,
    marketplace: str | None,
    rank_type: str = "2",
    relative_divergence: float | None = None,
) -> tuple[list[UniformityEvent], dict[str, object]]:
    """Uniformity-mode pipeline: flag parents whose children's ranks fan out.

    ``rank_type`` selects which category ranks to compare within each parent:
    "2" main only, "1" subcategory only, or "both". Each (parent, type,
    category) is its own independent sibling group.
    """
    parent_of = fetch_parent_map(registry_conn, amerge_id, marketplace)
    summary: dict[str, object] = {
        "amerge_id": amerge_id,
        "child_asin_count": len(parent_of),
        "parents_scanned": 0,
    }
    if not parent_of:
        return [], summary

    cur_date = fetch_latest_report_date(roas_conn)
    if cur_date is None:
        return [], summary
    window_start = cur_date - timedelta(days=days)
    fetch_start = window_start - timedelta(days=baseline_days)
    summary["window_start"] = window_start
    summary["window_end"] = cur_date
    summary["baseline_start"] = fetch_start

    series, categories = fetch_series(
        roas_conn, list(parent_of), fetch_start, cur_date, rank_type=rank_type
    )

    # Group into (parent, rank_type, category) -> {date -> {child_asin -> rank}}.
    # Type is in the key so a main-category title can't merge with a subcategory.
    groups: dict[tuple[str, int, str], dict[date, dict[str, int]]] = {}
    for key, observations in series.items():
        asin, rtype, _cat_hash = key
        parent = parent_of.get(asin)
        if parent is None:
            continue
        group_key = (parent, rtype, categories[key])
        by_date = groups.setdefault(group_key, {})
        for obs in observations:
            by_date.setdefault(obs.report_date, {})[asin] = obs.rank
    summary["parents_scanned"] = len({p for p, _, _ in groups})

    events: list[UniformityEvent] = []
    for (parent, rtype, category), per_date in groups.items():
        event = detect_uniformity_break(
            parent,
            category,
            per_date,
            window_start,
            rank_type=rtype,
            min_children=min_children,
            uniform_ratio=uniform_ratio,
            divergence_ratio=divergence_ratio,
            child_deviation_factor=child_deviation_factor,
            anomalous_fraction=anomalous_fraction,
            relative_divergence=relative_divergence,
        )
        if event is not None:
            events.append(event)
    events.sort(key=lambda e: (_SEVERITY_ORDER.get(e.severity, 9), -e.spread_ratio))
    return events, summary


# Calendar lead-in so the first in-window day still has a prior observation to
# pair with (covers weekend/missing-day gaps in the daily series).
CORANK_LEAD_IN_DAYS = 3


def find_corank_breaks(
    registry_conn: psycopg.Connection,
    roas_conn: psycopg.Connection,
    *,
    amerge_id: str,
    days: int,
    min_group_size: int,
    uniform_ratio: float,
    divergence_ratio: float,
    child_deviation_factor: float,
    anomalous_fraction: float,
    marketplace: str | None,
    rank_type: str = "2",
) -> tuple[list[CorankEvent], dict[str, object]]:
    """Co-rank pipeline: flag pseudo-groups (shared-rank ASINs) that fan out.

    Unlike ``uniformity`` this needs no registry ``parent_asin`` and no baseline:
    families are reconstructed each day from ASINs that share a rank within a
    category. ``rank_type`` "2" main, "1" subcategory, or "both" (each type is its
    own grouping space).
    """
    asins = fetch_account_asins(registry_conn, amerge_id, marketplace)
    summary: dict[str, object] = {
        "amerge_id": amerge_id,
        "asin_count": len(asins),
        "categories_scanned": 0,
    }
    if not asins:
        return [], summary

    cur_date = fetch_latest_report_date(roas_conn)
    if cur_date is None:
        return [], summary
    window_start = cur_date - timedelta(days=days)
    fetch_start = window_start - timedelta(days=CORANK_LEAD_IN_DAYS)
    summary["window_start"] = window_start
    summary["window_end"] = cur_date

    series, categories = fetch_series(
        roas_conn, asins, fetch_start, cur_date, rank_type=rank_type
    )

    # Group into (rank_type, cat_hash) -> {date -> {asin -> rank}}. Grouping is by
    # category, NOT parent; co-rank clustering separates families within it.
    groups: dict[tuple[int, str], dict[date, dict[str, int]]] = {}
    cat_title: dict[tuple[int, str], str] = {}
    for key, observations in series.items():
        asin, rtype, cat_hash = key
        group_key = (rtype, cat_hash)
        by_date = groups.setdefault(group_key, {})
        cat_title[group_key] = categories[key]
        for obs in observations:
            by_date.setdefault(obs.report_date, {})[asin] = obs.rank
    summary["categories_scanned"] = len(groups)

    events: list[CorankEvent] = []
    for (rtype, _cat_hash), per_date in groups.items():
        events.extend(
            detect_corank_breaks(
                cat_title[(rtype, _cat_hash)],
                per_date,
                window_start,
                rank_type=rtype,
                min_group_size=min_group_size,
                uniform_ratio=uniform_ratio,
                divergence_ratio=divergence_ratio,
                child_deviation_factor=child_deviation_factor,
                anomalous_fraction=anomalous_fraction,
            )
        )
    events.sort(key=lambda e: (_SEVERITY_ORDER.get(e.severity, 9), -e.spread_ratio))
    return events, summary


def _type_label(rank_type: int) -> str:
    return RANK_TYPE_LABELS.get(rank_type, str(rank_type))


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def render_table(moves: list[Move]) -> str:
    """Aligned text table of the largest move per ASIN/category."""
    header = (
        f"{'ASIN':<12} {'TYPE':<12} {'CATEGORY':<26} "
        f"{'FROM':>9} {'TO':>9} {'Δ POS':>8} {'Δ %':>9}  "
        f"{'MOVE DATE':<11} {'MOVE TIME (UTC)':<20}"
    )
    lines = [header, "-" * len(header)]
    for m in moves:
        lines.append(
            f"{m.asin:<12} {_type_label(m.rank_type):<12} "
            f"{_truncate(m.category, 26):<26} "
            f"{m.prev.rank:>9} {m.curr.rank:>9} "
            f"{m.abs_change:>8} {m.pct_change:>8.1f}%  "
            f"{m.move_date.isoformat():<11} "
            f"{m.move_datetime.strftime('%Y-%m-%d %H:%M:%S'):<20}"
        )
    return "\n".join(lines)


CSV_HEADER = [
    "asin",
    "rank_type",
    "rank_type_label",
    "category",
    "prev_date",
    "prev_rank",
    "move_date",
    "move_datetime",
    "current_rank",
    "abs_change",
    "pct_change",
]


def _ensure_parent_dir(path: str) -> None:
    """Create the directory holding ``path`` if it does not yet exist."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_csv(moves: list[Move], path: str) -> None:
    """Write findings to ``path`` as CSV (one row per ASIN/category move)."""
    _ensure_parent_dir(path)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for m in moves:
            writer.writerow(
                [
                    m.asin,
                    m.rank_type,
                    _type_label(m.rank_type),
                    m.category,
                    m.prev.report_date.isoformat(),
                    m.prev.rank,
                    m.move_date.isoformat(),
                    m.move_datetime.isoformat(),
                    m.curr.rank,
                    m.abs_change,
                    f"{m.pct_change:.1f}",
                ]
            )


def render_anomaly_table(findings: list[AnomalyFinding]) -> str:
    """Aligned text table of anomalous drops, sorted by z-score."""
    header = (
        f"{'ASIN':<12} {'TYPE':<12} {'CATEGORY':<24} "
        f"{'Z':>6} {'NORMAL':>9} {'FROM':>9} {'TO':>9} {'Δ %':>9} "
        f"{'SUST':>4}  {'MOVE DATE':<11}"
    )
    lines = [header, "-" * len(header)]
    for f in findings:
        m = f.move
        lines.append(
            f"{m.asin:<12} {_type_label(m.rank_type):<12} "
            f"{_truncate(m.category, 24):<24} "
            f"{f.z_score:>6.1f} {round(f.baseline_median_rank):>9} "
            f"{m.prev.rank:>9} {m.curr.rank:>9} {m.pct_change:>8.1f}% "
            f"{f.sustained_days:>4}  {m.move_date.isoformat():<11}"
        )
    return "\n".join(lines)


ANOMALY_CSV_HEADER = [
    "asin",
    "rank_type_label",
    "category",
    "prev_rank",
    "move_date",
    "current_rank",
    "positions_moved",
    "pct_change",
]


def write_anomaly_csv(findings: list[AnomalyFinding], path: str) -> None:
    """Write anomaly findings to ``path`` as CSV (one row per ASIN/category)."""
    _ensure_parent_dir(path)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(ANOMALY_CSV_HEADER)
        for f in findings:
            m = f.move
            writer.writerow(
                [
                    m.asin,
                    _type_label(m.rank_type),
                    m.category,
                    m.prev.rank,
                    m.move_date.isoformat(),
                    m.curr.rank,
                    m.abs_change,
                    f"{m.pct_change:.1f}",
                ]
            )


def render_uniformity_table(events: list[UniformityEvent]) -> str:
    """Aligned text table: one line per flagged parent (strongest event)."""
    header = (
        f"{'PARENT':<12} {'TYPE':<12} {'SEVERITY':<10} {'CATEGORY':<22} "
        f"{'EVENT DATE':<11} {'CHILD':>5} {'DIV':>4} {'FRAC':>5} "
        f"{'PRIOR':>8} {'SPREAD×':>8}  {'OTHER DATES':<24}"
    )
    lines = [header, "-" * len(header)]
    for e in events:
        others = ",".join(d.isoformat() for d in e.other_event_dates) or "-"
        lines.append(
            f"{e.parent_asin:<12} {_type_label(e.rank_type):<12} {e.severity:<10} "
            f"{_truncate(e.category, 22):<22} "
            f"{e.event_date.isoformat():<11} {e.n_children:>5} {e.n_diverged:>4} "
            f"{e.diverged_fraction:>5.0%} {round(e.prior_uniform_rank):>8} "
            f"{e.spread_ratio:>7.1f}x  {_truncate(others, 24):<24}"
        )
    return "\n".join(lines)


UNIFORMITY_CSV_HEADER = [
    "parent_asin",
    "rank_type",
    "rank_type_label",
    "asin",
    "category",
    "event_date",
    "last_event_date",
    "severity",
    "n_children",
    "n_diverged",
    "diverged_fraction",
    "prior_uniform_rank",
    "child_rank",
    "child_deviation_x",
    "child_diverged",
    "other_event_dates",
]


def write_uniformity_csv(events: list[UniformityEvent], path: str) -> None:
    """Write uniformity findings to ``path`` — one row per child of each parent."""
    _ensure_parent_dir(path)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(UNIFORMITY_CSV_HEADER)
        for e in events:
            others = ",".join(d.isoformat() for d in e.other_event_dates)
            for c in e.children:
                writer.writerow(
                    [
                        e.parent_asin,
                        e.rank_type,
                        _type_label(e.rank_type),
                        c.asin,
                        e.category,
                        e.event_date.isoformat(),
                        e.last_event_date.isoformat(),
                        e.severity,
                        e.n_children,
                        e.n_diverged,
                        f"{e.diverged_fraction:.3f}",
                        round(e.prior_uniform_rank),
                        c.rank,
                        f"{c.deviation_x:.2f}",
                        int(c.diverged),
                        others,
                    ]
                )


def render_corank_table(events: list[CorankEvent]) -> str:
    """Aligned text table: one line per flagged co-rank pseudo-group."""
    header = (
        f"{'PSEUDO-GROUP':<16} {'TYPE':<12} {'SEVERITY':<10} {'CATEGORY':<22} "
        f"{'GROUP→BREAK':<23} {'MEMB':>5} {'DIV':>4} {'FRAC':>5} "
        f"{'SHARED':>8} {'SPREAD×':>8}  {'OTHER BREAKS':<24}"
    )
    lines = [header, "-" * len(header)]
    for e in events:
        others = ",".join(d.isoformat() for d in e.other_break_dates) or "-"
        window = f"{e.group_date.isoformat()}→{e.break_date.isoformat()}"
        lines.append(
            f"{e.pseudo_group_id:<16} {_type_label(e.rank_type):<12} {e.severity:<10} "
            f"{_truncate(e.category, 22):<22} {window:<23} "
            f"{e.n_members:>5} {e.n_diverged:>4} {e.diverged_fraction:>5.0%} "
            f"{round(e.prior_shared_rank):>8} {e.spread_ratio:>7.1f}x  "
            f"{_truncate(others, 24):<24}"
        )
    return "\n".join(lines)


CORANK_CSV_HEADER = [
    "pseudo_group_id",
    "rank_type",
    "rank_type_label",
    "asin",
    "category",
    "group_date",
    "break_date",
    "severity",
    "n_members",
    "n_diverged",
    "diverged_fraction",
    "prior_shared_rank",
    "member_rank",
    "member_deviation_x",
    "member_diverged",
    "other_break_dates",
]


def write_corank_csv(events: list[CorankEvent], path: str) -> None:
    """Write co-rank findings to ``path`` — one row per member of each group."""
    _ensure_parent_dir(path)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CORANK_CSV_HEADER)
        for e in events:
            others = ",".join(d.isoformat() for d in e.other_break_dates)
            for m in e.members:
                writer.writerow(
                    [
                        e.pseudo_group_id,
                        e.rank_type,
                        _type_label(e.rank_type),
                        m.asin,
                        e.category,
                        e.group_date.isoformat(),
                        e.break_date.isoformat(),
                        e.severity,
                        e.n_members,
                        e.n_diverged,
                        f"{e.diverged_fraction:.3f}",
                        round(e.prior_shared_rank),
                        m.rank,
                        f"{m.deviation_x:.2f}",
                        int(m.diverged),
                        others,
                    ]
                )


def default_output_path(
    amerge_id: str, end_date: date | None, mode: str, days: int
) -> str:
    """Build a default CSV filename from mode, account id, time span and end."""
    safe_id = amerge_id.replace(":", "_").replace("/", "_")
    suffix = end_date.isoformat() if end_date else "latest"
    stem = {
        "anomaly": "rank_anomalies",
        "uniformity": "rank_uniformity",
        "corank": "rank_corank",
    }.get(mode, "rank_drops")
    filename = f"{stem}_{safe_id}_T-{days}_{suffix}.csv"
    return os.path.join(EXPORT_DIR, filename)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export ASINs whose best-seller rank dropped within a window.",
    )
    parser.add_argument("--amerge-id", required=True, help="Account amerge_id")
    parser.add_argument(
        "--mode",
        choices=("threshold", "anomaly", "uniformity", "corank"),
        default="threshold",
        help="Detection mode: 'threshold' (fixed %%/positions, default), "
        "'anomaly' (drops unusual vs each product's own history), "
        "'uniformity' (a parent's child ASINs suddenly fan out in main rank), or "
        "'corank' (data-driven pseudo-groups of shared-rank ASINs that fan out "
        "next day; no registry parent, no baseline).",
    )
    parser.add_argument(
        "--time-frame",
        type=parse_time_frame,
        default=parse_time_frame("T-7"),
        help="Detection window as 'T-N' days back (default T-7).",
    )
    parser.add_argument(
        "--min-pct",
        type=float,
        default=100.0,
        help="[threshold] Min %% day-over-day worsening to flag (default 100).",
    )
    parser.add_argument(
        "--min-positions",
        type=int,
        default=100,
        help="[threshold] Min day-over-day positions worsened (default 100).",
    )
    parser.add_argument(
        "--baseline-days",
        type=int,
        default=DEFAULT_BASELINE_DAYS,
        help=f"[anomaly] Trailing days to learn normal volatility "
        f"(default {DEFAULT_BASELINE_DAYS}).",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=DEFAULT_Z_THRESHOLD,
        help=f"[anomaly] Robust-z cutoff (default {DEFAULT_Z_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-sustain",
        type=int,
        default=DEFAULT_MIN_SUSTAIN,
        help=f"[anomaly] Observations the drop must persist "
        f"(default {DEFAULT_MIN_SUSTAIN}).",
    )
    parser.add_argument(
        "--min-floor-pct",
        type=float,
        default=DEFAULT_MIN_FLOOR_PCT,
        help=f"[anomaly] Volatility floor as a %% move "
        f"(default {DEFAULT_MIN_FLOOR_PCT}).",
    )
    parser.add_argument(
        "--cohort-denoise",
        action="store_true",
        help="[anomaly] Subtract each (rank_type, category) cohort's median "
        "daily move so account/category-wide shifts don't flag.",
    )
    parser.add_argument(
        "--min-cohort-size",
        type=int,
        default=DEFAULT_MIN_COHORT_SIZE,
        help=f"[anomaly] Min series in a cohort for de-noising to apply "
        f"(default {DEFAULT_MIN_COHORT_SIZE}).",
    )
    parser.add_argument(
        "--min-children",
        type=int,
        default=DEFAULT_MIN_CHILDREN,
        help=f"[uniformity/corank] Min ASINs a parent (or co-rank pseudo-group) "
        f"needs to test (default {DEFAULT_MIN_CHILDREN}).",
    )
    parser.add_argument(
        "--uniform-ratio",
        type=float,
        default=DEFAULT_UNIFORM_RATIO,
        help=f"[uniformity] max/min rank ratio still considered uniform "
        f"(default {DEFAULT_UNIFORM_RATIO}).",
    )
    parser.add_argument(
        "--divergence-ratio",
        type=float,
        default=DEFAULT_DIVERGENCE_RATIO,
        help=f"[uniformity] max/min ratio that counts as a fan-out "
        f"(default {DEFAULT_DIVERGENCE_RATIO}).",
    )
    parser.add_argument(
        "--child-deviation-factor",
        type=float,
        default=DEFAULT_CHILD_DEVIATION_FACTOR,
        help=f"[uniformity] a child diverged if rank >= factor x prior level "
        f"(default {DEFAULT_CHILD_DEVIATION_FACTOR}).",
    )
    parser.add_argument(
        "--anomalous-fraction",
        type=float,
        default=DEFAULT_ANOMALOUS_FRACTION,
        help="[uniformity] fraction of children diverging to call anomalous "
        "(default 2/3; pass 0.5 for >half, ~0.9 for all-but-few).",
    )
    parser.add_argument(
        "--relative-divergence",
        type=float,
        default=None,
        help="[uniformity] If set, the fan-out bar is K x the family's own normal "
        "spread instead of the absolute --divergence-ratio (e.g. 1.5 flags a "
        "family that spreads to 1.5x its usual tightness). Self-adjusts per family.",
    )
    parser.add_argument(
        "--rank-type",
        choices=("both", "1", "2"),
        default="both",
        help="Rank type: 1=subcategory, 2=main, both (default). "
        "In uniformity mode, 2=main only, 1=subcategories only, both=each "
        "(parent, category) checked separately.",
    )
    parser.add_argument(
        "--marketplace",
        default=None,
        help="Optional two-letter marketplace filter on the registry lookup.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV output path (default: rank_drops_<amerge_id>_<date>.csv).",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Suppress the preview table on stdout (only write the CSV).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    registry_dsn = os.environ.get("REGISTRY_DB_DSN")
    roas_dsn = os.environ.get("ROAS_DB_DSN")
    if not registry_dsn or not roas_dsn:
        print(
            "ERROR: REGISTRY_DB_DSN and ROAS_DB_DSN must be set (see .env.example).",
            file=sys.stderr,
        )
        return 2

    with (
        psycopg.connect(registry_dsn) as registry_conn,
        psycopg.connect(roas_dsn) as roas_conn,
    ):
        if args.mode == "uniformity":
            return _run_uniformity(registry_conn, roas_conn, args)
        if args.mode == "corank":
            return _run_corank(registry_conn, roas_conn, args)
        if args.mode == "anomaly":
            findings, summary = find_rank_anomalies(
                registry_conn,
                roas_conn,
                amerge_id=args.amerge_id,
                days=args.time_frame,
                baseline_days=args.baseline_days,
                z_threshold=args.z_threshold,
                min_sustain=args.min_sustain,
                min_floor_pct=args.min_floor_pct,
                rank_type=args.rank_type,
                marketplace=args.marketplace,
                cohort_denoise=args.cohort_denoise,
                min_cohort_size=args.min_cohort_size,
            )
        else:
            findings, summary = find_rank_drops(
                registry_conn,
                roas_conn,
                amerge_id=args.amerge_id,
                days=args.time_frame,
                min_pct=args.min_pct,
                min_positions=args.min_positions,
                rank_type=args.rank_type,
                marketplace=args.marketplace,
            )

    output_path = args.output or default_output_path(
        args.amerge_id, summary.get("window_end"), args.mode, args.time_frame
    )
    if args.mode == "anomaly":
        write_anomaly_csv(findings, output_path)
        table = render_anomaly_table(findings)
    else:
        write_csv(findings, output_path)
        table = render_table(findings)

    if not args.no_table:
        print(table if findings else "No ASINs met the rank-drop conditions.")

    flagged_asins = sorted(
        {f.move.asin if args.mode == "anomaly" else f.asin for f in findings}
    )
    mode_label = args.mode
    if args.mode == "anomaly" and args.cohort_denoise:
        mode_label = "anomaly+cohort-denoise"
    print(
        f"\n# {len(flagged_asins)} ASIN(s) / {len(findings)} series flagged out of "
        f"{summary['asin_count']} registered ({summary['series_scanned']} series "
        f"scanned) | mode {mode_label} | window {summary.get('window_start')} -> "
        f"{summary.get('window_end')} | CSV: {output_path}",
        file=sys.stderr,
    )
    return 0


def _run_uniformity(
    registry_conn: psycopg.Connection,
    roas_conn: psycopg.Connection,
    args: argparse.Namespace,
) -> int:
    """Run uniformity mode end to end (own output + summary shape)."""
    events, summary = find_uniformity_breaks(
        registry_conn,
        roas_conn,
        amerge_id=args.amerge_id,
        days=args.time_frame,
        baseline_days=args.baseline_days,
        min_children=args.min_children,
        uniform_ratio=args.uniform_ratio,
        divergence_ratio=args.divergence_ratio,
        child_deviation_factor=args.child_deviation_factor,
        anomalous_fraction=args.anomalous_fraction,
        marketplace=args.marketplace,
        rank_type=args.rank_type,
        relative_divergence=args.relative_divergence,
    )

    output_path = args.output or default_output_path(
        args.amerge_id, summary.get("window_end"), args.mode, args.time_frame
    )
    write_uniformity_csv(events, output_path)

    if not args.no_table:
        print(
            render_uniformity_table(events)
            if events
            else "No parents met the uniformity-break conditions."
        )

    n_anom = sum(e.severity == "anomalous" for e in events)
    n_susp = sum(e.severity == "suspect" for e in events)
    print(
        f"\n# {len(events)} parent(s) flagged ({n_anom} anomalous, {n_susp} suspect) "
        f"out of {summary['parents_scanned']} scanned "
        f"({summary['child_asin_count']} child ASINs) | mode uniformity | "
        f"window {summary.get('window_start')} -> {summary.get('window_end')} | "
        f"CSV: {output_path}",
        file=sys.stderr,
    )
    return 0


def _run_corank(
    registry_conn: psycopg.Connection,
    roas_conn: psycopg.Connection,
    args: argparse.Namespace,
) -> int:
    """Run co-rank mode end to end (own output + summary shape)."""
    events, summary = find_corank_breaks(
        registry_conn,
        roas_conn,
        amerge_id=args.amerge_id,
        days=args.time_frame,
        min_group_size=args.min_children,
        uniform_ratio=args.uniform_ratio,
        divergence_ratio=args.divergence_ratio,
        child_deviation_factor=args.child_deviation_factor,
        anomalous_fraction=args.anomalous_fraction,
        marketplace=args.marketplace,
        rank_type=args.rank_type,
    )

    output_path = args.output or default_output_path(
        args.amerge_id, summary.get("window_end"), args.mode, args.time_frame
    )
    write_corank_csv(events, output_path)

    if not args.no_table:
        print(
            render_corank_table(events)
            if events
            else "No co-rank pseudo-groups met the break conditions."
        )

    n_anom = sum(e.severity == "anomalous" for e in events)
    n_susp = sum(e.severity == "suspect" for e in events)
    print(
        f"\n# {len(events)} pseudo-group(s) flagged ({n_anom} anomalous, "
        f"{n_susp} suspect) out of {summary['categories_scanned']} co-rank "
        f"categories scanned ({summary['asin_count']} ASINs) | mode corank | "
        f"window {summary.get('window_start')} -> {summary.get('window_end')} | "
        f"CSV: {output_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
