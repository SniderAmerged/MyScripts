"""Detect Amazon best-seller rank drops via co-rank pseudo-groups (`corank`).

Given an ``amerge_id``, this resolves the account's ASINs (Registry DB), then
scans their daily rank history (ROAS DB ``asin_ranks``) over the last ``T-N``
days. Because Amazon assigns the *same* best-seller rank to every child
variation of a parent, ASINs that **share a rank within a category** are
effectively one variation family. For each consecutive day pair it forms
"pseudo-groups" from co-ranked ASINs on day *D*, then flags groups that **fan
out on day _D+1_** (a worsening of rank). This needs no registry ``parent_asin``
and no baseline, so it is robust to re-parenting, stale/missing registry links,
and membership changes.

Optionally (``--verify-parent``, off by default) a final check annotates each
flagged pseudo-group against the *current* ``parent_asin`` groups in the
registry — confirming whether a data-driven group maps to a real registry
family — without changing which groups are flagged.

Usage:
    uv run python scripts/rank_drop_checker.py \\
        --amerge-id "US:US:1771995219729009" --time-frame T-60 \\
        --rank-type both --uniform-ratio 1.0

Connection strings come from ``REGISTRY_DB_DSN`` and ``ROAS_DB_DSN`` (.env).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from statistics import median

import psycopg
from dotenv import load_dotenv

# Directory for generated CSV exports.
EXPORT_DIR = "export_output"

# asin_ranks.type -> human label (from live-data inspection).
RANK_TYPE_LABELS: dict[int, str] = {1: "subcategory", 2: "main"}

# --- corank-mode defaults --------------------------------------------------
# Min ASINs sharing a rank for a co-rank pseudo-group to be tested.
DEFAULT_MIN_GROUP_SIZE = 3
# Tolerance for ASINs to count as "co-ranked" on day D: max_rank <= ratio * min.
# 1.0 == exact shared rank (parent-pure families); higher merges a wider band.
DEFAULT_UNIFORM_RATIO = 1.5
# A group "breaks" on day D+1 when its max_rank >= this ratio * min_rank.
DEFAULT_DIVERGENCE_RATIO = 3.0
# A member has "diverged" when its rank >= this factor * the shared level.
DEFAULT_CHILD_DEVIATION_FACTOR = 2.0
# Fraction of members that must diverge to call the break anomalous (else suspect).
DEFAULT_ANOMALOUS_FRACTION = 2 / 3
# Calendar lead-in so the first in-window day still has a prior observation to
# pair with (covers weekend/missing-day gaps in the daily series).
CORANK_LEAD_IN_DAYS = 3
# Sort key for severity: anomalous before suspect.
_SEVERITY_ORDER = {"anomalous": 0, "suspect": 1}


@dataclass(frozen=True)
class Observation:
    """A single daily rank observation for one ASIN/category series."""

    report_date: date
    rank: int
    captured_at: datetime  # asin_ranks.date_add — when the row was recorded


@dataclass(frozen=True)
class ChildRank:
    """One member ASIN's standing on a pseudo-group's break day."""

    asin: str
    rank: int
    deviation_x: float  # rank / prior shared level (>=1 worse, <1 better)
    diverged: bool
    parent_asin: str | None = None  # set only by --verify-parent


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
    # Populated only by the optional --verify-parent check:
    dominant_parent: str | None = None
    parent_consistency: float | None = None  # frac of members sharing it

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


def _spread_ratio(ranks: list[int]) -> float:
    """max/min of a group's ranks (1.0 == perfectly uniform)."""
    lo = min(ranks)
    return max(ranks) / lo if lo > 0 else float("inf")


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
        events.append(replace(strongest, other_break_dates=others))
    return events


def annotate_with_parent(
    events: list[CorankEvent], parent_of: dict[str, str]
) -> list[CorankEvent]:
    """Final check: tag each flagged pseudo-group against registry parents.

    Pure function (no DB). For every member, attaches its current
    ``parent_asin`` (``""`` when none) from ``parent_of``, and records the
    group's ``dominant_parent`` plus ``parent_consistency`` (the fraction of
    members sharing that parent). Confirms whether a data-driven co-rank group
    maps to a real registry family; it never drops or re-flags groups.
    """
    annotated: list[CorankEvent] = []
    for e in events:
        members = [replace(m, parent_asin=parent_of.get(m.asin, "")) for m in e.members]
        parents = [m.parent_asin for m in members if m.parent_asin]
        if parents:
            dominant, count = Counter(parents).most_common(1)[0]
            consistency = count / len(members)
        else:
            dominant, consistency = "", 0.0
        annotated.append(
            replace(
                e,
                members=members,
                dominant_parent=dominant,
                parent_consistency=consistency,
            )
        )
    return annotated


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
    """Map each child ASIN (product_code) -> its current parent_asin.

    Only products with a non-empty parent are returned. If a product_code
    appears under more than one parent, the first seen wins (rare). Used only by
    the optional ``--verify-parent`` check.
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
    verify_parent: bool = False,
) -> tuple[list[CorankEvent], dict[str, object]]:
    """Co-rank pipeline: flag pseudo-groups (shared-rank ASINs) that fan out.

    Families are reconstructed each day from ASINs that share a rank within a
    category — no registry ``parent_asin`` and no baseline. ``rank_type`` "2"
    main, "1" subcategory, or "both" (each type is its own grouping space). When
    ``verify_parent`` is set, the flagged groups are annotated against the
    current registry parent groups as a final, non-destructive check.
    """
    asins = fetch_account_asins(registry_conn, amerge_id, marketplace)
    summary: dict[str, object] = {
        "amerge_id": amerge_id,
        "asin_count": len(asins),
        "categories_scanned": 0,
        "verify_parent": verify_parent,
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

    if verify_parent:
        parent_of = fetch_parent_map(registry_conn, amerge_id, marketplace)
        events = annotate_with_parent(events, parent_of)

    events.sort(key=lambda e: (_SEVERITY_ORDER.get(e.severity, 9), -e.spread_ratio))
    return events, summary


def _type_label(rank_type: int) -> str:
    return RANK_TYPE_LABELS.get(rank_type, str(rank_type))


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def render_corank_table(events: list[CorankEvent], *, verify_parent: bool) -> str:
    """Aligned text table: one line per flagged co-rank pseudo-group."""
    header = (
        f"{'PSEUDO-GROUP':<16} {'TYPE':<12} {'SEVERITY':<10} {'CATEGORY':<22} "
        f"{'GROUP→BREAK':<23} {'MEMB':>5} {'DIV':>4} {'FRAC':>5} "
        f"{'SHARED':>8} {'SPREAD×':>8}  {'OTHER BREAKS':<24}"
    )
    if verify_parent:
        header += f" {'PARENT (CONSIST.)':<22}"
    lines = [header, "-" * len(header)]
    for e in events:
        others = ",".join(d.isoformat() for d in e.other_break_dates) or "-"
        window = f"{e.group_date.isoformat()}→{e.break_date.isoformat()}"
        line = (
            f"{e.pseudo_group_id:<16} {_type_label(e.rank_type):<12} {e.severity:<10} "
            f"{_truncate(e.category, 22):<22} {window:<23} "
            f"{e.n_members:>5} {e.n_diverged:>4} {e.diverged_fraction:>5.0%} "
            f"{round(e.prior_shared_rank):>8} {e.spread_ratio:>7.1f}x  "
            f"{_truncate(others, 24):<24}"
        )
        if verify_parent:
            tag = (
                f"{e.dominant_parent or '-'} ({e.parent_consistency:.0%})"
                if e.parent_consistency is not None
                else "-"
            )
            line += f" {_truncate(tag, 22):<22}"
        lines.append(line)
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
    # populated only with --verify-parent (blank otherwise):
    "member_parent_asin",
    "group_dominant_parent",
    "group_parent_consistency",
]


def _ensure_parent_dir(path: str) -> None:
    """Create the directory holding ``path`` if it does not yet exist."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_corank_csv(events: list[CorankEvent], path: str) -> None:
    """Write co-rank findings to ``path`` — one row per member of each group."""
    _ensure_parent_dir(path)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CORANK_CSV_HEADER)
        for e in events:
            others = ",".join(d.isoformat() for d in e.other_break_dates)
            consistency = (
                f"{e.parent_consistency:.3f}"
                if e.parent_consistency is not None
                else ""
            )
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
                        m.parent_asin if m.parent_asin is not None else "",
                        e.dominant_parent if e.dominant_parent is not None else "",
                        consistency,
                    ]
                )


def default_output_path(amerge_id: str, end_date: date | None, days: int) -> str:
    """Build the default CSV filename from account id, time span and end date."""
    safe_id = amerge_id.replace(":", "_").replace("/", "_")
    suffix = end_date.isoformat() if end_date else "latest"
    filename = f"rank_corank_{safe_id}_T-{days}_{suffix}.csv"
    return os.path.join(EXPORT_DIR, filename)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flag co-rank pseudo-groups whose shared rank fans out "
        "(corank technique).",
    )
    parser.add_argument("--amerge-id", required=True, help="Account amerge_id")
    parser.add_argument(
        "--time-frame",
        type=parse_time_frame,
        default=parse_time_frame("T-7"),
        help="Detection window as 'T-N' days back (default T-7).",
    )
    parser.add_argument(
        "--rank-type",
        choices=("both", "1", "2"),
        default="both",
        help="Rank type to scan: 1=subcategory, 2=main, both (default). Each "
        "type is its own co-rank grouping space.",
    )
    parser.add_argument(
        "--min-group-size",
        "--min-children",
        dest="min_group_size",
        type=int,
        default=DEFAULT_MIN_GROUP_SIZE,
        help=f"Min ASINs sharing a rank to form a pseudo-group "
        f"(default {DEFAULT_MIN_GROUP_SIZE}).",
    )
    parser.add_argument(
        "--uniform-ratio",
        type=float,
        default=DEFAULT_UNIFORM_RATIO,
        help=f"Tolerance for ASINs to be 'co-ranked' on day D — max/min rank "
        f"ratio (default {DEFAULT_UNIFORM_RATIO}). Use 1.0 for exact-rank, "
        f"parent-pure grouping (recommended, lowest noise).",
    )
    parser.add_argument(
        "--divergence-ratio",
        type=float,
        default=DEFAULT_DIVERGENCE_RATIO,
        help=f"max/min ratio on day D+1 that counts as a fan-out "
        f"(default {DEFAULT_DIVERGENCE_RATIO}).",
    )
    parser.add_argument(
        "--child-deviation-factor",
        type=float,
        default=DEFAULT_CHILD_DEVIATION_FACTOR,
        help=f"A member diverged if its rank >= factor x the shared level "
        f"(default {DEFAULT_CHILD_DEVIATION_FACTOR}).",
    )
    parser.add_argument(
        "--anomalous-fraction",
        type=float,
        default=DEFAULT_ANOMALOUS_FRACTION,
        help="Fraction of members diverging to call a break anomalous "
        "(default 2/3; pass 0.5 for >half, ~0.9 for all-but-few).",
    )
    parser.add_argument(
        "--verify-parent",
        action="store_true",
        help="Final check (off by default): annotate each flagged pseudo-group "
        "against the current parent_asin groups in the registry, without "
        "changing which groups are flagged.",
    )
    parser.add_argument(
        "--marketplace",
        default=None,
        help="Optional two-letter marketplace filter on the registry lookup.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV output path (default: export_output/rank_corank_<id>_T-<N>_"
        "<date>.csv).",
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
        events, summary = find_corank_breaks(
            registry_conn,
            roas_conn,
            amerge_id=args.amerge_id,
            days=args.time_frame,
            min_group_size=args.min_group_size,
            uniform_ratio=args.uniform_ratio,
            divergence_ratio=args.divergence_ratio,
            child_deviation_factor=args.child_deviation_factor,
            anomalous_fraction=args.anomalous_fraction,
            marketplace=args.marketplace,
            rank_type=args.rank_type,
            verify_parent=args.verify_parent,
        )

    output_path = args.output or default_output_path(
        args.amerge_id, summary.get("window_end"), args.time_frame
    )
    write_corank_csv(events, output_path)

    if not args.no_table:
        print(
            render_corank_table(events, verify_parent=args.verify_parent)
            if events
            else "No co-rank pseudo-groups met the break conditions."
        )

    n_anom = sum(e.severity == "anomalous" for e in events)
    n_susp = sum(e.severity == "suspect" for e in events)
    mode_label = "corank+verify-parent" if args.verify_parent else "corank"
    print(
        f"\n# {len(events)} pseudo-group(s) flagged ({n_anom} anomalous, "
        f"{n_susp} suspect) out of {summary['categories_scanned']} co-rank "
        f"categories scanned ({summary['asin_count']} ASINs) | mode {mode_label} | "
        f"window {summary.get('window_start')} -> {summary.get('window_end')} | "
        f"CSV: {output_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
