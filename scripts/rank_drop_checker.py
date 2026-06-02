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


def write_csv(moves: list[Move], path: str) -> None:
    """Write findings to ``path`` as CSV (one row per ASIN/category move)."""
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


def default_output_path(
    amerge_id: str, end_date: date | None, mode: str, days: int
) -> str:
    """Build a default CSV filename from mode, account id, time span and end."""
    safe_id = amerge_id.replace(":", "_").replace("/", "_")
    suffix = end_date.isoformat() if end_date else "latest"
    stem = "rank_anomalies" if mode == "anomaly" else "rank_drops"
    return f"{stem}_{safe_id}_T-{days}_{suffix}.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export ASINs whose best-seller rank dropped within a window.",
    )
    parser.add_argument("--amerge-id", required=True, help="Account amerge_id")
    parser.add_argument(
        "--mode",
        choices=("threshold", "anomaly"),
        default="threshold",
        help="Detection mode: 'threshold' (fixed %%/positions, default) or "
        "'anomaly' (drops unusual vs each product's own history).",
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
        "--rank-type",
        choices=("both", "1", "2"),
        default="both",
        help="Rank type: 1=subcategory, 2=main, both (default).",
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


if __name__ == "__main__":
    raise SystemExit(main())
