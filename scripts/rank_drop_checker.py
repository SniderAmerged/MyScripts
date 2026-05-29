"""Detect ASINs whose Amazon best-seller rank worsened over a recent window.

Given an ``amerge_id``, this resolves the account's ASINs (Registry DB) and
compares each ASIN's rank now versus ``T-N`` days ago (ROAS DB ``asin_ranks``),
flagging the ones whose rank deteriorated past the given thresholds.

Usage:
    uv run python scripts/rank_drop_checker.py \\
        --amerge-id "EU:DE:1332086586752990" --time-frame T-7

See the repo plan / README for the data model. Connection strings come from
``REGISTRY_DB_DSN`` and ``ROAS_DB_DSN`` (loaded from ``.env``).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta

import psycopg
from dotenv import load_dotenv

# asin_ranks.type -> human label (from live-data inspection).
RANK_TYPE_LABELS: dict[int, str] = {1: "subcategory", 2: "main"}


@dataclass(frozen=True)
class RankPoint:
    """A single (rank, report_date) observation for an ASIN/type."""

    rank: int
    report_date: date


@dataclass(frozen=True)
class Finding:
    """A flagged ASIN/category rank series whose rank worsened past thresholds."""

    asin: str
    rank_type: int
    category: str
    baseline: RankPoint
    current: RankPoint

    @property
    def abs_change(self) -> int:
        """Positions the rank moved (positive = worsened)."""
        return self.current.rank - self.baseline.rank

    @property
    def pct_change(self) -> float:
        """Percentage worsening relative to the baseline rank."""
        return self.abs_change / self.baseline.rank * 100.0


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


def evaluate_rank_change(
    asin: str,
    rank_type: int,
    category: str,
    baseline: RankPoint | None,
    current: RankPoint | None,
    *,
    min_pct: float,
    min_positions: int,
) -> Finding | None:
    """Return a Finding if the rank worsened past BOTH thresholds, else None.

    Pure function (no DB) so it can be unit-tested. "Worsened" means the rank
    number increased. Returns None when either observation is missing or the
    change does not clear both the percentage and absolute-position thresholds.
    """
    if baseline is None or current is None or baseline.rank <= 0:
        return None
    abs_change = current.rank - baseline.rank
    if abs_change <= 0:  # unchanged or improved
        return None
    pct_change = abs_change / baseline.rank * 100.0
    if pct_change >= min_pct and abs_change >= min_positions:
        return Finding(asin, rank_type, category, baseline, current)
    return None


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
# Value carried per series: the observation plus the human category title.
Observation = tuple[RankPoint, str]


def fetch_current_ranks(
    conn: psycopg.Connection, asins: list[str], cur_date: date, rank_type: str
) -> dict[SeriesKey, Observation]:
    """Ranks at ``cur_date`` keyed by (asin, type, category_hash)."""
    frag, params = _type_filter(rank_type)
    sql = (
        "SELECT asin, type, split_part(id, ':', 3) AS cat_hash, title, rank "
        "FROM public.asin_ranks "
        "WHERE asin = ANY(%(asins)s) AND report_date = %(cur_date)s" + frag
    )
    with conn.cursor() as cur:
        cur.execute(sql, {"asins": asins, "cur_date": cur_date, **params})
        return {
            (asin, rtype, cat_hash): (RankPoint(rank=rank, report_date=cur_date), title)
            for asin, rtype, cat_hash, title, rank in cur.fetchall()
        }


def fetch_baseline_ranks(
    conn: psycopg.Connection, asins: list[str], base_target: date, rank_type: str
) -> dict[SeriesKey, Observation]:
    """Nearest rank on/before ``base_target`` per (asin, type, category_hash)."""
    frag, params = _type_filter(rank_type)
    sql = (
        "SELECT DISTINCT ON (asin, type, split_part(id, ':', 3)) "
        "asin, type, split_part(id, ':', 3) AS cat_hash, title, rank, report_date "
        "FROM public.asin_ranks "
        "WHERE asin = ANY(%(asins)s) AND report_date <= %(base_target)s" + frag + " "
        "ORDER BY asin, type, split_part(id, ':', 3), report_date DESC"
    )
    with conn.cursor() as cur:
        cur.execute(sql, {"asins": asins, "base_target": base_target, **params})
        return {
            (asin, rtype, cat_hash): (RankPoint(rank=rank, report_date=rdate), title)
            for asin, rtype, cat_hash, title, rank, rdate in cur.fetchall()
        }


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
) -> tuple[list[Finding], dict[str, object]]:
    """Run the full pipeline; return findings plus a small summary dict."""
    asins = fetch_account_asins(registry_conn, amerge_id, marketplace)
    summary: dict[str, object] = {
        "amerge_id": amerge_id,
        "asin_count": len(asins),
        "skipped_no_data": 0,
    }
    if not asins:
        return [], summary

    cur_date = fetch_latest_report_date(roas_conn)
    if cur_date is None:
        return [], summary
    base_target = cur_date - timedelta(days=days)
    summary["current_date"] = cur_date
    summary["baseline_target"] = base_target

    current = fetch_current_ranks(roas_conn, asins, cur_date, rank_type)
    baseline = fetch_baseline_ranks(roas_conn, asins, base_target, rank_type)

    findings: list[Finding] = []
    skipped = 0
    keys = set(current) | set(baseline)
    for key in sorted(keys):
        asin, rtype, _cat_hash = key
        base = baseline.get(key)
        curr = current.get(key)
        if base is None or curr is None:
            skipped += 1
            continue
        base_point, _ = base
        curr_point, category = curr
        finding = evaluate_rank_change(
            asin,
            rtype,
            category,
            base_point,
            curr_point,
            min_pct=min_pct,
            min_positions=min_positions,
        )
        if finding is not None:
            findings.append(finding)
    summary["skipped_no_data"] = skipped
    findings.sort(key=lambda f: f.pct_change, reverse=True)
    return findings, summary


def _type_label(rank_type: int) -> str:
    return RANK_TYPE_LABELS.get(rank_type, str(rank_type))


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def render_table(findings: list[Finding]) -> str:
    """Aligned text table of findings."""
    header = (
        f"{'ASIN':<12} {'TYPE':<12} {'CATEGORY':<28} "
        f"{'BASELINE':>10} {'CURRENT':>10} {'Δ POS':>8} {'Δ %':>9}  "
        f"{'BASE DATE':<11} {'CUR DATE':<11}"
    )
    lines = [header, "-" * len(header)]
    for f in findings:
        lines.append(
            f"{f.asin:<12} {_type_label(f.rank_type):<12} "
            f"{_truncate(f.category, 28):<28} "
            f"{f.baseline.rank:>10} {f.current.rank:>10} "
            f"{f.abs_change:>8} {f.pct_change:>8.1f}%  "
            f"{f.baseline.report_date.isoformat():<11} "
            f"{f.current.report_date.isoformat():<11}"
        )
    return "\n".join(lines)


def write_csv(findings: list[Finding]) -> None:
    """CSV of findings to stdout."""
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "asin",
            "rank_type",
            "rank_type_label",
            "category",
            "baseline_rank",
            "current_rank",
            "abs_change",
            "pct_change",
            "baseline_date",
            "current_date",
        ]
    )
    for f in findings:
        writer.writerow(
            [
                f.asin,
                f.rank_type,
                _type_label(f.rank_type),
                f.category,
                f.baseline.rank,
                f.current.rank,
                f.abs_change,
                f"{f.pct_change:.1f}",
                f.baseline.report_date.isoformat(),
                f.current.report_date.isoformat(),
            ]
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flag ASINs whose best-seller rank worsened over a time window.",
    )
    parser.add_argument("--amerge-id", required=True, help="Account amerge_id")
    parser.add_argument(
        "--time-frame",
        type=parse_time_frame,
        default=parse_time_frame("T-7"),
        help="Window as 'T-N' days back (default T-7).",
    )
    parser.add_argument(
        "--min-pct",
        type=float,
        default=100.0,
        help="Min %% rank worsening to flag (default 100).",
    )
    parser.add_argument(
        "--min-positions",
        type=int,
        default=100,
        help="Min absolute positions worsened to flag (default 100).",
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
        "--format",
        choices=("table", "csv"),
        default="table",
        help="Output format (default table).",
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

    if args.format == "csv":
        write_csv(findings)
    else:
        if findings:
            print(render_table(findings))
        else:
            print("No ASINs met the rank-drop conditions.")

    flagged_asins = sorted({f.asin for f in findings})
    print(
        f"\n# {len(flagged_asins)} ASIN(s) flagged out of {summary['asin_count']} "
        f"registered | window {summary.get('baseline_target')} -> "
        f"{summary.get('current_date')} | skipped (missing data): "
        f"{summary['skipped_no_data']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
