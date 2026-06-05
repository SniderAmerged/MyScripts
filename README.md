# MyScripts

Utility scripts. Managed with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv venv          # create the environment (if missing)
uv sync          # install dependencies
cp .env.example .env   # then fill in the DB credentials
```

## rank_drop_checker.py

Detects ASINs whose Amazon best-seller rank **worsened** significantly over a
recent window, for a given account (`amerge_id`).

It joins two Postgres servers (see `.env.example`):

- **Registry DB** — maps `amerge_id → ASIN` (`public.registry_api_asinregistry`).
- **ROAS DB** — daily rank history (`public.asin_ranks`).

Each ASIN can be ranked in several categories; every `(asin, type, category)`
series is compared independently. `type 1` = subcategory rank, `type 2` =
main/overall category rank. The category is parsed from `asin_ranks.id`
(3rd `:`-segment) so the same category is paired across dates.

It runs in one of four `--mode`s:

- **`threshold`** (default) — flags a day-over-day worsening of `--min-pct` percent
  **or** `--min-positions` positions within the window. Simple and explicit, but
  a flat threshold doesn't account for how volatile a given product normally is.
- **`anomaly`** — flags drops that are *unusual versus the product's own history*.
  It works in **log-rank** space (so main vs subcategory are comparable), learns
  each series' normal daily volatility over a trailing baseline (robust
  median + MAD), and flags a move only when it exceeds `--z-threshold` robust
  deviations **and** stays worse for `--min-sustain` observations (anti-blip).
  Best for surfacing likely Amazon detail-page issues out of normal churn.
- **`uniformity`** — looks *across a parent's child ASINs* (variations) in the
  **main category** only. Children normally share a near-identical rank; this
  mode flags a parent when that uniformity suddenly breaks (the ranks fan out).
  If a clear majority of children diverge it is **anomalous**; if only a few do
  while most stay uniform it is **suspect**. Output is one CSV row per child.
- **`corank`** — like `uniformity`, but groups are **reconstructed from the data
  each day** instead of from the registry's `parent_asin`. Because Amazon assigns
  the *same* best-seller rank to every child variation of a parent, ASINs that
  **share a rank within a category** are effectively one family. So for each
  consecutive day pair it forms "pseudo-groups" from co-ranked ASINs on day *D*,
  then checks whether they **fan out on day _D+1_**. Needs **no `parent_asin` and
  no baseline**, so it is robust to re-parenting, stale/missing registry links,
  and membership changes — and it also covers ASINs that have no parent. Same
  anomalous/suspect split; output is one CSV row per group member.

The first two modes report the largest qualifying move per ASIN/category with the
date it happened; `uniformity` reports each flagged parent and `corank` each
flagged pseudo-group (strongest event) with a per-member breakdown. All modes
export to a CSV file.

### Scan scope

The scan starts from the **Registry** (filtered to the given `amerge_id`) and then
looks those ASINs up in `asin_ranks` — it does **not** sweep the entire
`asin_ranks` table (which is global across all accounts). So the population is
this account's registered products that have rank history:

- `threshold` / `anomaly` scan every registered ASIN for the account that has
  rows in `asin_ranks`.
- `uniformity` additionally requires a **non-empty `parent_asin`** (it compares
  siblings, so standalone products are skipped) and that at least
  `--min-children` of a parent's siblings have rank data in a shared category.
- `corank` scans **all** the account's ASINs that have rank history (no
  `parent_asin` needed) and forms groups from co-ranked ASINs per category/day;
  a pseudo-group needs at least `--min-children` members.

Consequently, registered ASINs that are **not enrolled in rank tracking**
(`asin_track_requests`) have **no `asin_ranks` history** and therefore cannot be
flagged by any mode — they are invisible to the detector, not passed-as-clean.

### Usage

```bash
# Simple thresholds
uv run python scripts/rank_drop_checker.py \
    --amerge-id "US:US:1771995219729009" --time-frame T-7

# Anomaly detection (recommended for spotting PDP issues)
uv run python scripts/rank_drop_checker.py \
    --amerge-id "US:US:1771995219729009" --time-frame T-7 --mode anomaly

# Uniformity: a parent's variations suddenly fan out in main rank
uv run python scripts/rank_drop_checker.py \
    --amerge-id "US:US:1771995219729009" --time-frame T-14 --mode uniformity

# Co-rank: data-driven pseudo-groups (no parent/baseline) fan out next day.
# Recommended: --uniform-ratio 1.0 (exact rank = parent-pure families, low noise).
uv run python scripts/rank_drop_checker.py \
    --amerge-id "US:US:1771995219729009" --time-frame T-60 --mode corank \
    --rank-type both --uniform-ratio 1.0
```

Common options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--amerge-id` | (required) | Account identifier |
| `--mode` | `threshold` | `threshold`, `anomaly`, `uniformity`, or `corank` |
| `--time-frame` | `T-7` | Detection window as `T-N` days back |
| `--rank-type` | `both` | `1` (subcategory), `2` (main), or `both` (ignored in uniformity) |
| `--marketplace` | (none) | Optional two-letter marketplace filter |
| `--output` | auto | CSV path (default `export_output/rank_{drops,anomalies,uniformity,corank}_<id>_T-<N>_<date>.csv`) |
| `--no-table` | off | Suppress the stdout preview table |

`threshold`-mode options: `--min-pct` (100), `--min-positions` (100).

`anomaly`-mode options: `--baseline-days` (45), `--z-threshold` (3.5),
`--min-sustain` (2), `--min-floor-pct` (20).

`uniformity`-mode options: `--min-children` (3), `--uniform-ratio` (1.5),
`--divergence-ratio` (3.0), `--child-deviation-factor` (2.0),
`--anomalous-fraction` (0.667; pass `0.5` for ">half", ~`0.9` for "all-but-few").
Uses `--baseline-days` for the trailing uniformity baseline.

`corank`-mode options (no baseline): reuses `--min-children` (min group size, 3),
`--uniform-ratio` (1.5; the tolerance for ASINs to be "co-ranked" on day *D* —
set `1.0` for exact-rank, parent-pure grouping), `--divergence-ratio` (3.0; the
fan-out bar on day *D+1*), `--child-deviation-factor` (2.0) and
`--anomalous-fraction` (0.667).

> **Recommended:** run `corank` with `--uniform-ratio 1.0`. The default `1.5`
> merges a wide rank band into oversized groups (hundreds of ASINs), which
> over-flags; exact-rank grouping keeps each group a true variation family and
> sharply cuts noise. A good starting command:
>
> ```bash
> uv run python scripts/rank_drop_checker.py \
>     --amerge-id "<amerge_id>" --time-frame T-60 --mode corank \
>     --rank-type both --uniform-ratio 1.0
> ```

**Cohort de-noising** (opt-in, anomaly mode): add `--cohort-denoise` to subtract
each `(rank_type, category)` cohort's median daily move before scoring, so
account/category-wide shifts (Amazon recomputes) don't flag — only a product's
*excess* move over its peers counts. It only ever discounts a shared worsening,
never penalises a product for peers improving. `--min-cohort-size` (default 3)
sets how many series a cohort needs before its median is trusted. Lets you run a
looser `--z-threshold` without drowning in market-wide noise.

The preview table prints to stdout; a one-line summary (counts, window, CSV path)
prints to stderr.

## Development

```bash
uv run ruff check . --fix && uv run ruff format .
uv run pytest
```
