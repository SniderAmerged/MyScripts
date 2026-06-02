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

It runs in one of two `--mode`s:

- **`threshold`** (default) — flags a day-over-day worsening of `--min-pct` percent
  **or** `--min-positions` positions within the window. Simple and explicit, but
  a flat threshold doesn't account for how volatile a given product normally is.
- **`anomaly`** — flags drops that are *unusual versus the product's own history*.
  It works in **log-rank** space (so main vs subcategory are comparable), learns
  each series' normal daily volatility over a trailing baseline (robust
  median + MAD), and flags a move only when it exceeds `--z-threshold` robust
  deviations **and** stays worse for `--min-sustain` observations (anti-blip).
  Best for surfacing likely Amazon detail-page issues out of normal churn.

In both modes results are sorted (by size / by z-score), the largest qualifying
move per ASIN/category is reported with the date + datetime it happened, and all
findings are exported to a CSV file.

### Usage

```bash
# Simple thresholds
uv run python scripts/rank_drop_checker.py \
    --amerge-id "US:US:1771995219729009" --time-frame T-7

# Anomaly detection (recommended for spotting PDP issues)
uv run python scripts/rank_drop_checker.py \
    --amerge-id "US:US:1771995219729009" --time-frame T-7 --mode anomaly
```

Common options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--amerge-id` | (required) | Account identifier |
| `--mode` | `threshold` | `threshold` or `anomaly` |
| `--time-frame` | `T-7` | Detection window as `T-N` days back |
| `--rank-type` | `both` | `1` (subcategory), `2` (main), or `both` |
| `--marketplace` | (none) | Optional two-letter marketplace filter |
| `--output` | auto | CSV path (default `rank_{drops,anomalies}_<id>_T-<N>_<date>.csv`) |
| `--no-table` | off | Suppress the stdout preview table |

`threshold`-mode options: `--min-pct` (100), `--min-positions` (100).

`anomaly`-mode options: `--baseline-days` (45), `--z-threshold` (3.5),
`--min-sustain` (2), `--min-floor-pct` (20).

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
