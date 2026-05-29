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

A series is flagged when, between `T-N` and the latest report date, the rank
number **increased** (worsened) by **both** at least `--min-pct` percent **and**
`--min-positions` positions. An ASIN is reported if any of its series is flagged.

### Usage

```bash
uv run python scripts/rank_drop_checker.py \
    --amerge-id "US:US:1771995219729009" \
    --time-frame T-7
```

Options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--amerge-id` | (required) | Account identifier |
| `--time-frame` | `T-7` | Window as `T-N` days back (bare integer also accepted) |
| `--min-pct` | `100` | Min % rank worsening to flag |
| `--min-positions` | `100` | Min absolute positions worsened to flag |
| `--rank-type` | `both` | `1` (subcategory), `2` (main), or `both` |
| `--marketplace` | (none) | Optional two-letter marketplace filter on the registry lookup |
| `--format` | `table` | `table` or `csv` |

Results print to stdout; a one-line summary (counts, window, skipped) prints to stderr.

## Development

```bash
uv run ruff check . --fix && uv run ruff format .
uv run pytest
```
