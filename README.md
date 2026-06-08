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
recent window, for a given account (`amerge_id`), using the **co-rank
pseudo-group** technique.

It joins two Postgres servers (see `.env.example`):

- **Registry DB** — maps `amerge_id → ASIN` (`public.registry_api_asinregistry`).
- **ROAS DB** — daily rank history (`public.asin_ranks`).

`type 1` = subcategory rank, `type 2` = main/overall category rank. The category
is parsed from `asin_ranks.id` (3rd `:`-segment) so the same category is paired
across dates.

### How `corank` works

Because Amazon assigns the **same best-seller rank to every child variation of a
parent**, ASINs that **share a rank within a category** are effectively one
variation family. So for each consecutive day pair the tool:

1. forms **pseudo-groups** from co-ranked ASINs on day *D* (clustering ranks
   within `--uniform-ratio`; `1.0` = exact rank);
2. checks whether those same ASINs **fan out on day _D+1_** (`max/min ≥
   --divergence-ratio`), counting a member as *diverged* when its rank worsens
   to `≥ --child-deviation-factor ×` the shared level;
3. labels the break **anomalous** when `≥ --anomalous-fraction` of members
   diverge, else **suspect** (needs ≥1).

Groups are reconstructed **from the data each day** — no registry `parent_asin`
and no baseline — so the technique is robust to re-parenting, stale/missing
registry links, and membership changes, and it covers ASINs that have no parent.
Each flagged pseudo-group is reported with its strongest break (other break
dates listed) and a per-member breakdown, exported one CSV row per member.

### Optional parent check (`--verify-parent`)

Off by default. When set, a final step annotates each flagged pseudo-group
against the **current** `parent_asin` groups in the registry — adding each
member's `parent_asin` plus the group's dominant parent and a **consistency**
score (fraction of members sharing it). This confirms whether a data-driven
co-rank group maps to a real registry family; it never changes which groups are
flagged. The detection itself never depends on the registry parent.

### Scan scope

The scan starts from the **Registry** (filtered to the given `amerge_id`) and
then looks those ASINs up in `asin_ranks` — it does **not** sweep the entire
`asin_ranks` table (which is global across all accounts). It scans **all** of the
account's ASINs that have rank history (no `parent_asin` needed) and forms groups
from co-ranked ASINs per category/day; a pseudo-group needs at least
`--min-group-size` members.

Consequently, registered ASINs that are **not enrolled in rank tracking**
(`asin_track_requests`) have **no `asin_ranks` history** and therefore cannot be
flagged — they are invisible to the detector, not passed-as-clean.

### Usage

```bash
# Recommended: exact-rank grouping (--uniform-ratio 1.0) = parent-pure
# families, lowest noise.
uv run python scripts/rank_drop_checker.py \
    --amerge-id "US:US:1771995219729009" --time-frame T-60 \
    --rank-type both --uniform-ratio 1.0

# Same, plus the optional registry parent_asin confirmation:
uv run python scripts/rank_drop_checker.py \
    --amerge-id "US:US:1771995219729009" --time-frame T-60 \
    --rank-type both --uniform-ratio 1.0 --verify-parent
```

Options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--amerge-id` | (required) | Account identifier |
| `--time-frame` | `T-7` | Detection window as `T-N` days back |
| `--rank-type` | `both` | `1` (subcategory), `2` (main), or `both` (each is its own grouping space) |
| `--min-group-size` | `3` | Min ASINs sharing a rank to form a pseudo-group (alias `--min-children`) |
| `--uniform-ratio` | `1.5` | Tolerance for "co-ranked" on day *D* (`max/min`). **Use `1.0`** for exact-rank, parent-pure grouping (lowest noise) |
| `--divergence-ratio` | `3.0` | `max/min` fan-out bar on day *D+1* |
| `--child-deviation-factor` | `2.0` | A member diverged if rank ≥ factor × shared level |
| `--anomalous-fraction` | `0.667` | Fraction diverging to call a break anomalous (`0.5` = >half, ~`0.9` = all-but-few) |
| `--verify-parent` | off | Final check: annotate groups against registry `parent_asin` (non-destructive) |
| `--marketplace` | (none) | Optional two-letter marketplace filter |
| `--output` | auto | CSV path (default `export_output/rank_corank_<id>_T-<N>_<date>.csv`) |
| `--no-table` | off | Suppress the stdout preview table |

> **Tip — if it flags almost everything:** the cause is usually the default
> `--uniform-ratio 1.5`, which merges a wide rank band into oversized groups
> (hundreds of ASINs) where one random mover always trips a *suspect*. Set
> `--uniform-ratio 1.0` (exact rank) so each group is a true family; raise
> `--divergence-ratio` / `--child-deviation-factor` to require bigger fan-outs.

The preview table prints to stdout; a one-line summary (counts, window, CSV path)
prints to stderr.

## Development

```bash
uv run ruff check . --fix && uv run ruff format .
uv run pytest
```
