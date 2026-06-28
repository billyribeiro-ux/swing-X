# Architecture

A self-learning swing-trading research engine. Rust orchestrator + Python ML
sidecar + SvelteKit dashboard over a PostgreSQL 16 / TimescaleDB point-in-time store.

## The one invariant
Improvement is measured ONLY on out-of-sample (OOS) and live-forward data — never
on the training slice. Enforced structurally, in three places:
1. **Data layer** — `se-store::PitContext` hard-codes `as_of <= decision_ts` on every
   read. A value knowable only after the decision bar is invisible. (Proven by
   `crates/se-store/tests/pit_leakage.rs`.)
2. **Validation layer** — purge + embargo in Combinatorial Purged CV (Python worker).
3. **Type layer** — in-sample fit results and OOS scores are distinct types; only OOS
   scores can rank/promote. `win_rate` is never a ranking key.

## Components
- **Rust workspace (`crates/`)** — ingest, PIT store, the conditional 4-layer feature
  engine, regime classifier, triple-barrier labeler, search/mutation loop, adaptation
  monitor, signal generator, paper journal, axum API+WS, nightly orchestrator.
- **Python ML sidecar (`ml-worker/`)** — triple-barrier, CPCV, DSR, PBO, gradient
  boosting, meta-labeling, SHAP/permutation, isotonic/Platt calibration. Arrow/Parquet
  handoff; the only place that knows Python exists is `se-mlclient`.
- **SvelteKit dashboard (`apps/web/`)** — scoreboard, signal detail w/ chart+attribution,
  population, monitor alerts, journal, weekly changelog.

## Data flow (nightly)
ingest session → PIT store → roll walk-forward window → search/mutate candidates →
fit (in-sample) → score (OOS) → keep/kill → monitor live/paper → adapt → emit signals →
journal → weekly changelog.

## Provider mapping (v1)
- **FMP** (primary, `/stable`): daily bars, VIX complex (`^VIX/^VIX9D/^VIX3M/^VVIX`),
  treasury rates (2s10s), DXY/gold/oil/copper, ETF info/holdings, index constituents.
- **FRED** (free): HY/IG credit spreads, net liquidity (WALCL − WTREGEN − RRPONTSYD).
- **Mock**: deterministic synthetic data for tests.
- **Proprietary hooks** (stubbed): dealer GEX/charm/vanna/walls, tick order-flow, DIX,
  MOVE, `^SKEW`. Return `Unavailable` — never fabricated. v1 genomes are restricted to
  real (available/derivable) features so the gate validates real signal, not noise.

## Crate map
`se-core` (domain types incl. Horizon profiles) · `se-config` · `se-store` (PIT) ·
`se-provider` (DataProvider + FMP/FRED/Mock + proprietary) · `se-features` (layers 0–3
+ events) · `se-regime` · `se-labeler` · `se-mlclient` · `se-validation` (promotion gate)
· `se-search` · `se-monitor` · `se-signal` · `se-journal` · `se-orchestrator` · `se-api`
· `se-cli`.
