# Build phases & checkpoints

Status legend: ✅ done · 🚧 in progress · ⬜ pending

| Phase | Scope | Status |
|---|---|---|
| P0 | Monorepo scaffold, toolchain, PIT schema, DB up | ✅ |
| P1 | PIT store + daily ingest + tradeability gate | ✅ |
| P2 | Regime layer + classifier (label 24mo, sanity-check vs events) | ✅ |
| P3 | Location + trigger layers + triple-barrier labeler → first signals | 🚧 |
| P4 | Validation harness (CPCV/DSR/PBO) + cost model | 🚧 |
| P5 | Search + mutation loop wired to OOS scoreboard | ⬜ |
| P6 | Forward-adaptation monitor on live/paper | ⬜ |
| P7 | Signal output + paper-trade journal + API + dashboard | ⬜ |
| P8 | Horizon generalization (short_swing/day/0dte/scalp) | ⬜ |

Parallel tracks: SvelteKit dashboard (`apps/web`) and Python ML worker (`ml-worker`)
are built alongside the backend phases.

## §8 checkpoints (stop and report)
- **After P4** — feed a DELIBERATELY LEAKY feature through the harness and CONFIRM it's
  caught (in-sample looks perfect; CPCV-OOS + DSR collapse; PBO spikes > 0.5 → rejected).
  If leakage passes, the harness is broken — fix before anything else.
- **Before any promotion** — show the full promotion gate passing: DSR > 0, PBO < 0.5,
  cost-aware OOS expectancy > 0, positive regime-conditional contribution across ≥2 regimes.
- **Before live sizing** — ≥2-regime OOS stability + working §4 monitor + calibration curve.
- **Weekly** — report what decayed, what got retired, what adapted (the system's changelog).

## What's been verified so far
- P0: workspace compiles; migration applies; 4 Timescale hypertables created.
- P1: PIT leakage unit test passes (future-knowledge feature invisible at decision bar);
  FMP adapter verified live (real SPY bars, ^VIX); `se scan` shows 10/10 universe pass on
  real FMP data + 5 illustrative rejects with reasons; clippy clean; workspace tests green.
- P2: `regime-sanity-check` ALL PASS vs known events — COVID/2022 bear → risk_off/vol_expansion,
  2024 calm → risk_on/vol_compression. Macro PIT store leakage test passes. Unavailable series
  (SKEW tier-limited, FRED credit/liquidity without key) skipped, never fabricated.
- ML worker (`ml-worker`): 45 tests; **leakage checkpoint proven at the Python level** — a planted
  look-ahead feature scores ~0.99 in-sample but collapses OOS → DSR=0 → gate rejects; genuine edge
  passes (DSR=1.0, PBO=0); noise rejected.
- Dashboard (`apps/web`): svelte-check 0/0 (strict TS), build + 19 unit + 1 e2e green.
