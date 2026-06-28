# swing-X — Self-Learning Swing Scanner

A self-learning swing-trading research engine over a fixed 10-ticker US equity-index ETF universe
(SPY, QQQ, IWM, DIA, XLF, XLK, XLE, SMH, XLV, XLU). It learns from history, validates on data it
never trained on, retrains nightly, detects its own decay, adapts, and emits concrete executable
signals (entry / stop / target / attribution).

**Cardinal invariant:** improvement is measured ONLY on out-of-sample (OOS) and live-forward data —
never on the training slice. This is enforced structurally (a point-in-time query layer, train/OOS
type separation, purged+embargoed cross-validation) and proven by a deliberate-leakage checkpoint.

## Architecture

- **Rust orchestrator** (cargo workspace under `crates/`) — ingest, point-in-time feature store,
  the conditional 4-layer feature engine, search/mutation loop, forward-adaptation monitor, signal
  generator, paper journal, axum API + WebSocket, nightly scheduler.
- **Python ML sidecar** (`ml-worker/`) — triple-barrier labeling, Combinatorial Purged CV, Deflated
  Sharpe Ratio, Probability of Backtest Overfit, gradient boosting, meta-labeling, SHAP/permutation
  importance, isotonic/Platt calibration. Talks to Rust over a clean job/result boundary.
- **SvelteKit dashboard** (`apps/web/`) — operator UI: signal scoreboard, signal detail with chart +
  driver attribution, strategy population, adaptation-monitor alerts, paper journal, weekly changelog.
- **PostgreSQL 16 + TimescaleDB** — bitemporal, provenance-tagged point-in-time store.

See `docs/` for ADRs and phase checklists, and `crates/*/` for per-module responsibilities.

## Quick start

```bash
# 1. Node (pin via nvm) + pnpm deps
nvm install        # reads .nvmrc -> 24.18.0
pnpm install

# 2. Database (Postgres + TimescaleDB via docker)
cp .env.example .env   # then set FMP_API_KEY
pnpm db:up

# 3. Migrations + backend
cargo build --workspace
# sqlx migrate run   (or: cargo run -p se-cli -- migrate)

# 4. Python ML worker
cd ml-worker && uv sync && uv run uvicorn se_ml.server:app --port 8088
```

## Data providers

Primary: **Financial Modeling Prep** (`/stable` API) — OHLCV, VIX term structure, treasury rates,
economic indicators, sector performance, DXY/commodities, index constituents. Secondary: **FRED**
(free) — credit spreads + net liquidity. A deterministic **mock** provider backs tests. Options-derived
signals (dealer GEX/charm/vanna/walls), tick order-flow, DIX and MOVE are `PROPRIETARY_FEATURE` hooks,
stubbed in v1.
