# se_ml — Python ML sidecar

The authoritative implementation of the swing-X engine's financial-ML statistics:
triple-barrier labeling, meta-labeling, Combinatorial Purged CV (CPCV), the Deflated
Sharpe Ratio (DSR), the Probability of Backtest Overfit (PBO via CSCV), gradient boosting,
SHAP + permutation importance, isotonic/Platt calibration, and the hard promotion gate.

The Rust orchestrator (`se-mlclient`) calls this FastAPI service over HTTP. **Bulk data is
exchanged as Parquet/Arrow by file URI; request/response bodies carry only metadata and
metrics.** Field names are snake_case and the Rust side mirrors them exactly; the
contract lives in [`src/se_ml/contract.py`](src/se_ml/contract.py).

## Cardinal invariant

Improvement is measured ONLY out-of-sample. Validation is structurally purged + embargoed
(CPCV), corrected for selection bias (DSR), and overfit-tested (PBO). A deliberate-leakage
fixture proves the harness rejects a planted future-peeking feature
(`tests/test_leakage_fixture.py`).

## Quick start

```bash
cd ml-worker
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run mypy src
uv run uvicorn se_ml.server:app --port 8088   # GET /health -> {"status":"ok",...}
```

Configuration via env: `ML_WORKER_PORT` (8088), `ML_WORKER_DATA_DIR`,
`ML_WORKER_ARTIFACT_DIR`, `ML_WORKER_SEED` (42).

## Dataset contract (on-disk parquet)

A features+labels parquet contains:

| column           | meaning                                                              |
|------------------|---------------------------------------------------------------------|
| `ts`             | event timestamp (sorted ascending) — **required**                   |
| `t1`             | barrier-touch timestamp = label-window end — **required for CV**     |
| `label`          | realized return in **R units** (float) — **required for fit/validate** |
| `regime`         | optional regime tag (e.g. `bull`/`bear`/`chop`)                      |
| `layer__feature` | feature columns; prefix before `__` is the feature *layer*          |

## HTTP contract

All bodies are JSON; bulk data goes by `dataset_uri` (local path or `file://`).

### `GET /health`
```json
{ "status": "ok", "version": "0.1.0" }
```

### `POST /fit`
Request:
```json
{
  "dataset_uri": "/data/spy_swing.parquet",
  "horizon": "swing",
  "model_params": { "num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300 },
  "seed": 42
}
```
Response:
```json
{
  "model_id": "gbm-3f2a1b9c0d4e",
  "artifact_uri": "/tmp/se_ml/artifacts/gbm-3f2a1b9c0d4e.pkl",
  "in_sample_metrics": {
    "expectancy": 0.12, "profit_factor": 1.8, "sharpe": 1.4,
    "cvar5": -0.9, "mar": 2.1, "n": 1200
  }
}
```

### `POST /validate`
Request:
```json
{
  "dataset_uri": "/data/spy_swing.parquet",
  "horizon": "swing",
  "fold_spec": { "n_groups": 8, "k_test_groups": 2, "embargo_bars": 5, "purge": true },
  "n_trials": 50
}
```
Response:
```json
{
  "dsr": 0.61,
  "pbo": 0.08,
  "oos_expectancy_cost_aware": 0.04,
  "profit_factor": 1.35,
  "cvar5": -0.70,
  "mar": 1.20,
  "regime_contrib": { "bull": 0.06, "bear": 0.01, "chop": 0.03 },
  "n_regimes_positive": 3,
  "passed_gate": true
}
```

### `POST /calibrate`
Request:
```json
{ "dataset_uri": "/data/spy_swing.parquet", "model_id": "gbm-3f2a1b9c0d4e" }
```
Response:
```json
{
  "calibration_map": { "method": "isotonic", "x": [0.0, 0.5, 1.0], "y": [0.02, 0.41, 0.93] },
  "reliability_points": [ { "predicted": 0.1, "realized": 0.08, "count": 40 } ],
  "brier": 0.18
}
```

### `POST /importance`
Request:
```json
{ "dataset_uri": "/data/spy_swing.parquet", "model_id": "gbm-3f2a1b9c0d4e" }
```
Response:
```json
{
  "per_feature": { "momentum__signal": { "shap": 0.21, "permutation": 0.18 } },
  "per_layer":   { "momentum": { "shap": 0.40, "permutation": 0.35 } }
}
```

## Promotion gate (`src/se_ml/gates.py`)

`evaluate(...)` is a pure function and the single source of truth the Rust side
re-checks. A candidate is promoted only when **all** hold:

1. `dsr > 0`
2. `pbo < 0.5`
3. `oos_expectancy_cost_aware > 0`
4. `n_regimes_positive >= 2`

It returns each sub-condition (`dsr_ok`, `pbo_ok`, `oos_expectancy_ok`, `regimes_ok`) plus
the overall `passed`.

## Algorithms

- **Triple-barrier** (`labeling/triple_barrier.py`): ATR-sized target/stop + time barrier,
  first-touch, conservative intrabar ordering (stop wins on ambiguity), returns in R units.
- **Meta-labeling** (`labeling/meta_labeling.py`): secondary act/size model on a primary signal.
- **CPCV** (`cv/cpcv.py`): C(N,k) test combinations, label-window **purge** + **embargo**.
- **DSR** (`stats/dsr.py`): Bailey & López de Prado, corrects for trials, skew, kurtosis, length.
- **PBO** (`stats/pbo.py`): CSCV — logit of OOS rank of the IS-best across all splits.
- **Metrics** (`stats/metrics.py`): expectancy, profit factor, CVaR(5%), MAR/Calmar, Sharpe.
  **Win rate is deliberately not a selection metric.**
- **Calibration** (`calibration/calibrate.py`): isotonic + Platt, reliability curve, Brier.
- **Importance** (`importance/shap_perm.py`): SHAP + permutation, per-feature and per-layer.
