"""FastAPI app + routes for the ``se_ml`` sidecar.

Endpoints (see :mod:`se_ml.contract` for the exact JSON contract the Rust side mirrors):

  * GET  /health
  * POST /fit
  * POST /validate
  * POST /calibrate
  * POST /importance

Bulk data is exchanged by ``dataset_uri`` (Parquet); bodies carry only metadata + metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from . import __version__
from .calibration.calibrate import calibrate
from .config import CONFIG
from .contract import (
    CalibrateRequest,
    CalibrateResult,
    CalibrationMap,
    FitRequest,
    FitResult,
    HealthResponse,
    ImportanceRequest,
    ImportanceResult,
    ImportanceScore,
    InSampleMetrics,
    ReliabilityPoint,
    ValidateRequest,
    ValidateResult,
)
from .gates import evaluate as gate_evaluate
from .importance.shap_perm import compute_importance
from .io_arrow import (
    REGIME_COL,
    T1_COL,
    TS_COL,
    read_dataset,
    split_features_labels,
)
from .models.gbm import GbmModel, fit_gbm
from .stats import metrics as mx
from .stats.dsr import deflated_sharpe_ratio
from .stats.pbo import probability_of_backtest_overfit

app = FastAPI(title="se_ml", version=__version__)

# Round-trip trading cost charged per trade (in R) for cost-aware OOS expectancy.
DEFAULT_COST_PER_TRADE_R = 0.05


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _artifact_path(model_id: str) -> str:
    return str(CONFIG.artifact_dir / f"{model_id}.pkl")


def _load_model_for(model_id: str) -> GbmModel:
    path = CONFIG.artifact_dir / f"{model_id}.pkl"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"unknown model_id: {model_id}")
    return GbmModel.load(path)


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.post("/fit", response_model=FitResult)
def fit(req: FitRequest) -> FitResult:
    try:
        df = read_dataset(req.dataset_uri)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    X, y = split_features_labels(df)
    model = fit_gbm(X, y, seed=req.seed, params=req.model_params)
    model.save(CONFIG.artifact_dir)

    # In-sample edge metrics on the realized R labels (the achievable per-trade return).
    in_sample = mx.summary(y.to_numpy())
    return FitResult(
        model_id=model.model_id,
        artifact_uri=_artifact_path(model.model_id),
        in_sample_metrics=InSampleMetrics(
            expectancy=in_sample["expectancy"],
            profit_factor=in_sample["profit_factor"],
            sharpe=in_sample["sharpe"],
            cvar5=in_sample["cvar5"],
            mar=in_sample["mar"],
            n=int(in_sample["n"]),
        ),
    )


def _regime_contrib(df: pd.DataFrame, oos_returns: np.ndarray) -> tuple[dict[str, float], int]:
    """Per-regime mean OOS return and count of regimes with positive contribution."""
    if REGIME_COL not in df.columns:
        # No regime tags: treat the whole sample as one regime.
        contrib = {"all": float(np.mean(oos_returns)) if oos_returns.size else 0.0}
        return contrib, int(sum(v > 0 for v in contrib.values()))
    regimes = df[REGIME_COL].to_numpy()
    contrib = {}
    for r in pd.unique(regimes):
        mask = regimes == r
        vals = oos_returns[mask]
        contrib[str(r)] = float(np.mean(vals)) if vals.size else 0.0
    return contrib, int(sum(v > 0 for v in contrib.values()))


def _trial_grid(n_trials: int, features: list[str], seed: int) -> list[dict[str, object]]:
    """Generate ``n_trials`` candidate configs spanning the realistic search space.

    Each config varies BOTH LightGBM hyperparameters AND a random feature subset. The
    feature-subset variation is what makes PBO meaningful: on a dataset with a genuine
    edge, subsets that contain the signal consistently beat noise-only subsets (the IS-best
    keeps winning OOS -> low PBO); on pure noise every subset is equivalent, so the IS-best
    is OOS-random (-> high PBO). The grid is deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    leaves = [15, 31, 63]
    lrs = [0.03, 0.05, 0.1]
    grid: list[dict[str, object]] = []
    n = max(len(features), 1)
    for i in range(max(2, n_trials)):
        # Random non-empty feature subset (at least half the features, to keep models sane).
        k = rng.integers(max(1, n // 2), n + 1)
        subset = sorted(rng.choice(features, size=int(k), replace=False).tolist())
        grid.append(
            {
                "num_leaves": leaves[i % len(leaves)],
                "learning_rate": lrs[(i // len(leaves)) % len(lrs)],
                "n_estimators": 120,
                "_features": subset,  # consumed by _oos_returns_for_trial, not by LightGBM
            }
        )
    return grid


def _oos_returns_for_trial(
    X: pd.DataFrame,
    y_np: np.ndarray,
    splits: list,
    params: dict[str, object],
    seed: int,
) -> np.ndarray:
    """Full-timeline OOS per-observation returns for one trial, via CPCV.

    A row is an "acted" trade when the purged-trained model is confident (proba >= 0.5);
    its return is the realized R label, else 0. Rows never appearing in a test fold stay 0.
    The trial's feature subset (``params['_features']``) restricts the model's inputs.
    """
    model_params = {k: v for k, v in params.items() if not k.startswith("_")}
    features = params.get("_features")
    Xf = X[features] if isinstance(features, list) else X
    out = np.zeros(len(X), dtype=np.float64)
    for sp in splits:
        if sp.train_idx.size == 0 or sp.test_idx.size == 0:
            continue
        model = fit_gbm(
            Xf.iloc[sp.train_idx], y_np[sp.train_idx], seed=seed, params=model_params
        )
        proba = model.predict_proba(Xf.iloc[sp.test_idx])
        out[sp.test_idx] = np.where(proba >= 0.5, y_np[sp.test_idx], 0.0)
    return out


@app.post("/validate", response_model=ValidateResult)
def validate(req: ValidateRequest) -> ValidateResult:
    """Run CPCV over a trial grid, compute DSR/PBO, and apply the promotion gate.

    Each of ``n_trials`` hyperparameter configs is evaluated with purged+embargoed CPCV to
    produce a full-timeline OOS return series. PBO (CSCV) operates on the (T x n_trials)
    matrix: it asks whether the IS-best config stays good OOS. The headline OOS metrics
    (expectancy, DSR, profit factor, ...) report the IS-selected best config — exactly the
    config a naive search would promote — so the gate judges what would actually ship.
    """
    from .cv.cpcv import cpcv_splits  # local import keeps server import light

    try:
        df = read_dataset(req.dataset_uri).reset_index(drop=True)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if T1_COL not in df.columns:
        raise HTTPException(status_code=400, detail=f"validate requires a '{T1_COL}' column")

    X, y = split_features_labels(df)
    y_np = y.to_numpy()
    fs = req.fold_spec

    splits = cpcv_splits(
        event_ts=df[TS_COL].to_numpy(),
        t1=df[T1_COL].to_numpy(),
        n_groups=fs.n_groups,
        k_test_groups=fs.k_test_groups,
        embargo_bars=fs.embargo_bars,
        purge=fs.purge,
    )

    grid = _trial_grid(req.n_trials, list(X.columns), seed=CONFIG.seed)
    # (T x n_trials) OOS performance matrix; one column per candidate config.
    trial_returns = np.column_stack(
        [_oos_returns_for_trial(X, y_np, splits, p, seed=1000 + i) for i, p in enumerate(grid)]
    )

    # In-sample selection: the config a naive search would pick (best IS mean return).
    # We split the timeline in half and pick the best config on the first half, then report
    # honestly on the second half — the standard "select IS, judge OOS" discipline.
    t = trial_returns.shape[0]
    mid = t // 2
    is_perf = trial_returns[:mid].mean(axis=0)
    best = int(np.argmax(is_perf))
    selected = trial_returns[:, best]

    # Mask to rows actually traded by the selected config across the OOS (second) half so
    # the metrics reflect realized trades, not the zero-padded no-trade rows.
    oos_slice = selected[mid:]
    traded = oos_slice[oos_slice != 0.0]
    realized = traded if traded.size > 0 else oos_slice
    if realized.size == 0:
        raise HTTPException(status_code=422, detail="no OOS observations produced by CPCV")

    cost_aware = mx.cost_aware_returns(realized, DEFAULT_COST_PER_TRADE_R)
    dsr = deflated_sharpe_ratio(cost_aware, n_trials=max(1, req.n_trials))

    # PBO via CSCV on the full trial matrix (needs >= 2 trials).
    if trial_returns.shape[1] >= 2:
        usable_rows = trial_returns.shape[0]
        n_splits = min(16, usable_rows)
        n_splits -= n_splits % 2  # must be even
        n_splits = max(2, n_splits)
        try:
            pbo = probability_of_backtest_overfit(trial_returns, n_splits=n_splits)
        except ValueError:
            pbo = 1.0
    else:
        pbo = 1.0

    # Regime attribution on the OOS-half traded rows of the selected config.
    oos_half_df = df.iloc[mid:].reset_index(drop=True)
    if traded.size > 0:
        traded_mask = oos_slice != 0.0
        contrib, n_pos = _regime_contrib(
            oos_half_df.loc[traded_mask].reset_index(drop=True), realized
        )
    else:
        contrib, n_pos = _regime_contrib(oos_half_df, realized)

    gate = gate_evaluate(
        dsr=dsr,
        pbo=pbo,
        oos_expectancy_cost_aware=float(np.mean(cost_aware)),
        n_regimes_positive=n_pos,
    )

    return ValidateResult(
        dsr=float(dsr),
        pbo=float(pbo),
        oos_expectancy_cost_aware=float(np.mean(cost_aware)),
        profit_factor=mx.profit_factor(cost_aware),
        cvar5=mx.cvar5(cost_aware),
        mar=mx.mar(cost_aware),
        regime_contrib=contrib,
        n_regimes_positive=n_pos,
        passed_gate=gate["passed"],
    )


@app.post("/calibrate", response_model=CalibrateResult)
def calibrate_endpoint(req: CalibrateRequest) -> CalibrateResult:
    try:
        df = read_dataset(req.dataset_uri)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    model = _load_model_for(req.model_id)

    X, y = split_features_labels(df)
    scores = model.predict_proba(X)
    outcomes = (y.to_numpy() > 0).astype(int)
    result = calibrate(scores, outcomes)

    return CalibrateResult(
        calibration_map=CalibrationMap(method=result.method, x=result.map_x, y=result.map_y),
        reliability_points=[
            ReliabilityPoint(predicted=p.predicted, realized=p.realized, count=p.count)
            for p in result.reliability
        ],
        brier=result.brier,
    )


@app.post("/importance", response_model=ImportanceResult)
def importance_endpoint(req: ImportanceRequest) -> ImportanceResult:
    try:
        df = read_dataset(req.dataset_uri)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    model = _load_model_for(req.model_id)

    X, y = split_features_labels(df)
    per_feature, per_layer = compute_importance(model, X, y)

    return ImportanceResult(
        per_feature={
            f: ImportanceScore(shap=s.shap, permutation=s.permutation)
            for f, s in per_feature.items()
        },
        per_layer={
            layer_name: ImportanceScore(shap=s.shap, permutation=s.permutation)
            for layer_name, s in per_layer.items()
        },
    )
