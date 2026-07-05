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
from .labeling.meta_labeling import make_meta_labels
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


def _oos_proba_for_trial(
    X: pd.DataFrame,
    y_np: np.ndarray,
    splits: list,
    params: dict[str, object],
    seed: int,
) -> np.ndarray:
    """Full-timeline OOS per-observation probabilities for one trial, via CPCV.

    ``proba[i]`` is the purged-CPCV test-fold probability of a profitable trade for row ``i``;
    rows that never appear in any test fold are ``np.nan``. The trial's feature subset
    (``params['_features']``) restricts the model's inputs. This is the raw probability layer
    on top of which an acting threshold tau is applied.
    """
    model_params = {k: v for k, v in params.items() if not k.startswith("_")}
    features = params.get("_features")
    Xf = X[features] if isinstance(features, list) else X
    out = np.full(len(X), np.nan, dtype=np.float64)
    for sp in splits:
        if sp.train_idx.size == 0 or sp.test_idx.size == 0:
            continue
        model = fit_gbm(
            Xf.iloc[sp.train_idx], y_np[sp.train_idx], seed=seed, params=model_params
        )
        out[sp.test_idx] = model.predict_proba(Xf.iloc[sp.test_idx])
    return out


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

    Thin wrapper over :func:`_oos_proba_for_trial`: this thresholded-at-0.5 return matrix
    drives PBO and IS config selection and is kept UNCHANGED so that selection discipline is
    untouched by the precision/threshold layer.
    """
    proba = _oos_proba_for_trial(X, y_np, splits, params, seed)
    acted = np.where(np.isnan(proba), False, proba >= 0.5)
    return np.where(acted, y_np, 0.0)


def _candidate_grid(proba: np.ndarray) -> np.ndarray:
    """Deterministic candidate threshold grid: proba deciles ∪ coarse linspace.

    The grid is the unique data quantile deciles of ``proba`` plus a coarse linspace over
    [0.30, 0.90] (deduplicated, sorted ascending). Deterministic given ``proba``.
    """
    deciles = np.quantile(proba, np.linspace(0.0, 0.9, 10))
    coarse = np.linspace(0.30, 0.90, 7)
    return np.unique(np.concatenate([deciles, coarse]))


def _threshold_stats(
    proba: np.ndarray,
    r: np.ndarray,
    tau: float,
    cost: float,
) -> tuple[int, float, float, float]:
    """Acting stats for ``tau`` on one (proba, r) fold.

    Returns ``(n_acted, precision, recall, expectancy)`` where precision is the fraction of
    acted rows that are NET-profitable (``make_meta_labels`` R > ``cost``), recall is
    #acted-and-net-profitable / #net-profitable, and expectancy is the cost-aware mean R over
    acted rows (``-inf`` when nothing acted, so it never beats a real candidate on tie-breaks).

    Profitability is measured NET of the round-trip cost so that "precision" is P(net profit |
    acted), not the cost-blind P(R>0) — a 0<R<cost trade is a net loss and must NOT count as a
    precision win (else the shipped conviction + live floor pass net-negative setups).
    """
    labels = make_meta_labels(r, cost)
    n_profit = int(labels.sum())
    acted = proba >= tau
    n_acted = int(acted.sum())
    if n_acted == 0:
        return 0, 0.0, 0.0, -np.inf
    acted_profit = int((acted & (labels == 1)).sum())
    precision = acted_profit / n_acted
    recall = (acted_profit / n_profit) if n_profit > 0 else 0.0
    expectancy = float(np.mean(mx.cost_aware_returns(r[acted], cost)))
    return n_acted, float(precision), float(recall), expectancy


def _passes_constraints(
    proba: np.ndarray,
    r: np.ndarray,
    tau: float,
    cost: float,
    min_acted: int,
    require_recall: bool,
) -> bool:
    """Whether ``tau`` clears the acting constraints on one fold.

    Constraints: cost-aware expectancy strictly > 0, n_acted >= ``min_acted``, and (only when
    ``require_recall``) IS recall >= 0.10. The recall floor is checked on the primary
    (selection) fold; the secondary robustness fold only re-checks expectancy + min acted.
    """
    n_acted, _prec, recall, expectancy = _threshold_stats(proba, r, tau, cost)
    if n_acted < min_acted or expectancy <= 0.0:
        return False
    return not (require_recall and recall < 0.10)


def _rank_precision_first(
    proba: np.ndarray,
    r: np.ndarray,
    grid: np.ndarray,
    cost: float,
    min_acted: int,
) -> list[float]:
    """Candidate thresholds that clear the constraints, best-first (precision-first).

    A candidate qualifies only if it clears ALL of: cost-aware expectancy > 0,
    n_acted >= ``min_acted``, IS recall >= 0.10. Qualifying candidates are ranked by:
    (1) higher in-sample precision, (2) higher cost-aware expectancy, (3) lower threshold
    (more coverage). Returns the ordered tau list (empty if none qualify). Deterministic.
    """
    scored: list[tuple[float, float, float, float]] = []
    for tau in grid:
        n_acted, precision, recall, expectancy = _threshold_stats(proba, r, float(tau), cost)
        if n_acted < min_acted or expectancy <= 0.0 or recall < 0.10:
            continue
        # Sort key: precision desc, expectancy desc, tau asc. Negate the descending fields.
        scored.append((-precision, -expectancy, float(tau), float(tau)))
    scored.sort()
    return [tau for _np, _ne, _t, tau in scored]


def select_act_threshold(
    proba_is: np.ndarray,
    r_is: np.ndarray,
    cost: float,
) -> float:
    """Select the PRECISION-FIRST acting threshold tau* on the in-sample half (no OOS peeking).

    ``proba_is`` and ``r_is`` are the FINITE-proba IS-half probability and realized-R arrays
    (same length, already masked to rows that appeared in a test fold). tau* MAXIMIZES
    in-sample PRECISION — the fraction of acted IS rows that are profitable
    (``make_meta_labels`` R > 0, acted = proba >= tau) — subject to ALL of:

      * cost-aware IS expectancy strictly > 0 (never trade a precise-but-unprofitable tau),
      * number of acted rows >= ``max(8, ceil(0.10 * n_is_finite))``, and
      * IS recall >= 0.10  (recall = #acted-and-profitable / #profitable).

    Ties in precision are broken by higher cost-aware expectancy, then by lower threshold
    (more coverage).

    ROBUSTNESS (anti threshold-overfit): the IS half is split into two deterministic
    sub-folds by index — first 70% (selection) and last 30% (confirmation). Candidates are
    ranked precision-first on the FIRST sub-fold; the chosen tau* is the best-ranked one that
    ALSO clears the constraints (expectancy > 0, n_acted >= min on that sub-fold) on the
    SECOND sub-fold. We walk down the ranked list until one holds on BOTH sub-folds. This
    rejects a tau that only looks precise on a sliver of the IS half.

    The candidate grid is the unique IS-half proba deciles plus a coarse linspace over
    [0.30, 0.90] (deduplicated, sorted). If NO candidate satisfies the constraints on both
    sub-folds, falls back to ``0.5`` — preserving the legacy proba >= 0.5 behavior.
    Deterministic given inputs.
    """
    n_is_finite = int(proba_is.size)
    if n_is_finite == 0:
        return 0.5

    grid = _candidate_grid(proba_is)

    # Deterministic 70/30 split of the IS half by index.
    cut = int(np.floor(0.70 * n_is_finite))
    proba_a, r_a = proba_is[:cut], r_is[:cut]
    proba_b, r_b = proba_is[cut:], r_is[cut:]

    # If a sub-fold is too small to be meaningful, fall back to single-fold selection over
    # the whole IS half (still precision-first under the same constraints).
    if proba_a.size == 0 or proba_b.size == 0:
        min_acted = max(8, int(np.ceil(0.10 * n_is_finite)))
        ranked = _rank_precision_first(proba_is, r_is, grid, cost, min_acted)
        return ranked[0] if ranked else 0.5

    min_acted_a = max(8, int(np.ceil(0.10 * proba_a.size)))
    # Min-acted floor on the (smaller) confirmation fold scales with its own size.
    min_acted_b = max(8, int(np.ceil(0.10 * proba_b.size)))

    # Rank precision-first on the selection sub-fold, then require each candidate also holds
    # (expectancy > 0, min acted) on the confirmation sub-fold; take the first that does.
    ranked = _rank_precision_first(proba_a, r_a, grid, cost, min_acted_a)
    for tau in ranked:
        if _passes_constraints(
            proba_b, r_b, tau, cost, min_acted_b, require_recall=False
        ):
            return tau
    return 0.5


def precision_recall_at(
    proba: np.ndarray,
    r: np.ndarray,
    tau: float,
    cost: float,
) -> tuple[float, float, int]:
    """Precision, recall and acted-count for acting at threshold ``tau``.

    ``proba`` and ``r`` are same-length finite-proba arrays (probability and realized R).
    Acted = ``proba >= tau``. Returns ``(precision, recall, n_acted)`` where:

      * precision = fraction of ACTED trades that were NET-profitable (R > ``cost``); 0.0 if
        none acted.
      * recall    = #acted-and-net-profitable / #net-profitable opportunities; 0.0 if none.
      * n_acted   = number of acted rows.

    Profitability is NET of the round-trip ``cost`` (``make_meta_labels(r, cost)``): a trade
    with 0<R<cost is a net loss, so the north-star P(profit | acted) is P(NET profit | acted),
    not the cost-blind win rate. This is the number that becomes the live conviction + floor.
    """
    labels = make_meta_labels(r, cost)
    acted = proba >= tau
    n_acted = int(acted.sum())
    n_profit = int(labels.sum())
    acted_profit = int((acted & (labels == 1)).sum())
    precision = (acted_profit / n_acted) if n_acted > 0 else 0.0
    recall = (acted_profit / n_profit) if n_profit > 0 else 0.0
    return float(precision), float(recall), n_acted


def forward_holdout_precision(
    X: pd.DataFrame,
    y: np.ndarray,
    ts: np.ndarray,
    params: dict[str, object],
    cost: float,
    seed: int,
) -> tuple[float, float, int]:
    """Strict TIME-ORDERED forward-holdout precision/expectancy for the selected config.

    Unlike CPCV (which shuffles folds across the whole timeline and can inflate reported
    precision via regime/bull bias), this fits ONLY on the earliest 70% of the timeline and
    judges on the latest 30% — a "does the edge hold going FORWARD" durability check that
    separates a real edge from regime-fitting. It is REPORTED, never gating.

    Algorithm (deterministic given ``seed``; the caller passes rows already in event-time
    order, and we sort by ``ts`` defensively without shuffling):

      1. Order rows by ``ts`` (stable). Split at 70%: TRAIN = earliest 70%, HOLDOUT = latest
         30%.
      2. Fit one GBM on TRAIN features->labels using the SELECTED config's params (the
         trial's ``_features`` subset is honored; LightGBM-only params are passed through).
         Predict profitability probabilities on HOLDOUT.
      3. Select the acting threshold tau* on TRAIN (``select_act_threshold`` on the TRAIN
         proba/returns), then MEASURE on HOLDOUT with ``precision_recall_at``:
           * ``precision`` = P(profit | acted) on the forward holdout at tau*,
           * ``expectancy`` = mean cost-aware R over acted holdout rows,
           * ``n`` = acted holdout count.

    Degenerate cases (empty split, empty acted set) return ``(0.0, 0.0, 0)`` — never crash.
    """
    n = int(len(X))
    if n == 0:
        return 0.0, 0.0, 0

    # Defensive stable sort by event time; the dataset is already time-ordered so this is a
    # no-op there, but it guarantees the forward split is genuinely chronological.
    order = np.argsort(ts, kind="stable")
    X_ord = X.iloc[order].reset_index(drop=True)
    y_ord = np.asarray(y, dtype=np.float64)[order]

    cut = int(np.floor(0.70 * n))
    if cut <= 0 or cut >= n:
        # No usable train or no usable holdout: degenerate, report zeros.
        return 0.0, 0.0, 0

    model_params = {k: v for k, v in params.items() if not k.startswith("_")}
    features = params.get("_features")
    Xf = X_ord[features] if isinstance(features, list) else X_ord

    X_train, X_hold = Xf.iloc[:cut], Xf.iloc[cut:]
    y_train, y_hold = y_ord[:cut], y_ord[cut:]

    # Fit the primary model on TRAIN only, predict on HOLDOUT.
    model = fit_gbm(X_train, y_train, seed=seed, params=model_params)
    proba_train = model.predict_proba(X_train)
    proba_hold = model.predict_proba(X_hold)

    # Select tau* on TRAIN (no HOLDOUT peeking), then measure on HOLDOUT.
    tau = select_act_threshold(proba_train, y_train, cost)
    precision, _recall, n_acted = precision_recall_at(proba_hold, y_hold, tau, cost)
    if n_acted == 0:
        return 0.0, 0.0, 0

    acted = proba_hold >= tau
    expectancy = float(np.mean(mx.cost_aware_returns(y_hold[acted], cost)))
    return float(precision), expectancy, int(n_acted)


@app.post("/validate", response_model=ValidateResult)
def validate(req: ValidateRequest) -> ValidateResult:
    """Run CPCV over a trial grid, compute DSR/PBO, and apply the promotion gate.

    Each of ``n_trials`` hyperparameter configs is evaluated with purged+embargoed CPCV to
    produce a full-timeline OOS return series. PBO (CSCV) operates on the (T x n_trials)
    matrix: it asks whether the IS-best config stays good OOS. The headline OOS metrics
    (expectancy, DSR, profit factor, ...) report the IS-selected best config — exactly the
    config a naive search would promote — so the gate judges what would actually ship.

    On top of the selected config we add a precision-optimized meta-labeling acting layer:
    an acting threshold tau* is chosen on the IS half (maximizing IS precision under a
    profitability + coverage + recall constraint set, confirmed on a second IS sub-fold for
    robustness — see :func:`select_act_threshold`), and the gate's OOS metrics plus the
    reported out-of-sample precision/recall are measured on the tau*-acted OOS rows. Win rate
    is never computed; precision is the north-star OOS metric.

    Alongside the CPCV precision we also report a STRICTER, time-ordered forward-holdout
    durability metric (``precision_forward``, ``expectancy_forward``, ``n_forward``): the
    selected config is refit on the earliest 70% of the timeline and judged on the latest 30%
    (see :func:`forward_holdout_precision`). Because CPCV shuffles folds across the whole
    timeline, its precision can be inflated by regime/bull bias; the forward holdout asks
    whether the edge holds going FORWARD and so separates a real edge from regime-fitting. It
    is REPORTED, never gating.
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

    # Precision/threshold layer on the SELECTED config. Recover its raw per-row OOS
    # probabilities across the full timeline (np.nan where a row never appeared in a test
    # fold) so we can choose an acting threshold tau* instead of the hardcoded 0.5.
    proba = _oos_proba_for_trial(X, y_np, splits, grid[best], seed=1000 + best)
    idx = np.arange(t)
    finite = ~np.isnan(proba)
    is_mask = (idx < mid) & finite
    oos_mask = (idx >= mid) & finite

    # Select tau* on the IS half only (no peeking at OOS), maximizing IS precision subject to
    # profitability + coverage + recall constraints (two-sub-fold confirmed for robustness);
    # falls back to 0.5 if nothing qualifies.
    act_threshold = select_act_threshold(
        proba[is_mask], y_np[is_mask], DEFAULT_COST_PER_TRADE_R
    )

    # Measure precision/recall on the OOS half at tau*. Precision is NET of cost (P(net profit
    # | acted)) — a sub-cost winner (0<R<cost) is a net loss, not a precision win.
    precision_oos, recall_oos, n_acted_oos = precision_recall_at(
        proba[oos_mask], y_np[oos_mask], act_threshold, DEFAULT_COST_PER_TRADE_R
    )

    # The whole gate is precision-optimized: feed the tau*-acted OOS realized returns as the
    # `realized` series. `row_mask` is the full-timeline boolean selecting exactly the rows in
    # `realized`, so regime attribution aligns row-for-row. Fall back to a non-empty series so
    # a viable dataset never 422s.
    row_mask = oos_mask & (np.nan_to_num(proba, nan=-np.inf) >= act_threshold)
    realized = y_np[row_mask]
    if realized.size == 0:
        # tau* acted on nothing OOS: fall back to the legacy 0.5-threshold traded series so
        # the endpoint stays informative, then the whole OOS half if that is empty too.
        selected_oos = selected[mid:]
        traded = selected_oos[selected_oos != 0.0]
        if traded.size > 0:
            row_mask = np.zeros(t, dtype=bool)
            row_mask[mid:] = selected_oos != 0.0
        else:
            row_mask = np.zeros(t, dtype=bool)
            row_mask[mid:] = True
        realized = y_np[row_mask]
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

    # Regime attribution on exactly the tau*-acted OOS rows feeding the gate.
    contrib, n_pos = _regime_contrib(df.loc[row_mask].reset_index(drop=True), realized)

    gate = gate_evaluate(
        dsr=dsr,
        pbo=pbo,
        oos_expectancy_cost_aware=float(np.mean(cost_aware)),
        n_regimes_positive=n_pos,
    )

    # STRICT forward-held-out durability (reported, NOT gating): refit the SELECTED config on
    # the earliest 70% of the (time-ordered) timeline and judge on the latest 30%. Never
    # crashes; degenerate/empty holdout acted sets report zeros.
    try:
        precision_forward, expectancy_forward, n_forward = forward_holdout_precision(
            X,
            y_np,
            df[TS_COL].to_numpy(),
            grid[best],
            DEFAULT_COST_PER_TRADE_R,
            seed=1000 + best,
        )
    except (ValueError, KeyError):
        precision_forward, expectancy_forward, n_forward = 0.0, 0.0, 0

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
        precision_oos=float(precision_oos),
        recall_oos=float(recall_oos),
        act_threshold=float(act_threshold),
        n_acted_oos=int(n_acted_oos),
        precision_forward=float(precision_forward),
        expectancy_forward=float(expectancy_forward),
        n_forward=int(n_forward),
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
