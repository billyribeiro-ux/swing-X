"""Path-averaged CPCV probabilities (combinatorial variance reduction).

Combinatorial CPCV holds each row out in MULTIPLE test folds. The single-path
``_oos_proba_for_trial`` keeps only the last fold's write (order-dependent), collapsing the
design to one path. ``_oos_proba_path_averaged`` averages each row's probability over ALL
folds that hold it out, recovering the variance reduction. These tests pin that the averaged
value is exactly the per-fold mean, that it differs from the single-path value, and that it is
order-invariant where the single path is not.
"""

from __future__ import annotations

import numpy as np

from se_ml.cv.cpcv import cpcv_splits
from se_ml.io_arrow import split_features_labels
from se_ml.models.gbm import fit_gbm
from se_ml.server import _oos_proba_for_trial, _oos_proba_path_averaged

from .fixtures import genuine_edge_dataset

_PARAMS: dict[str, object] = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 50,
    "_features": [
        "momentum__signal",
        "trend__slope",
        "volatility__atr_norm",
        "momentum__noise",
    ],
}


def _splits(df):
    return cpcv_splits(
        event_ts=df["ts"].to_numpy(),
        t1=df["t1"].to_numpy(),
        n_groups=6,
        k_test_groups=2,
        embargo_bars=3,
        purge=True,
    )


def test_path_averaged_equals_mean_of_per_fold_probabilities() -> None:
    df = genuine_edge_dataset(n=300, seed=0)
    X, y = split_features_labels(df)
    y_np = y.to_numpy()
    splits = _splits(df)
    seed = 1234

    pa = _oos_proba_path_averaged(X, y_np, splits, _PARAMS, seed=seed)

    # Manual per-fold accumulation (unweighted; ts/t1 default None here).
    model_params = {k: v for k, v in _PARAMS.items() if not k.startswith("_")}
    feats = _PARAMS["_features"]
    assert isinstance(feats, list)
    Xf = X[feats]
    n = len(X)
    psum = np.zeros(n)
    pcount = np.zeros(n)
    for sp in splits:
        if sp.train_idx.size == 0 or sp.test_idx.size == 0:
            continue
        m = fit_gbm(Xf.iloc[sp.train_idx], y_np[sp.train_idx], seed=seed, params=model_params)
        psum[sp.test_idx] += m.predict_proba(Xf.iloc[sp.test_idx])
        pcount[sp.test_idx] += 1.0
    expected = np.where(pcount > 0, psum / np.where(pcount > 0, pcount, 1.0), np.nan)

    # The combinatorial design tests some rows in more than one fold (the whole point).
    assert pcount.max() > 1
    assert np.allclose(pa, expected, equal_nan=True)


def test_path_averaged_differs_from_single_path_last_write() -> None:
    df = genuine_edge_dataset(n=300, seed=0)
    X, y = split_features_labels(df)
    y_np = y.to_numpy()
    splits = _splits(df)

    pa = _oos_proba_path_averaged(X, y_np, splits, _PARAMS, seed=7)
    sp = _oos_proba_for_trial(X, y_np, splits, _PARAMS, seed=7)

    finite = ~np.isnan(pa) & ~np.isnan(sp)
    assert finite.any()
    # Averaging over folds moves the value off the single last-write path.
    assert not np.allclose(pa[finite], sp[finite])


def test_path_averaged_order_invariant_single_path_is_not() -> None:
    df = genuine_edge_dataset(n=300, seed=0)
    X, y = split_features_labels(df)
    y_np = y.to_numpy()
    splits = _splits(df)
    rev = list(reversed(splits))

    pa1 = _oos_proba_path_averaged(X, y_np, splits, _PARAMS, seed=3)
    pa2 = _oos_proba_path_averaged(X, y_np, rev, _PARAMS, seed=3)
    sp1 = _oos_proba_for_trial(X, y_np, splits, _PARAMS, seed=3)
    sp2 = _oos_proba_for_trial(X, y_np, rev, _PARAMS, seed=3)

    # Path-averaged proba is invariant to fold order (cross-fold-order stability improves).
    assert np.allclose(pa1, pa2, equal_nan=True)
    # Single-path (last-write) proba depends on fold order for multi-fold rows.
    finite = ~np.isnan(sp1) & ~np.isnan(sp2)
    assert not np.allclose(sp1[finite], sp2[finite])
