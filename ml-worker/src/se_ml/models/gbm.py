"""LightGBM fit/predict wrapper: deterministic seeding + artifact save/load.

Wraps a LightGBM classifier behind a tiny, picklable surface. The model predicts the
probability that a trade is profitable (the meta-label / direction-quality target). We
binarise the R-unit label (``label > 0``) for the classifier and keep the artifact
self-describing (feature names + params) so calibration and importance can reload it
without the original training frame.

Determinism: a single ``seed`` threads into ``random_state``, ``bagging_seed`` and
``feature_fraction_seed``, and we force single-threaded deterministic histogram building
so golden tests are stable across machines.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMClassifier

from ..config import DEFAULT_SEED

_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "n_jobs": 1,             # deterministic
    "verbosity": -1,
    "deterministic": True,
    "force_row_wise": True,
}


@dataclass
class GbmModel:
    """A fitted LightGBM classifier plus the metadata needed to reuse it."""

    booster: LGBMClassifier
    feature_names: list[str]
    seed: int
    params: dict[str, Any] = field(default_factory=dict)
    model_id: str = ""

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Probability of the positive class (profitable trade) per row."""
        Xa = X[self.feature_names]
        proba = self.booster.predict_proba(Xa)
        # LGBMClassifier returns (n, 2); take the positive-class column robustly.
        classes = list(self.booster.classes_)
        pos = classes.index(1) if 1 in classes else proba.shape[1] - 1
        return np.asarray(proba[:, pos], dtype=np.float64)

    def save(self, artifact_dir: Path) -> Path:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"{self.model_id}.pkl"
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: Path) -> GbmModel:
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)  # noqa: S301 — trusted local artifact written by this service
        if not isinstance(obj, GbmModel):
            raise TypeError(f"artifact at {path} is not a GbmModel")
        return obj


def _compute_model_id(feature_names: list[str], params: dict[str, Any], seed: int) -> str:
    payload = json.dumps(
        {"features": feature_names, "params": params, "seed": seed},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"gbm-{digest}"


def fit_gbm(
    X: pd.DataFrame,
    y: npt.ArrayLike,
    *,
    seed: int = DEFAULT_SEED,
    params: dict[str, Any] | None = None,
    sample_weight: npt.ArrayLike | None = None,
) -> GbmModel:
    """Fit a deterministic LightGBM classifier on (X, binarised y).

    ``y`` is the R-unit label; the classifier target is ``(y > 0)``.

    ``sample_weight`` (optional) is a per-row weight vector passed straight through to
    LightGBM's ``fit`` — used to carry Lopez de Prado average-uniqueness weights so
    overlapping (concurrent) labels do not over-count. ``None`` keeps the unweighted (iid)
    fit unchanged. Determinism (seed threading, single-threaded histograms) is unaffected.
    """
    feature_names = list(X.columns)
    merged = {**_DEFAULT_PARAMS, **(params or {})}
    merged.update(
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        data_random_seed=seed,
    )
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    y_bin = (y_arr > 0).astype(int)

    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).ravel()
    clf = LGBMClassifier(**merged)
    clf.fit(X[feature_names], y_bin, sample_weight=sw)

    model_id = _compute_model_id(feature_names, merged, seed)
    return GbmModel(
        booster=clf,
        feature_names=feature_names,
        seed=seed,
        params=merged,
        model_id=model_id,
    )
