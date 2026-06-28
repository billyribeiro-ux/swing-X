"""Feature importance: SHAP values + permutation importance, per-feature and per-layer.

Two complementary views:

  * **SHAP** (TreeExplainer): mean absolute SHAP value per feature — a model-internal,
    additive attribution of the prediction.
  * **Permutation**: drop in a score when a feature's column is shuffled — a
    model-agnostic measure of how much the model *relies* on the feature.

Both are aggregated per feature and rolled up per "layer" (the ``layer__feature`` naming
convention, see :mod:`se_ml.io_arrow`), so the engine can attribute edge to whole feature
families (momentum / trend / volatility / regime / ...). Scores are normalised to sum to
1 within each view, making per-feature and per-layer numbers comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
from sklearn.metrics import accuracy_score

from ..io_arrow import layer_of
from ..models.gbm import GbmModel


@dataclass
class ImportanceScores:
    shap: float
    permutation: float


def _normalise(d: dict[str, float]) -> dict[str, float]:
    total = sum(abs(v) for v in d.values())
    if total <= 0:
        return dict.fromkeys(d, 0.0)
    return {k: abs(v) / total for k, v in d.items()}


def shap_importance(model: GbmModel, X: pd.DataFrame) -> dict[str, float]:
    """Mean absolute SHAP value per feature (raw, un-normalised)."""
    Xa = X[model.feature_names]
    explainer = shap.TreeExplainer(model.booster)
    values = explainer.shap_values(Xa)
    # Binary LightGBM via shap may return a list [neg, pos] or a single array; normalise.
    arr = np.asarray(values[-1]) if isinstance(values, list) else np.asarray(values)
    if arr.ndim == 3:  # (n, features, classes)
        arr = arr[:, :, -1]
    mean_abs = np.abs(arr).mean(axis=0)
    return {f: float(v) for f, v in zip(model.feature_names, mean_abs, strict=True)}


def permutation_importance(
    model: GbmModel,
    X: pd.DataFrame,
    y: npt.ArrayLike,
    *,
    n_repeats: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Permutation importance: mean accuracy drop when each feature is shuffled.

    Uses the model's own binarised target convention (``y > 0``). Negative drops are
    clipped to 0 (a feature the model does not rely on).
    """
    rng = np.random.default_rng(seed)
    Xa = X[model.feature_names].copy().reset_index(drop=True)
    y_bin = (np.asarray(y, dtype=np.float64).ravel() > 0).astype(int)

    baseline_pred = (model.predict_proba(Xa) >= 0.5).astype(int)
    baseline = accuracy_score(y_bin, baseline_pred)

    out: dict[str, float] = {}
    for feat in model.feature_names:
        drops = []
        original = Xa[feat].to_numpy().copy()
        for _ in range(n_repeats):
            shuffled = original.copy()
            rng.shuffle(shuffled)
            Xa[feat] = shuffled
            pred = (model.predict_proba(Xa) >= 0.5).astype(int)
            drops.append(baseline - accuracy_score(y_bin, pred))
        Xa[feat] = original
        out[feat] = float(max(0.0, np.mean(drops)))
    return out


def _rollup_layers(per_feature: dict[str, float]) -> dict[str, float]:
    layers: dict[str, float] = {}
    for feat, val in per_feature.items():
        layers[layer_of(feat)] = layers.get(layer_of(feat), 0.0) + val
    return layers


def compute_importance(
    model: GbmModel,
    X: pd.DataFrame,
    y: npt.ArrayLike,
    *,
    n_repeats: int = 5,
    seed: int = 42,
) -> tuple[dict[str, ImportanceScores], dict[str, ImportanceScores]]:
    """Compute normalised per-feature and per-layer SHAP + permutation importance."""
    shap_raw = shap_importance(model, X)
    perm_raw = permutation_importance(model, X, y, n_repeats=n_repeats, seed=seed)

    shap_n = _normalise(shap_raw)
    perm_n = _normalise(perm_raw)

    per_feature = {
        f: ImportanceScores(shap=shap_n.get(f, 0.0), permutation=perm_n.get(f, 0.0))
        for f in model.feature_names
    }

    shap_layer = _normalise(_rollup_layers(shap_raw))
    perm_layer = _normalise(_rollup_layers(perm_raw))
    layers = set(shap_layer) | set(perm_layer)
    per_layer = {
        layer_name: ImportanceScores(
            shap=shap_layer.get(layer_name, 0.0),
            permutation=perm_layer.get(layer_name, 0.0),
        )
        for layer_name in sorted(layers)
    }
    return per_feature, per_layer
