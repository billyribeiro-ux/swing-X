"""Probability calibration: isotonic regression and Platt scaling.

A model's raw scores rarely match empirical hit rates. We calibrate them and report:

  * the **calibration map** — a monotone function from raw score to calibrated
    probability, returned as ``(x, y)`` knot points so the Rust side can interpolate;
  * the **reliability curve** — binned (predicted, realized, count) triples to plot;
  * the **Brier score** — mean squared error of the calibrated probabilities, the
    headline scalar for calibration quality (lower is better).

Method selection: we fit both isotonic and Platt (sigmoid) calibrators and keep whichever
has the lower Brier score on the provided data, recording which one won.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

FloatArray = npt.NDArray[np.float64]


@dataclass
class ReliabilityPoint:
    predicted: float
    realized: float
    count: int


@dataclass
class CalibrationResult:
    method: str
    map_x: list[float]              # raw-score knots
    map_y: list[float]              # calibrated-probability knots
    reliability: list[ReliabilityPoint]
    brier: float

    def apply(self, scores: npt.ArrayLike) -> FloatArray:
        """Apply the stored piecewise-linear calibration map to new raw scores."""
        s = np.asarray(scores, dtype=np.float64).ravel()
        return np.interp(s, self.map_x, self.map_y)


def brier_score(prob: npt.ArrayLike, outcome: npt.ArrayLike) -> float:
    """Mean squared error between predicted probabilities and binary outcomes."""
    p = np.asarray(prob, dtype=np.float64).ravel()
    o = np.asarray(outcome, dtype=np.float64).ravel()
    if p.size == 0:
        return 0.0
    return float(np.mean((p - o) ** 2))


def reliability_curve(
    prob: npt.ArrayLike, outcome: npt.ArrayLike, n_bins: int = 10
) -> list[ReliabilityPoint]:
    """Bin predictions into ``n_bins`` and report (mean predicted, mean realized, count)."""
    p = np.asarray(prob, dtype=np.float64).ravel()
    o = np.asarray(outcome, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    points: list[ReliabilityPoint] = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p >= lo) & (p < hi) if b < n_bins - 1 else (p >= lo) & (p <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        points.append(
            ReliabilityPoint(
                predicted=float(p[mask].mean()),
                realized=float(o[mask].mean()),
                count=cnt,
            )
        )
    return points


def _isotonic_map(scores: FloatArray, outcomes: FloatArray) -> tuple[FloatArray, FloatArray]:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(scores, outcomes)
    xs = np.unique(scores)
    ys = iso.predict(xs)
    return xs, np.asarray(ys, dtype=np.float64)


def _platt_map(scores: FloatArray, outcomes: FloatArray) -> tuple[FloatArray, FloatArray]:
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(scores.reshape(-1, 1), outcomes.astype(int))
    xs = np.linspace(float(scores.min()), float(scores.max()), 50)
    ys = lr.predict_proba(xs.reshape(-1, 1))[:, 1]
    return xs, np.asarray(ys, dtype=np.float64)


def calibrate(
    scores: npt.ArrayLike,
    outcomes: npt.ArrayLike,
    n_bins: int = 10,
) -> CalibrationResult:
    """Fit isotonic + Platt, keep the lower-Brier one, and report reliability + Brier.

    ``scores`` are raw model probabilities/scores in [0, 1]; ``outcomes`` are binary.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    o = np.asarray(outcomes, dtype=np.float64).ravel()
    if s.size != o.size:
        raise ValueError("scores and outcomes must have equal length")
    if s.size == 0:
        raise ValueError("empty calibration input")

    candidates: dict[str, tuple[FloatArray, FloatArray]] = {}
    # Isotonic needs variation in the target to be meaningful.
    if np.unique(o).size > 1:
        candidates["isotonic"] = _isotonic_map(s, o)
        candidates["platt"] = _platt_map(s, o)
    else:
        # Degenerate target: fall back to a flat map at the base rate.
        base = float(o.mean())
        xs = np.unique(s)
        candidates["isotonic"] = (xs, np.full_like(xs, base))

    best_method = ""
    best_brier = np.inf
    best_xy: tuple[FloatArray, FloatArray] = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    for method, (xs, ys) in candidates.items():
        calibrated = np.interp(s, xs, ys)
        b = brier_score(calibrated, o)
        if b < best_brier:
            best_brier, best_method, best_xy = b, method, (xs, ys)

    xs, ys = best_xy
    calibrated_best = np.interp(s, xs, ys)
    return CalibrationResult(
        method=best_method,
        map_x=[float(v) for v in xs],
        map_y=[float(v) for v in ys],
        reliability=reliability_curve(calibrated_best, o, n_bins=n_bins),
        brier=float(best_brier),
    )
