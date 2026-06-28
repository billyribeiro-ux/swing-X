"""Tests for isotonic + Platt calibration, reliability curve, and Brier score."""

from __future__ import annotations

import numpy as np

from se_ml.calibration.calibrate import brier_score, calibrate, reliability_curve


def test_brier_score_known_value():
    prob = np.array([0.0, 1.0, 0.5, 0.5])
    outcome = np.array([0, 1, 1, 0])
    # errors: 0, 0, 0.25, 0.25 -> mean 0.125
    assert np.isclose(brier_score(prob, outcome), 0.125)


def test_calibration_improves_brier_on_miscalibrated_scores():
    rng = np.random.default_rng(0)
    n = 4000
    # True probability is the raw score, but the model reports an over-confident
    # transform; calibration should recover and lower the Brier score vs raw.
    true_p = rng.uniform(0, 1, size=n)
    outcomes = (rng.uniform(0, 1, size=n) < true_p).astype(int)
    raw = np.clip(true_p**2, 0, 1)  # systematically miscalibrated (too low)

    raw_brier = brier_score(raw, outcomes)
    result = calibrate(raw, outcomes, n_bins=10)
    assert result.brier <= raw_brier + 1e-9
    assert result.method in ("isotonic", "platt")


def test_reliability_curve_monotone_for_calibrated_data():
    rng = np.random.default_rng(1)
    n = 5000
    p = rng.uniform(0, 1, size=n)
    outcomes = (rng.uniform(0, 1, size=n) < p).astype(int)
    pts = reliability_curve(p, outcomes, n_bins=10)
    # For well-calibrated data, realized should track predicted across bins.
    preds = np.array([pt.predicted for pt in pts])
    reals = np.array([pt.realized for pt in pts])
    assert np.all(np.abs(preds - reals) < 0.08)
    assert sum(pt.count for pt in pts) == n


def test_calibration_map_is_applicable():
    rng = np.random.default_rng(2)
    n = 2000
    p = rng.uniform(0, 1, size=n)
    outcomes = (rng.uniform(0, 1, size=n) < p).astype(int)
    result = calibrate(p, outcomes)
    mapped = result.apply(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert mapped.shape == (5,)
    assert np.all((mapped >= 0.0) & (mapped <= 1.0))


def test_degenerate_single_class_target():
    scores = np.linspace(0, 1, 100)
    outcomes = np.ones(100, dtype=int)  # all positive
    result = calibrate(scores, outcomes)
    # Flat map at the base rate (1.0); Brier should be small.
    assert result.brier < 0.05
