"""Precision-optimized acting-threshold selection on the /validate path.

The promotion gate's north-star metric is OUT-OF-SAMPLE PRECISION (win rate is banned). The
``/validate`` route picks a meta-labeling acting threshold tau* on the in-sample half (no OOS
peeking) and reports precision/recall measured on the OOS half. These tests pin the pure,
deterministic helpers that implement that layer — no HTTP server required.

Key property: on a dataset where higher model-proba genuinely correlates with profit, acting
at the SELECTED tau* must lift (or at least maintain) precision versus acting at 0.5.
"""

from __future__ import annotations

import numpy as np

from se_ml.server import (
    DEFAULT_COST_PER_TRADE_R,
    precision_recall_at,
    select_act_threshold,
)


def _synth_signal_dataset(
    n: int = 2000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities that genuinely correlate with profit.

    ``proba`` is uniform in [0, 1]; the realized R is more likely positive (and larger) the
    higher the proba, so a higher acting threshold concentrates on profitable trades and lifts
    precision. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    proba = rng.uniform(0.0, 1.0, size=n)
    # Higher proba => higher chance of profit and bigger edge; noise keeps it non-trivial.
    win = rng.uniform(0.0, 1.0, size=n) < proba
    r = np.where(win, rng.uniform(0.1, 2.0, size=n), rng.uniform(-2.0, -0.1, size=n))
    return proba, r.astype(np.float64)


def _split_half(proba: np.ndarray, r: np.ndarray) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    mid = proba.size // 2
    return proba[:mid], r[:mid], proba[mid:], r[mid:]


def test_selected_tau_lifts_or_maintains_precision_oos() -> None:
    proba, r = _synth_signal_dataset(n=2000, seed=0)
    proba_is, r_is, proba_oos, r_oos = _split_half(proba, r)

    tau = select_act_threshold(proba_is, r_is, DEFAULT_COST_PER_TRADE_R)

    prec_tau, recall_tau, n_acted = precision_recall_at(proba_oos, r_oos, tau)
    prec_half, _, _ = precision_recall_at(proba_oos, r_oos, 0.5)

    # The chosen threshold must lift or maintain OOS precision versus acting at 0.5.
    assert prec_tau >= prec_half, (
        f"tau*={tau} precision {prec_tau} should be >= 0.5-threshold precision {prec_half}"
    )

    # Field-range invariants the contract guarantees.
    assert 0.0 <= prec_tau <= 1.0
    assert 0.0 <= recall_tau <= 1.0
    assert 0.0 <= tau <= 1.0
    assert n_acted >= 0


def test_pure_noise_returns_valid_fields_and_does_not_crash() -> None:
    rng = np.random.default_rng(7)
    n = 1000
    proba = rng.uniform(0.0, 1.0, size=n)
    r = rng.normal(0.0, 1.0, size=n)  # independent of proba: no real edge
    proba_is, r_is, proba_oos, r_oos = _split_half(proba, r)

    tau = select_act_threshold(proba_is, r_is, DEFAULT_COST_PER_TRADE_R)
    prec, recall, n_acted = precision_recall_at(proba_oos, r_oos, tau)

    assert 0.0 <= tau <= 1.0
    assert 0.0 <= prec <= 1.0
    assert 0.0 <= recall <= 1.0
    assert n_acted >= 0


def test_select_act_threshold_falls_back_to_half_when_no_candidate_qualifies() -> None:
    # Too few rows to satisfy the min-acted floor (max(8, ...)) at any threshold => fallback.
    proba = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    r = np.array([0.5, -0.5, 0.5], dtype=np.float64)
    assert select_act_threshold(proba, r, DEFAULT_COST_PER_TRADE_R) == 0.5


def test_select_act_threshold_empty_is_fallback() -> None:
    empty = np.array([], dtype=np.float64)
    assert select_act_threshold(empty, empty, DEFAULT_COST_PER_TRADE_R) == 0.5


def test_precision_recall_at_no_acted_is_zero_precision() -> None:
    proba = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    r = np.array([1.0, -1.0, 1.0], dtype=np.float64)
    prec, recall, n_acted = precision_recall_at(proba, r, tau=0.9)
    assert n_acted == 0
    assert prec == 0.0
    assert recall == 0.0


def test_precision_recall_at_known_counts() -> None:
    # proba >= 0.5 acts on rows {0.5, 0.7, 0.9} -> R {1.0, -1.0, 2.0}: 2 of 3 profitable.
    # Profitable opportunities overall: rows with R>0 = {1.0(0.5), 2.0(0.9), 0.3(0.2)} = 3.
    proba = np.array([0.5, 0.7, 0.9, 0.2], dtype=np.float64)
    r = np.array([1.0, -1.0, 2.0, 0.3], dtype=np.float64)
    prec, recall, n_acted = precision_recall_at(proba, r, tau=0.5)
    assert n_acted == 3
    assert prec == 2.0 / 3.0
    assert recall == 2.0 / 3.0  # captured 2 of the 3 profitable opportunities


def test_select_act_threshold_is_deterministic() -> None:
    proba, r = _synth_signal_dataset(n=1500, seed=3)
    proba_is, r_is, _, _ = _split_half(proba, r)
    a = select_act_threshold(proba_is, r_is, DEFAULT_COST_PER_TRADE_R)
    b = select_act_threshold(proba_is, r_is, DEFAULT_COST_PER_TRADE_R)
    assert a == b
