"""Tests for PBO via CSCV: low PBO for genuine edge, high PBO for overfit/random."""

from __future__ import annotations

from se_ml.stats.pbo import (
    cscv_logits,
    n_combinations,
    probability_of_backtest_overfit,
)

from .fixtures import perf_matrix_genuine, perf_matrix_overfit


def test_genuine_edge_has_low_pbo():
    M = perf_matrix_genuine(t=1200, n_strategies=20, seed=11)
    pbo = probability_of_backtest_overfit(M, n_splits=10)
    assert pbo < 0.2, f"genuine edge should yield low PBO, got {pbo}"


def test_overfit_random_has_high_pbo():
    # Best-of-many pure-noise selection: the IS winner is essentially random OOS, so PBO
    # is dramatically elevated relative to a genuine edge (which is ~0). Finite samples
    # keep it below the asymptotic 0.5, but well into the "overfit" regime.
    M = perf_matrix_overfit(t=1200, n_strategies=300, seed=12)
    pbo = probability_of_backtest_overfit(M, n_splits=12)
    assert pbo >= 0.35, f"best-of-many-noise should yield high PBO, got {pbo}"


def test_pbo_ordering_genuine_below_overfit():
    genuine = probability_of_backtest_overfit(
        perf_matrix_genuine(t=1200, n_strategies=20, seed=1), n_splits=12
    )
    overfit = probability_of_backtest_overfit(
        perf_matrix_overfit(t=1200, n_strategies=300, seed=2), n_splits=12
    )
    # The scientifically meaningful claim: overfit PBO is far above genuine PBO.
    assert overfit > genuine + 0.3


def test_logits_length_matches_combinations():
    M = perf_matrix_overfit(t=400, n_strategies=10, seed=3)
    logits = cscv_logits(M, n_splits=8)
    assert logits.size == n_combinations(8)  # C(8, 4) = 70


def test_pbo_bounds():
    M = perf_matrix_overfit(t=400, n_strategies=50, seed=4)
    pbo = probability_of_backtest_overfit(M, n_splits=8)
    assert 0.0 <= pbo <= 1.0


def test_rejects_odd_splits():
    M = perf_matrix_overfit(t=400, n_strategies=10, seed=5)
    try:
        probability_of_backtest_overfit(M, n_splits=7)
    except ValueError as e:
        assert "even" in str(e)
    else:
        raise AssertionError("expected ValueError for odd n_splits")
