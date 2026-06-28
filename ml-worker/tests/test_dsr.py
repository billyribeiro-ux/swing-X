"""Golden tests for the Deflated Sharpe Ratio on fixed inputs."""

from __future__ import annotations

import numpy as np
from scipy import stats

from se_ml.stats.dsr import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)


def test_psr_matches_hand_computation():
    # Hand-computed PSR for a normal series: skew=0, kurt=3, benchmark=0.
    observed_sr = 0.10
    n = 250
    skew, kurt = 0.0, 3.0
    denom = 1.0 - skew * observed_sr + (kurt - 1.0) / 4.0 * observed_sr**2
    z = (observed_sr - 0.0) * np.sqrt(n - 1.0) / np.sqrt(denom)
    expected = float(stats.norm.cdf(z))
    got = probabilistic_sharpe_ratio(observed_sr, 0.0, n, skew, kurt)
    assert np.isclose(got, expected, atol=1e-12)
    # sanity: with SR>0 and 250 obs, PSR should be well above 0.5.
    assert got > 0.9


def test_expected_max_sharpe_increases_with_trials():
    var = 0.04  # sd of per-trial SR = 0.2
    sr0_1 = expected_max_sharpe(1, var)
    sr0_10 = expected_max_sharpe(10, var)
    sr0_100 = expected_max_sharpe(100, var)
    assert sr0_1 == 0.0  # single trial -> no deflation
    assert sr0_10 < sr0_100  # more trials -> higher bar
    assert sr0_10 > 0.0


def test_expected_max_sharpe_golden_value():
    # Golden: E[max] for N=100, with variance 1.0 (sd=1) so SR0 == E[max].
    # E[max] = (1-gamma)*Z^{-1}(1-1/100) + gamma*Z^{-1}(1-1/(100 e))
    gamma = 0.5772156649015329
    z1 = stats.norm.ppf(1.0 - 1.0 / 100.0)
    z2 = stats.norm.ppf(1.0 - 1.0 / (100.0 * np.e))
    expected = (1.0 - gamma) * z1 + gamma * z2
    got = expected_max_sharpe(100, 1.0)
    assert np.isclose(got, expected, atol=1e-12)
    # Numerically pinned golden constant (regression guard).
    assert np.isclose(got, 2.5306, atol=1e-3)


def test_dsr_deflates_with_more_trials():
    rng = np.random.default_rng(123)
    # A modestly positive normal return series.
    returns = rng.normal(loc=0.05, scale=1.0, size=500)
    dsr_few = deflated_sharpe_ratio(returns, n_trials=1)
    dsr_many = deflated_sharpe_ratio(returns, n_trials=1000)
    assert dsr_few > dsr_many  # more trials -> lower (deflated) probability
    assert 0.0 <= dsr_many <= 1.0
    assert 0.0 <= dsr_few <= 1.0


def test_dsr_positive_for_strong_edge_single_trial():
    rng = np.random.default_rng(7)
    returns = rng.normal(loc=0.3, scale=1.0, size=1000)  # SR ~0.3, T=1000
    dsr = deflated_sharpe_ratio(returns, n_trials=1)
    assert dsr > 0.99  # overwhelming evidence of a positive Sharpe


def test_dsr_low_for_zero_edge_many_trials():
    rng = np.random.default_rng(9)
    returns = rng.normal(loc=0.0, scale=1.0, size=300)
    dsr = deflated_sharpe_ratio(returns, n_trials=500)
    assert dsr < 0.5  # no real edge, many trials -> deflated below a coin flip


def test_dsr_constant_returns_edge_cases():
    assert deflated_sharpe_ratio(np.ones(50) * 0.1, n_trials=1) == 1.0  # positive constant
    assert deflated_sharpe_ratio(-np.ones(50) * 0.1, n_trials=1) == 0.0  # negative constant
    assert deflated_sharpe_ratio([0.1], n_trials=1) == 0.5  # too short
