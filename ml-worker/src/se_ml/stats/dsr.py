"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

The DSR corrects an observed Sharpe ratio for (a) the number of independent strategy
trials that were run to find it, (b) the non-normality (skew/kurtosis) of the returns,
and (c) the length of the track record. It returns a probability in (0, 1): the
probability that the *true* Sharpe ratio is greater than zero, given that the observed
SR is the maximum of ``n_trials`` trials.

References
----------
Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting, and Non-Normality." Journal of Portfolio
Management, 40(5), 94-107.

Formulas
--------
Expected maximum of ``N`` independent standard-normal trials (Eq. for SR*):

    E[max] ≈ (1 - γ) · Z⁻¹(1 - 1/N) + γ · Z⁻¹(1 - 1/(N·e))

where ``γ`` is the Euler-Mascheroni constant and ``Z⁻¹`` is the standard-normal inverse
CDF. The deflated SR threshold is ``SR0 = sqrt(var_sr_across_trials) · E[max]``.

The deflated probability is:

    DSR = Z( ( (SR_hat - SR0) · sqrt(T - 1) )
             / sqrt( 1 - skew·SR_hat + (kurt - 1)/4 · SR_hat² ) )

with ``SR_hat`` the observed (non-annualised, per-observation) Sharpe ratio, ``T`` the
number of observations, ``skew`` and ``kurt`` the skewness and (non-excess) kurtosis of
the returns.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import stats

# Euler-Mascheroni constant.
_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: int, variance_across_trials: float) -> float:
    """Expected maximum Sharpe ratio across ``n_trials`` independent trials.

    ``variance_across_trials`` is the variance of the Sharpe ratios obtained across the
    trials. Returns the deflation threshold ``SR0`` (in the same per-observation units as
    the observed SR).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if variance_across_trials < 0:
        raise ValueError("variance_across_trials must be >= 0")
    sd = float(np.sqrt(variance_across_trials))
    if n_trials == 1:
        # No selection bias to deflate against.
        return 0.0
    n = float(n_trials)
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    e_max = (1.0 - _GAMMA) * z1 + _GAMMA * z2
    return sd * float(e_max)


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > benchmark_sr).

    ``observed_sr`` and ``benchmark_sr`` are per-observation Sharpe ratios. ``kurtosis``
    is the non-excess kurtosis (3.0 for a normal distribution).
    """
    if n_obs < 2:
        return 0.5
    denom = 1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr**2
    # Guard against pathological moments producing a non-positive variance term.
    denom = max(denom, 1e-12)
    numer = (observed_sr - benchmark_sr) * np.sqrt(n_obs - 1.0)
    z = numer / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: npt.ArrayLike,
    n_trials: int,
    variance_across_trials: float | None = None,
) -> float:
    """Deflated Sharpe Ratio for a per-trade return series.

    Parameters
    ----------
    returns
        Per-trade (or per-period) return series.
    n_trials
        Number of independent strategy trials that produced the candidate. Higher
        ``n_trials`` raises the deflation bar.
    variance_across_trials
        Variance of Sharpe ratios across the trials. If ``None``, it is estimated from
        the asymptotic variance of the SR estimator under the observed sample
        (``(1 + 0.5·SR²)/T``), a standard fallback when the per-trial SRs are not tracked.

    Returns
    -------
    float
        Probability in (0, 1) that the true (deflated) Sharpe ratio exceeds zero.
    """
    arr = np.asarray(returns, dtype=np.float64).ravel()
    arr = arr[~np.isnan(arr)]
    t = arr.size
    if t < 2:
        return 0.5
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        return 1.0 if mean > 0 else 0.0
    sr_hat = mean / sd
    skew = float(stats.skew(arr, bias=False))
    kurt = float(stats.kurtosis(arr, fisher=False, bias=False))  # non-excess

    if variance_across_trials is None:
        # Asymptotic variance of the SR estimator (Lo, 2002), used as a proxy for the
        # cross-trial dispersion when only a single trial's returns are available.
        variance_across_trials = (1.0 + 0.5 * sr_hat**2) / t

    sr0 = expected_max_sharpe(n_trials, variance_across_trials)
    return probabilistic_sharpe_ratio(sr_hat, sr0, t, skew, kurt)
