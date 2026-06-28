"""Probability of Backtest Overfit (PBO) via Combinatorially-Symmetric CV (CSCV).

Implements the CSCV procedure of Bailey, Borwein, Lopez de Prado & Zhu (2017),
"The Probability of Backtest Overfitting." Given a performance matrix ``M`` of shape
(T observations, N strategies/trials), CSCV:

  1. Splits the T rows into ``S`` disjoint, equally sized contiguous sub-matrices.
  2. Forms every combination C(S, S/2) that assigns half the sub-matrices to an
     in-sample (IS) set J and the complementary half to an out-of-sample (OOS) set J̄.
  3. For each split: rank strategies by IS performance, pick the IS-best strategy n*,
     then find its *relative rank* among the OOS performances. Map that rank to a
     logit ``λ = ln( ω / (1 - ω) )`` where ``ω`` is the relative rank in (0, 1).
  4. PBO = fraction of splits whose IS-best strategy lands below the OOS median, i.e.
     ``P(λ <= 0)`` — the probability that the strategy selected as best in-sample is in
     fact below-median out-of-sample.

A genuine, persistent edge -> low PBO (the IS-best stays good OOS). A best-of-many-noise
selection -> PBO near 0.5+ (IS-best is essentially random OOS).
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _performance(sub: FloatArray) -> FloatArray:
    """Per-strategy performance on a stacked sub-matrix block.

    We use the Sharpe ratio across the block's rows (mean / std). This is the standard
    CSCV performance functional; it is scale-free and rewards consistency, not just total
    return. Columns with zero variance score 0.
    """
    mean = sub.mean(axis=0)
    sd = sub.std(axis=0, ddof=1) if sub.shape[0] > 1 else np.ones(sub.shape[1])
    out = np.zeros_like(mean)
    nz = sd > 0
    out[nz] = mean[nz] / sd[nz]
    return out


def cscv_logits(perf_matrix: npt.ArrayLike, n_splits: int = 16) -> FloatArray:
    """Return the logit ``λ`` for every CSCV combination.

    Parameters
    ----------
    perf_matrix
        Matrix of shape (T, N): per-observation performance contributions for each of N
        strategies/trials.
    n_splits
        Number of disjoint sub-matrices ``S`` (must be even). The number of combinations
        is ``C(S, S/2)``.
    """
    M = np.asarray(perf_matrix, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError("perf_matrix must be 2-D (T observations x N strategies)")
    t, n = M.shape
    if n < 2:
        raise ValueError("need at least 2 strategies/trials to assess overfit")
    if n_splits % 2 != 0:
        raise ValueError("n_splits (S) must be even")
    if n_splits > t:
        raise ValueError("n_splits cannot exceed number of observations")

    # Partition rows into S contiguous, equally sized blocks (drop the remainder rows so
    # every block has identical length — required for the symmetric recombination).
    block_size = t // n_splits
    usable = block_size * n_splits
    blocks = [M[i * block_size : (i + 1) * block_size] for i in range(n_splits)]
    _ = usable  # documentation aid; trailing rows intentionally unused

    all_idx = set(range(n_splits))
    half = n_splits // 2
    logits: list[float] = []

    for is_combo in combinations(range(n_splits), half):
        is_set = list(is_combo)
        oos_set = list(all_idx - set(is_combo))

        is_block = np.vstack([blocks[i] for i in is_set])
        oos_block = np.vstack([blocks[i] for i in oos_set])

        is_perf = _performance(is_block)
        oos_perf = _performance(oos_block)

        # Strategy that looked best in-sample.
        n_star = int(np.argmax(is_perf))

        # Relative rank of n* among OOS performances, in (0, 1).
        # rank = (# strategies it beats OOS) -> ordinal; convert to a (0,1) fraction.
        order = np.argsort(oos_perf)  # ascending
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        omega = ranks[n_star] / (n + 1.0)  # in (0, 1), never exactly 0 or 1
        lam = float(np.log(omega / (1.0 - omega)))
        logits.append(lam)

    return np.asarray(logits, dtype=np.float64)


def probability_of_backtest_overfit(perf_matrix: npt.ArrayLike, n_splits: int = 16) -> float:
    """PBO: fraction of CSCV splits where the IS-best strategy is OOS below median.

    Returns a probability in [0, 1]. < 0.5 indicates the selection generalises; >= 0.5
    indicates overfitting (the IS-best is no better than a coin flip out of sample).
    """
    logits = cscv_logits(perf_matrix, n_splits=n_splits)
    if logits.size == 0:
        return 1.0
    # λ <= 0  <=>  OOS relative rank <= median.
    return float(np.mean(logits <= 0.0))


def n_combinations(n_splits: int) -> int:
    """Number of CSCV combinations C(S, S/2) for ``n_splits`` blocks."""
    return comb(n_splits, n_splits // 2)
