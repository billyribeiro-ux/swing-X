"""Promotion gate — the single source of truth the Rust side re-checks.

A candidate strategy is promoted to live/paper ONLY when ALL of the following hold:

  1. ``dsr > 0``                         — Deflated Sharpe Ratio is positive (edge survives
                                            selection bias + non-normality + track length).
  2. ``pbo < 0.5``                       — Probability of Backtest Overfit is below a coin
                                            flip (the IS-best generalises out of sample).
  3. ``oos_expectancy_cost_aware > 0``   — cost-aware out-of-sample expectancy is positive
                                            (the edge survives realistic trading costs).
  4. ``n_regimes_positive >= 2``         — positive in at least two distinct market regimes
                                            (not a one-regime fluke).

:func:`evaluate` is a PURE function returning each sub-condition as a boolean plus the
overall ``passed`` flag. The Rust ``se-validation`` crate mirrors these exact keys and
re-derives ``passed`` independently as a defence-in-depth check.
"""

from __future__ import annotations

from typing import TypedDict

# Thresholds are constants so both sides agree byte-for-byte.
DSR_MIN: float = 0.0
PBO_MAX: float = 0.5
OOS_EXPECTANCY_MIN: float = 0.0
MIN_POSITIVE_REGIMES: int = 2


class GateResult(TypedDict):
    dsr_ok: bool
    pbo_ok: bool
    oos_expectancy_ok: bool
    regimes_ok: bool
    passed: bool


def evaluate(
    dsr: float,
    pbo: float,
    oos_expectancy_cost_aware: float,
    n_regimes_positive: int,
) -> GateResult:
    """Evaluate the promotion gate. Pure; returns each sub-condition + overall ``passed``.

    Parameters
    ----------
    dsr
        Deflated Sharpe Ratio (probability in (0, 1)); must be strictly > ``DSR_MIN``.
    pbo
        Probability of Backtest Overfit; must be strictly < ``PBO_MAX``.
    oos_expectancy_cost_aware
        Cost-aware out-of-sample expectancy in R; must be strictly > ``OOS_EXPECTANCY_MIN``.
    n_regimes_positive
        Number of regimes with positive contribution; must be >= ``MIN_POSITIVE_REGIMES``.
    """
    dsr_ok = dsr > DSR_MIN
    pbo_ok = pbo < PBO_MAX
    oos_expectancy_ok = oos_expectancy_cost_aware > OOS_EXPECTANCY_MIN
    regimes_ok = n_regimes_positive >= MIN_POSITIVE_REGIMES
    passed = bool(dsr_ok and pbo_ok and oos_expectancy_ok and regimes_ok)
    return GateResult(
        dsr_ok=dsr_ok,
        pbo_ok=pbo_ok,
        oos_expectancy_ok=oos_expectancy_ok,
        regimes_ok=regimes_ok,
        passed=passed,
    )
