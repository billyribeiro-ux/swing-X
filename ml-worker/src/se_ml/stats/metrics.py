"""Performance metrics for trade-return series (returns expressed in R units).

Selection metrics used by the engine: expectancy, profit factor, CVaR(5%), MAR/Calmar,
Sharpe. **Win rate is intentionally NOT exposed as a selection metric** — it is trivially
gameable (tiny wins / huge losses) and rewards exactly the overfit behaviour the
validation harness exists to reject. Use expectancy + profit factor + CVaR instead.

All functions accept a 1-D array-like of per-trade returns (R multiples). They are pure
and deterministic.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _as_array(returns: npt.ArrayLike) -> FloatArray:
    arr = np.asarray(returns, dtype=np.float64).ravel()
    return arr[~np.isnan(arr)]


def expectancy(returns: npt.ArrayLike) -> float:
    """Mean return per trade (in R). The headline edge measure."""
    arr = _as_array(returns)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def profit_factor(returns: npt.ArrayLike) -> float:
    """Gross profit / gross loss.

    Returns ``inf`` when there are no losses and >0 gross profit; ``0.0`` when there is
    no gross profit. A genuine edge sits comfortably above 1.0.
    """
    arr = _as_array(returns)
    gross_profit = float(arr[arr > 0].sum())
    gross_loss = float(-arr[arr < 0].sum())
    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0.0 else 0.0
    return gross_profit / gross_loss


def sharpe(returns: npt.ArrayLike, periods_per_year: float | None = None) -> float:
    """Sharpe ratio of the per-trade return series.

    With ``periods_per_year`` given, the ratio is annualised by ``sqrt(periods_per_year)``.
    Uses the sample standard deviation (ddof=1). Returns 0.0 for degenerate input.
    """
    arr = _as_array(returns)
    if arr.size < 2:
        return 0.0
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        return 0.0
    ratio = float(arr.mean()) / sd
    if periods_per_year is not None:
        ratio *= float(np.sqrt(periods_per_year))
    return ratio


def cvar(returns: npt.ArrayLike, alpha: float = 0.05) -> float:
    """Conditional Value at Risk (expected shortfall) at level ``alpha``.

    Mean of the worst ``alpha`` fraction of returns. Returned as a (typically negative)
    return value, i.e. the average loss in the tail. ``cvar5`` in the contract is
    ``cvar(returns, 0.05)``.
    """
    arr = _as_array(returns)
    if arr.size == 0:
        return 0.0
    sorted_r = np.sort(arr)
    # Number of tail observations; at least one.
    k = max(1, int(np.floor(alpha * sorted_r.size)))
    return float(sorted_r[:k].mean())


def cvar5(returns: npt.ArrayLike) -> float:
    """CVaR at the 5% level — the contract's ``cvar5`` field."""
    return cvar(returns, 0.05)


def max_drawdown(returns: npt.ArrayLike) -> float:
    """Maximum drawdown of the cumulative (additive, R-unit) equity curve.

    Returned as a non-negative magnitude (0.0 means no drawdown).
    """
    arr = _as_array(returns)
    if arr.size == 0:
        return 0.0
    equity = np.cumsum(arr)
    running_max = np.maximum.accumulate(equity)
    drawdowns = running_max - equity
    return float(drawdowns.max())


def mar(returns: npt.ArrayLike) -> float:
    """MAR / Calmar-style ratio: total return divided by max drawdown.

    Uses additive R-unit equity. Returns ``inf`` when there is no drawdown but positive
    total return, and 0.0 when total return is non-positive.
    """
    arr = _as_array(returns)
    if arr.size == 0:
        return 0.0
    total = float(arr.sum())
    mdd = max_drawdown(arr)
    if mdd == 0.0:
        return float("inf") if total > 0.0 else 0.0
    return total / mdd


def cost_aware_returns(returns: npt.ArrayLike, cost_per_trade: float) -> FloatArray:
    """Subtract a fixed round-trip cost (in R) from each trade return."""
    arr = _as_array(returns)
    return arr - float(cost_per_trade)


def summary(returns: npt.ArrayLike, periods_per_year: float | None = None) -> dict[str, float]:
    """Bundle the selection metrics into a single dict (no win_rate, by design)."""
    arr = _as_array(returns)
    return {
        "expectancy": expectancy(arr),
        "profit_factor": profit_factor(arr),
        "sharpe": sharpe(arr, periods_per_year),
        "cvar5": cvar5(arr),
        "mar": mar(arr),
        "n": int(arr.size),
    }
