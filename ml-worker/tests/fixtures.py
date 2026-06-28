"""Synthetic dataset generators for the golden tests.

Three flavours, all returning a pandas DataFrame in the on-disk contract layout
(``ts``, ``t1``, ``label`` in R units, ``regime``, and ``layer__feature`` columns):

  * :func:`genuine_edge_dataset` — features genuinely (causally) predict the label, so a
    model trained on the past keeps an edge out-of-sample. Used to prove the harness does
    NOT reject good strategies (low PBO, positive DSR, gate PASSES).
  * :func:`pure_noise_dataset` — features are independent of the label. Any apparent edge
    is selection noise. Used to prove high PBO / failing gate.
  * :func:`leaky_dataset` — contains a future-peeking feature equal to (a noised copy of)
    the label itself. In-sample it looks spectacular; out-of-sample, once the leak is
    purged/embargoed and the model can't actually see the future, it collapses. THE
    leakage checkpoint.

Also provides :func:`make_perf_matrix_*` builders for the PBO test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_REGIMES = np.array(["bull", "bear", "chop"])


def _timeline(n: int, start: str = "2020-01-01") -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Event timestamps and (event + hold) barrier timestamps on a daily grid."""
    ts = pd.date_range(start=start, periods=n, freq="D")
    # Each label window spans 3 bars (so adjacent events overlap — exercises purging).
    t1 = ts + pd.Timedelta(days=3)
    return ts, t1


def genuine_edge_dataset(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Features causally drive the label; the edge is real and persistent."""
    rng = np.random.default_rng(seed)
    ts, t1 = _timeline(n)

    momentum = rng.normal(size=n)
    trend = rng.normal(size=n)
    vol = rng.normal(size=n)
    noise_feat = rng.normal(size=n)

    # A stable linear signal + noise drives the R outcome.
    signal = 0.9 * momentum + 0.6 * trend - 0.3 * vol
    label = signal + rng.normal(scale=1.0, size=n)
    # Express in R-ish units centred so there is a genuine positive expectancy.
    label = np.clip(label, -1.0, 2.0)

    regime = _REGIMES[rng.integers(0, 3, size=n)]
    return pd.DataFrame(
        {
            "ts": ts,
            "t1": t1,
            "label": label,
            "regime": regime,
            "momentum__signal": momentum,
            "trend__slope": trend,
            "volatility__atr_norm": vol,
            "momentum__noise": noise_feat,
        }
    )


def pure_noise_dataset(n: int = 1500, seed: int = 1) -> pd.DataFrame:
    """Features are independent of the label, and the label has ZERO expectancy.

    The label is symmetric around zero (no structural drift), so absent any real predictive
    feature the achievable edge is nil — and once a per-trade cost is subtracted, the
    cost-aware OOS expectancy is negative. The harness must therefore REJECT it.
    """
    rng = np.random.default_rng(seed)
    ts, t1 = _timeline(n)

    label = np.clip(rng.normal(scale=1.0, size=n), -1.5, 1.5)
    regime = _REGIMES[rng.integers(0, 3, size=n)]
    return pd.DataFrame(
        {
            "ts": ts,
            "t1": t1,
            "label": label,
            "regime": regime,
            "momentum__a": rng.normal(size=n),
            "trend__b": rng.normal(size=n),
            "volatility__c": rng.normal(size=n),
            "momentum__d": rng.normal(size=n),
        }
    )


def leaky_dataset(n: int = 1500, seed: int = 2) -> pd.DataFrame:
    """A planted look-ahead leak: spectacular in-sample, collapses out-of-sample.

    The leak feature reveals the label's sign — but only in the FIRST part of the
    timeline. In the held-out (later) part its sign FLIPS to point the wrong way::

        leak__lookahead[i] =  +label_signal[i]   for the early (in-sample) bars
                              -label_signal[i]   for the later (out-of-sample) bars

    This is the signature of look-ahead bias / a non-point-in-time feature: a quantity that
    *appears* to predict the label on the data the researcher fit on, but whose relationship
    does not hold on genuinely unseen, later data. Because the genuine features carry no
    edge, a naive search selects the leak-driven config on the in-sample slice (great IS
    metrics) and the validation harness — which selects IS but judges on the later OOS
    slice — sees it collapse: cost-aware OOS expectancy goes non-positive and DSR deflates,
    so the promotion gate REJECTS it. A genuine, stationary edge does not flip and passes.
    """
    rng = np.random.default_rng(seed)
    ts, t1 = _timeline(n)

    base = rng.normal(size=n)
    label = base + rng.normal(scale=1.0, size=n)
    label = np.clip(label, -1.5, 1.5)  # symmetric: no free drift once the leak flips

    # The leak tracks the label early, then its sign flips for the later (OOS) portion.
    sign = np.ones(n)
    sign[n // 2 :] = -1.0
    leak = sign * base + rng.normal(scale=0.01, size=n)

    regime = _REGIMES[rng.integers(0, 3, size=n)]
    return pd.DataFrame(
        {
            "ts": ts,
            "t1": t1,
            "label": label,
            "regime": regime,
            "leak__lookahead": leak,
            # Genuine features are pure noise: nothing real survives the regime flip.
            "momentum__a": rng.normal(size=n),
            "trend__b": rng.normal(size=n),
            "volatility__c": rng.normal(size=n),
        }
    )


# --------------------------------------------------------------------------- #
# PBO performance-matrix builders
# --------------------------------------------------------------------------- #
def perf_matrix_genuine(t: int = 800, n_strategies: int = 20, seed: int = 7) -> np.ndarray:
    """Performance matrix where ONE strategy has a persistent (IS+OOS) edge.

    The genuinely good strategy is good across the whole timeline, so CSCV's IS-best
    selection keeps choosing it and it stays above-median OOS -> low PBO.
    """
    rng = np.random.default_rng(seed)
    M = rng.normal(scale=1.0, size=(t, n_strategies))
    # Strategy 0 has a real, time-stable positive mean.
    M[:, 0] += 0.5
    return M


def perf_matrix_overfit(t: int = 800, n_strategies: int = 200, seed: int = 8) -> np.ndarray:
    """Many pure-noise strategies; the IS-best is best by luck -> high PBO.

    With many independent noise strategies, whichever wins in-sample is essentially random
    out-of-sample, so it lands below median OOS about half the time -> PBO ~ 0.5+.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(scale=1.0, size=(t, n_strategies))
