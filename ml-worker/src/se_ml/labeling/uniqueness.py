"""Sample-uniqueness weighting + effective-N for overlapping triple-barrier labels.

Triple-barrier labels overlap: row ``i`` occupies the span ``[ts_i, t1_i]`` (``t1`` up to
``max_hold`` bars ahead), so labels covering the same trending window are NOT independent
draws. Training them iid over-weights busy/concurrent periods, and treating the rows as iid
in DSR/PBO overstates significance (the effective sample size is much smaller than the
nominal row count). This module implements Lopez de Prado's *average-uniqueness* weights
(``Advances in Financial Machine Learning``, ch. 4) and the Kish *effective sample size*.

Both functions are pure and deterministic. They are PIT-safe: they use only ``ts`` and the
ALREADY-REALIZED barrier-touch time ``t1`` — the very same ``t1`` the CPCV purge relies on —
so nothing here looks ahead beyond what the labeler already committed to.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]


def _to_int64_and_validity(
    values: npt.ArrayLike,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """Coerce timestamps (datetime64 / numeric) to sortable int64 plus a validity mask.

    A value is INVALID when it is NaN (numeric) or NaT (datetime). Invalid entries receive
    an int64 placeholder (0) that callers must not treat as a real timestamp — the returned
    boolean mask flags them.
    """
    arr = np.asarray(values)
    if arr.dtype == object:
        # Mixed / python-object input (e.g. a column of Timestamps or python floats): try a
        # datetime coercion first, then fall back to numeric.
        conv = pd.to_datetime(pd.Series(arr), errors="coerce")
        if bool(conv.notna().any()):
            arr = conv.to_numpy()
        else:
            arr = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy()
    if np.issubdtype(arr.dtype, np.datetime64):
        valid = ~np.isnat(arr)
        ints = arr.astype("datetime64[ns]").astype(np.int64)
        return np.where(valid, ints, 0).astype(np.int64), valid
    farr = arr.astype(np.float64)
    valid = ~np.isnan(farr)
    ints = np.rint(np.where(valid, farr, 0.0)).astype(np.int64)
    return ints, valid


def average_uniqueness(ts: npt.ArrayLike, t1: npt.ArrayLike) -> FloatArray:
    """Lopez de Prado average-uniqueness weights for overlapping labels.

    For label ``i`` spanning ``[ts_i, t1_i]``, the *concurrency* at each event point ``p``
    inside the span is the number of labels whose own span ``[ts_j, t1_j]`` covers ``p``
    (including ``i`` itself, so concurrency is always ``>= 1``). The label's uniqueness is the
    mean of ``1 / concurrency`` over the DISTINCT event points (the ``ts`` values) that fall
    inside its span. Returns per-label weights in ``(0, 1]``.

    Robustness:
      * ``n == 0`` -> empty array; ``n == 1`` -> ``[1.0]``.
      * A row whose ``t1`` is missing / NaN / NaT (or ``t1 < ts``) has no well-formed span:
        it is treated as a point event for *others'* concurrency and its OWN weight falls
        back to ``1.0``.
      * Input need not be sorted; all comparisons are by timestamp VALUE.

    The computation is ``O(n^2)`` (fine for the few-thousand-row training subsets it runs on),
    deterministic, and PIT-safe (only ``ts`` and the realized barrier time ``t1`` are read).
    """
    starts, _ = _to_int64_and_validity(ts)
    n = int(starts.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)

    ends_raw, t1_valid = _to_int64_and_validity(t1)
    if int(ends_raw.size) != n:
        raise ValueError("ts and t1 must have the same length")

    # A span is well-formed only when t1 is present AND t1 >= ts. Otherwise the row becomes a
    # point event (end = start): it contributes concurrency only at its own timestamp and its
    # own weight is forced to 1.0 below.
    well_formed = t1_valid & (ends_raw >= starts)
    ends = np.where(well_formed, ends_raw, starts)

    # cover[j, k] = does label k's span cover event point starts[j]?  (O(n^2), one pass.)
    cover = (starts[None, :] <= starts[:, None]) & (starts[:, None] <= ends[None, :])
    conc_at = cover.sum(axis=1).astype(np.float64)  # >= 1 (k == j always covers point j)
    inv = 1.0 / conc_at

    weights = np.empty(n, dtype=np.float64)
    for i in range(n):
        in_span = (starts >= starts[i]) & (starts <= ends[i])
        # Mean of 1/concurrency over the DISTINCT event points inside the span. Concurrency is
        # a function of the point value, so duplicate ts collapse to one point.
        vals = starts[in_span]
        _uniq, first = np.unique(vals, return_index=True)
        pos = np.nonzero(in_span)[0][first]
        weights[i] = float(np.mean(inv[pos]))

    # Rows without a well-formed span carry no overlap information -> full weight.
    weights[~well_formed] = 1.0
    return weights


def effective_n(weights: npt.ArrayLike) -> float:
    """Kish effective sample size ``(sum w)^2 / sum(w^2)`` for a weight vector.

    Equals ``n`` when all weights are equal and shrinks toward ``1`` as the mass concentrates
    on a few rows. Returns ``0.0`` for an empty (or all-NaN / all-zero) input.
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    w = w[~np.isnan(w)]
    if w.size == 0:
        return 0.0
    s1 = float(w.sum())
    s2 = float(np.dot(w, w))
    if s2 <= 0.0:
        return 0.0
    return s1 * s1 / s2
