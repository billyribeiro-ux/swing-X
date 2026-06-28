"""Combinatorial Purged Cross-Validation (CPCV) with explicit purge + embargo.

Standard k-fold CV leaks in a financial setting because (a) labels span a *window*
``[entry_ts, t1]`` so a training observation whose window overlaps a test block has
in effect "seen" the test period, and (b) serial correlation makes observations
immediately *after* a test block informative about it. CPCV fixes both:

  * **Combinatorial**: partition the timeline into ``N`` contiguous groups and, for every
    way to choose ``k`` of them as the test set (``C(N, k)`` combinations), use the rest
    for training. This yields many train/test paths instead of one.
  * **Purge**: drop any TRAIN observation whose label window ``[entry_ts, t1]`` overlaps
    the time span of *any* selected test group.
  * **Embargo**: additionally drop TRAIN observations that start within an ``embargo``
    buffer of bars immediately AFTER each test block, killing leakage from serial
    correlation.

The public entry point :func:`cpcv_splits` returns a list of ``(train_idx, test_idx)``
integer-index pairs. :func:`overlaps` and the test module assert that no purged/embargoed
leakage remains.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt
import pandas as pd

IntArray = npt.NDArray[np.int_]


@dataclass(frozen=True)
class Split:
    train_idx: IntArray
    test_idx: IntArray


def _as_int64(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Coerce timestamps (datetime64 or numeric) to a sortable int64 representation."""
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]").astype(np.int64)
    return arr.astype(np.int64)


def _group_bounds(n: int, n_groups: int) -> list[tuple[int, int]]:
    """Return [start, end) positional index bounds for ``n_groups`` contiguous groups."""
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    if n_groups > n:
        raise ValueError("n_groups cannot exceed number of observations")
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    return [(int(edges[g]), int(edges[g + 1])) for g in range(n_groups)]


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Inclusive interval overlap test on integer-valued timestamps."""
    return a_start <= b_end and b_start <= a_end


def cpcv_splits(
    event_ts: npt.ArrayLike,
    t1: npt.ArrayLike,
    n_groups: int = 8,
    k_test_groups: int = 2,
    embargo_bars: int = 0,
    purge: bool = True,
) -> list[Split]:
    """Generate CPCV train/test splits with purge + embargo.

    Parameters
    ----------
    event_ts
        Entry timestamp of each observation (label window start). Must be sorted ascending.
    t1
        Barrier-touch timestamp of each observation (label window end). ``t1[i] >= event_ts[i]``.
    n_groups
        Number of contiguous time groups ``N``.
    k_test_groups
        Groups held out as the test set per combination ``k`` (``1 <= k < N``).
    embargo_bars
        Number of bars (observations) embargoed AFTER each test block.
    purge
        If True, purge train observations whose label window overlaps any test span.

    Returns
    -------
    list[Split]
        One :class:`Split` per combination, with positional integer indices.
    """
    starts = _as_int64(event_ts)
    ends = _as_int64(t1)
    n = starts.size
    if ends.size != n:
        raise ValueError("event_ts and t1 must have the same length")
    if np.any(np.diff(starts) < 0):
        raise ValueError("event_ts must be sorted ascending")
    if np.any(ends < starts):
        raise ValueError("t1 must be >= event_ts elementwise")
    if not 1 <= k_test_groups < n_groups:
        raise ValueError("require 1 <= k_test_groups < n_groups")

    bounds = _group_bounds(n, n_groups)
    all_pos = np.arange(n)
    splits: list[Split] = []

    for combo in combinations(range(n_groups), k_test_groups):
        # Positional indices of the test groups.
        test_pos_list: list[int] = []
        for g in combo:
            s, e = bounds[g]
            test_pos_list.extend(range(s, e))
        test_idx = np.asarray(sorted(test_pos_list), dtype=np.int_)

        # Time spans covered by the selected test groups (in timestamp units).
        test_spans: list[tuple[int, int]] = []
        for g in combo:
            s, e = bounds[g]
            if e > s:
                test_spans.append((int(starts[s]), int(ends[s:e].max())))

        # Embargo: positional ranges immediately AFTER each test block.
        embargo_pos: set[int] = set()
        if embargo_bars > 0:
            for g in combo:
                _, e = bounds[g]
                embargo_pos.update(range(e, min(e + embargo_bars, n)))

        test_set = set(test_idx.tolist())
        train_pos: list[int] = []
        for i in all_pos:
            if i in test_set:
                continue
            if i in embargo_pos:
                continue
            if purge:
                # Purge if this train observation's label window overlaps any test span.
                leaks = any(
                    overlaps(int(starts[i]), int(ends[i]), span_s, span_e)
                    for span_s, span_e in test_spans
                )
                if leaks:
                    continue
            train_pos.append(int(i))

        splits.append(
            Split(
                train_idx=np.asarray(train_pos, dtype=np.int_),
                test_idx=test_idx,
            )
        )

    return splits


def assert_no_leakage(
    event_ts: npt.ArrayLike,
    t1: npt.ArrayLike,
    split: Split,
    n_groups: int,
    embargo_bars: int,
) -> None:
    """Raise AssertionError if any train obs overlaps a test span or sits in the embargo.

    Used by the test suite to *prove* purge + embargo actually remove leakage.
    """
    starts = _as_int64(event_ts)
    ends = _as_int64(t1)
    n = starts.size
    bounds = _group_bounds(n, n_groups)

    # Reconstruct the test spans and embargo positions from the test indices.
    test_set = set(split.test_idx.tolist())
    test_spans: list[tuple[int, int]] = []
    embargo_pos: set[int] = set()
    for s, e in bounds:
        group_pos = set(range(s, e))
        if group_pos & test_set:
            if e > s:
                test_spans.append((int(starts[s]), int(ends[s:e].max())))
            if embargo_bars > 0:
                embargo_pos.update(range(e, min(e + embargo_bars, n)))

    for i in split.train_idx.tolist():
        for span_s, span_e in test_spans:
            assert not overlaps(int(starts[i]), int(ends[i]), span_s, span_e), (
                f"train obs {i} (window [{starts[i]},{ends[i]}]) overlaps test span "
                f"[{span_s},{span_e}] — purge failed"
            )
        assert i not in embargo_pos, f"train obs {i} is inside the embargo buffer"


def to_frame(splits: list[Split]) -> pd.DataFrame:
    """Summarise splits as a frame of sizes (for logging / debugging)."""
    return pd.DataFrame(
        {
            "split": list(range(len(splits))),
            "n_train": [s.train_idx.size for s in splits],
            "n_test": [s.test_idx.size for s in splits],
        }
    )
