"""Tests for Combinatorial Purged CV: assert purge + embargo actually remove leakage."""

from __future__ import annotations

from math import comb

import numpy as np

from se_ml.cv.cpcv import assert_no_leakage, cpcv_splits, overlaps


def _events(n=120, window=3):
    ts = np.arange(n, dtype=np.int64)
    t1 = ts + window  # each label window spans `window` bars -> adjacent events overlap
    return ts, t1


def test_number_of_combinations():
    ts, t1 = _events()
    splits = cpcv_splits(ts, t1, n_groups=6, k_test_groups=2, embargo_bars=0)
    # C(6, 2) = 15 combinations for the test-group choice.
    assert len(splits) == 15
    assert len(splits) == comb(6, 2)


def test_train_and_test_are_disjoint():
    ts, t1 = _events()
    splits = cpcv_splits(ts, t1, n_groups=8, k_test_groups=2, embargo_bars=3)
    for sp in splits:
        assert set(sp.train_idx.tolist()).isdisjoint(set(sp.test_idx.tolist()))


def test_purge_removes_overlapping_train_observations():
    # With overlapping label windows, the train rows whose [ts, t1] overlap a test span
    # must be removed. Compare purge=True vs purge=False: purge must drop strictly more.
    ts, t1 = _events(n=120, window=5)
    purged = cpcv_splits(ts, t1, n_groups=8, k_test_groups=2, embargo_bars=0, purge=True)
    unpurged = cpcv_splits(ts, t1, n_groups=8, k_test_groups=2, embargo_bars=0, purge=False)
    total_purged = sum(s.train_idx.size for s in purged)
    total_unpurged = sum(s.train_idx.size for s in unpurged)
    assert total_purged < total_unpurged


def test_embargo_removes_adjacent_train_observations():
    ts, t1 = _events(n=120, window=1)  # minimal windows so only embargo bites
    no_embargo = cpcv_splits(ts, t1, n_groups=8, k_test_groups=2, embargo_bars=0, purge=True)
    embargo = cpcv_splits(ts, t1, n_groups=8, k_test_groups=2, embargo_bars=4, purge=True)
    assert sum(s.train_idx.size for s in embargo) < sum(s.train_idx.size for s in no_embargo)


def test_no_leakage_invariant_holds():
    # THE assertion: for every split, no train obs overlaps a test span or sits in embargo.
    ts, t1 = _events(n=160, window=4)
    n_groups, embargo = 8, 5
    splits = cpcv_splits(ts, t1, n_groups=n_groups, k_test_groups=2,
                         embargo_bars=embargo, purge=True)
    for sp in splits:
        assert_no_leakage(ts, t1, sp, n_groups=n_groups, embargo_bars=embargo)


def test_overlap_helper():
    assert overlaps(0, 5, 5, 9)        # touch at boundary -> overlap (inclusive)
    assert overlaps(0, 10, 3, 4)       # contained
    assert not overlaps(0, 5, 6, 9)    # disjoint


def test_sorted_requirement_enforced():
    ts = np.array([5, 1, 2], dtype=np.int64)
    t1 = ts + 1
    try:
        cpcv_splits(ts, t1, n_groups=3, k_test_groups=1)
    except ValueError as e:
        assert "sorted" in str(e)
    else:
        raise AssertionError("expected ValueError for unsorted event_ts")
