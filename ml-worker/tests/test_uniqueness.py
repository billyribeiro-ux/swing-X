"""Unit tests for average-uniqueness weights and Kish effective sample size.

Overlapping triple-barrier labels are not iid; :func:`average_uniqueness` down-weights
concurrent labels and :func:`effective_n` reports the honest sample size. Both are pure and
deterministic, so these tests pin hand-computable cases.
"""

from __future__ import annotations

import numpy as np

from se_ml.labeling.uniqueness import average_uniqueness, effective_n


def test_average_uniqueness_hand_computed_overlapping() -> None:
    # Three overlapping spans on an integer timeline:
    #   label 0: [0, 2]   label 1: [1, 3]   label 2: [2, 4]
    # Event points are the starts {0, 1, 2}; concurrency at each point:
    #   p=0 -> 1 (only label 0),  p=1 -> 2 (labels 0,1),  p=2 -> 3 (labels 0,1,2).
    # Uniqueness = mean of 1/concurrency over the DISTINCT points inside each span:
    #   w0 = mean(1, 1/2, 1/3) = 11/18   (points 0,1,2 all in [0,2])
    #   w1 = mean(1/2, 1/3)    = 5/12    (points 1,2 in [1,3])
    #   w2 = 1/3               = 1/3     (point 2 in [2,4])
    ts = np.array([0, 1, 2], dtype=np.int64)
    t1 = np.array([2, 3, 4], dtype=np.int64)
    w = average_uniqueness(ts, t1)
    expected = np.array([11.0 / 18.0, 5.0 / 12.0, 1.0 / 3.0])
    assert np.allclose(w, expected)
    # Weights live in (0, 1].
    assert np.all(w > 0.0) and np.all(w <= 1.0)


def test_average_uniqueness_non_overlapping_all_ones() -> None:
    # Disjoint spans: every label's window contains only its own event point -> weight 1.0.
    ts = np.array([0, 10, 20, 30], dtype=np.int64)
    t1 = np.array([1, 11, 21, 31], dtype=np.int64)
    w = average_uniqueness(ts, t1)
    assert np.allclose(w, np.ones(4))


def test_average_uniqueness_datetime_matches_integer() -> None:
    # datetime64 inputs must give the same geometry as the equivalent integer timeline.
    ts = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
    t1 = np.array(["2020-01-03", "2020-01-04", "2020-01-05"], dtype="datetime64[ns]")
    w = average_uniqueness(ts, t1)
    expected = np.array([11.0 / 18.0, 5.0 / 12.0, 1.0 / 3.0])
    assert np.allclose(w, expected)


def test_average_uniqueness_unsorted_is_permutation_invariant() -> None:
    ts = np.array([0, 1, 2], dtype=np.int64)
    t1 = np.array([2, 3, 4], dtype=np.int64)
    base = average_uniqueness(ts, t1)
    perm = np.array([2, 0, 1])
    w_perm = average_uniqueness(ts[perm], t1[perm])
    assert np.allclose(w_perm, base[perm])


def test_average_uniqueness_nan_t1_falls_back_to_one() -> None:
    # A missing t1 has no well-formed span -> that row's own weight is 1.0, and it acts as a
    # point event for the others' concurrency (never crashes).
    ts = np.array([0.0, 1.0, 2.0])
    t1 = np.array([2.0, np.nan, 4.0])
    w = average_uniqueness(ts, t1)
    assert w[1] == 1.0
    assert np.all(w > 0.0) and np.all(w <= 1.0)


def test_average_uniqueness_edge_sizes() -> None:
    assert average_uniqueness(np.array([]), np.array([])).shape == (0,)
    assert np.array_equal(average_uniqueness(np.array([5]), np.array([9])), np.array([1.0]))


def test_effective_n_uniform_weights_equals_count() -> None:
    for c in (1.0, 0.3, 7.5):
        w = np.full(20, c)
        assert np.isclose(effective_n(w), 20.0)


def test_effective_n_one_dominant_weight_is_about_one() -> None:
    w = np.concatenate([[1.0], np.full(50, 1e-6)])
    assert effective_n(w) < 1.001
    assert effective_n(w) > 0.999


def test_effective_n_empty_and_degenerate() -> None:
    assert effective_n(np.array([])) == 0.0
    assert effective_n(np.array([0.0, 0.0, 0.0])) == 0.0
    # Overlapping labels shrink effective-N below the nominal count.
    ts = np.arange(30, dtype=np.int64)
    t1 = ts + 5  # each label overlaps ~5 neighbours
    eff = effective_n(average_uniqueness(ts, t1))
    assert 0.0 < eff < 30.0
