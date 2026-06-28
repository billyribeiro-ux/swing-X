"""Golden tests for meta-labeling (se_ml.labeling.meta_labeling).

Meta-labeling is a staged component (see the module docstring): a secondary model decides
whether to ACT on a primary signal and how much to SIZE it. These tests pin the exact,
deterministic behavior of its pure functions so the capability is verified and ready to wire
into the validation pipeline — not advertised-but-unchecked.
"""

from __future__ import annotations

import numpy as np

from se_ml.labeling.meta_labeling import (
    MetaDecision,
    apply_sizing,
    decide,
    make_meta_labels,
)


def test_make_meta_labels_default_threshold_is_any_profit() -> None:
    # Label is 1 iff realized R is strictly positive (any profit); 0 (breakeven) is NOT a win.
    labels = make_meta_labels([0.5, -0.2, 0.0, 1.3])
    assert labels.tolist() == [1, 0, 0, 1]


def test_make_meta_labels_respects_minimum_edge_threshold() -> None:
    # With threshold_r=0.3, only returns strictly above 0.3 count as "act".
    labels = make_meta_labels([0.5, 0.3, 0.31, -1.0], threshold_r=0.3)
    assert labels.tolist() == [1, 0, 1, 0]


def test_decide_gates_below_threshold_and_sizes_linearly() -> None:
    # threshold 0.5, max_size 1.0 => span 0.5. size scales 0 at threshold -> 1 at p=1.0.
    decisions = decide([0.4, 0.5, 0.75, 1.0], act_threshold=0.5, max_size=1.0)
    assert [d.act for d in decisions] == [False, True, True, True]
    sizes = [round(d.size, 6) for d in decisions]
    assert sizes == [0.0, 0.0, 0.5, 1.0]


def test_decide_respects_max_size_cap() -> None:
    # max_size 0.5 caps the top of the linear ramp at 0.5, not 1.0.
    decisions = decide([0.5, 1.0], act_threshold=0.5, max_size=0.5)
    assert [round(d.size, 6) for d in decisions] == [0.0, 0.5]


def test_apply_sizing_zeros_non_acts_and_scales_acts() -> None:
    decisions = [
        MetaDecision(act=False, size=0.0, proba=0.4),
        MetaDecision(act=True, size=0.2, proba=0.6),
        MetaDecision(act=True, size=1.0, proba=1.0),
    ]
    sized = apply_sizing([2.0, 2.0, -1.0], decisions)
    assert np.allclose(sized.to_numpy(), [0.0, 0.4, -1.0])
    assert sized.name == "meta_ret_r"


def test_decide_then_apply_sizing_roundtrip() -> None:
    # End-to-end: probabilities -> decisions -> sized returns, all consistent.
    proba = [0.4, 0.6, 1.0]
    ret_r = [2.0, 2.0, -1.0]
    decisions = decide(proba, act_threshold=0.5, max_size=1.0)
    sized = apply_sizing(ret_r, decisions)
    # idx0 gated off -> 0; idx1 size 0.2 -> 0.4; idx2 size 1.0 -> -1.0
    assert np.allclose(sized.to_numpy(), [0.0, 0.4, -1.0])
