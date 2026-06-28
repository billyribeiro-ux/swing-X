"""Meta-labeling: a secondary model that decides whether to ACT on a primary signal.

The primary model (or rule) produces a directional signal (side). The triple-barrier
labeler then tells us, in hindsight, whether each acted-on signal made money. The
meta-label is binary: 1 if the primary signal was profitable (positive R), else 0.

A secondary classifier is trained to predict that meta-label from features. At inference
its probability is used both to GATE (act only when ``p >= threshold``) and to SIZE the
position (e.g. proportional to ``p``). This decouples "which way" from "how much / whether"
and is a standard way to lift precision without touching the primary's recall.

Status: STAGED. These pure functions are verified by ``tests/test_meta_labeling.py`` and are
ready to wire into the validation pipeline (a secondary classifier feeding ``decide`` before the
gate computes cost-aware OOS expectancy on the sized returns). They are deliberately not yet on
the live path so the promotion gate keeps measuring the primary edge in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


def make_meta_labels(ret_r: npt.ArrayLike, threshold_r: float = 0.0) -> npt.NDArray[np.int_]:
    """Binary meta-labels from realized R returns.

    Label is 1 when the realized return exceeds ``threshold_r`` (default: any profit),
    else 0. The ``threshold_r`` lets the caller require a minimum edge to count as "act".
    """
    arr = np.asarray(ret_r, dtype=np.float64).ravel()
    return (arr > threshold_r).astype(int)


@dataclass
class MetaDecision:
    act: bool       # whether to take the trade
    size: float     # position size in [0, 1]
    proba: float    # secondary-model probability of a profitable trade


def decide(
    proba: npt.ArrayLike,
    act_threshold: float = 0.5,
    max_size: float = 1.0,
) -> list[MetaDecision]:
    """Turn secondary-model probabilities into act/size decisions.

    Acts only when ``proba >= act_threshold``. Size scales linearly from 0 at the
    threshold to ``max_size`` at probability 1.0, so marginal signals get small size.
    """
    p = np.asarray(proba, dtype=np.float64).ravel()
    decisions: list[MetaDecision] = []
    span = max(1e-9, 1.0 - act_threshold)
    for pi in p:
        if pi >= act_threshold:
            size = float(np.clip((pi - act_threshold) / span * max_size, 0.0, max_size))
            decisions.append(MetaDecision(act=True, size=size, proba=float(pi)))
        else:
            decisions.append(MetaDecision(act=False, size=0.0, proba=float(pi)))
    return decisions


def apply_sizing(ret_r: npt.ArrayLike, decisions: list[MetaDecision]) -> pd.Series:
    """Apply meta decisions to realized returns: 0 when not acting, else ``size · R``."""
    arr = np.asarray(ret_r, dtype=np.float64).ravel()
    sized = np.array(
        [d.size * r if d.act else 0.0 for r, d in zip(arr, decisions, strict=True)],
        dtype=np.float64,
    )
    return pd.Series(sized, name="meta_ret_r")
