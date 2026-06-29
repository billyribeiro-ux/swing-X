"""Precision-optimized acting-threshold selection on the /validate path.

The promotion gate's north-star metric is OUT-OF-SAMPLE PRECISION (win rate is banned). The
``/validate`` route picks a meta-labeling acting threshold tau* on the in-sample half (no OOS
peeking) and reports precision/recall measured on the OOS half. These tests pin the pure,
deterministic helpers that implement that layer — no HTTP server required.

Key property: on a dataset where higher model-proba genuinely correlates with profit, acting
at the SELECTED tau* must lift (or at least maintain) precision versus acting at 0.5.
"""

from __future__ import annotations

import numpy as np

from se_ml.server import (
    DEFAULT_COST_PER_TRADE_R,
    precision_recall_at,
    select_act_threshold,
)


def _legacy_expectancy_max_threshold(
    proba_is: np.ndarray,
    r_is: np.ndarray,
    cost: float,
) -> float:
    """Reference implementation of the OLD expectancy-maximizing selector.

    Kept inline so the precision-first selector can be contrasted against the behavior it
    replaced: maximize cost-aware expectancy over acted rows subject to the same min-acted
    and recall>=0.10 floors, single fold over the whole IS half. Used only by the tests.
    """
    from se_ml.labeling.meta_labeling import make_meta_labels
    from se_ml.stats import metrics as mx

    n = int(proba_is.size)
    if n == 0:
        return 0.5
    min_acted = max(8, int(np.ceil(0.10 * n)))
    labels = make_meta_labels(r_is)
    n_profit = int(labels.sum())
    deciles = np.quantile(proba_is, np.linspace(0.0, 0.9, 10))
    coarse = np.linspace(0.30, 0.90, 7)
    grid = np.unique(np.concatenate([deciles, coarse]))
    best_tau, best_score, found = 0.5, -np.inf, False
    for tau in grid:
        acted = proba_is >= tau
        n_acted = int(acted.sum())
        if n_acted < min_acted:
            continue
        recall = (int((acted & (labels == 1)).sum()) / n_profit) if n_profit > 0 else 0.0
        if recall < 0.10:
            continue
        score = float(np.mean(mx.cost_aware_returns(r_is[acted], cost)))
        if score > best_score:
            best_score, best_tau, found = score, float(tau), True
    return best_tau if found else 0.5


def _synth_signal_dataset(
    n: int = 2000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities that genuinely correlate with profit.

    ``proba`` is uniform in [0, 1]; the realized R is more likely positive (and larger) the
    higher the proba, so a higher acting threshold concentrates on profitable trades and lifts
    precision. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    proba = rng.uniform(0.0, 1.0, size=n)
    # Higher proba => higher chance of profit and bigger edge; noise keeps it non-trivial.
    win = rng.uniform(0.0, 1.0, size=n) < proba
    r = np.where(win, rng.uniform(0.1, 2.0, size=n), rng.uniform(-2.0, -0.1, size=n))
    return proba, r.astype(np.float64)


def _split_half(proba: np.ndarray, r: np.ndarray) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    mid = proba.size // 2
    return proba[:mid], r[:mid], proba[mid:], r[mid:]


def test_selected_tau_lifts_or_maintains_precision_oos() -> None:
    proba, r = _synth_signal_dataset(n=2000, seed=0)
    proba_is, r_is, proba_oos, r_oos = _split_half(proba, r)

    tau = select_act_threshold(proba_is, r_is, DEFAULT_COST_PER_TRADE_R)

    prec_tau, recall_tau, n_acted = precision_recall_at(proba_oos, r_oos, tau)
    prec_half, _, _ = precision_recall_at(proba_oos, r_oos, 0.5)

    # The chosen threshold must lift or maintain OOS precision versus acting at 0.5.
    assert prec_tau >= prec_half, (
        f"tau*={tau} precision {prec_tau} should be >= 0.5-threshold precision {prec_half}"
    )

    # Field-range invariants the contract guarantees.
    assert 0.0 <= prec_tau <= 1.0
    assert 0.0 <= recall_tau <= 1.0
    assert 0.0 <= tau <= 1.0
    assert n_acted >= 0


def test_pure_noise_returns_valid_fields_and_does_not_crash() -> None:
    rng = np.random.default_rng(7)
    n = 1000
    proba = rng.uniform(0.0, 1.0, size=n)
    r = rng.normal(0.0, 1.0, size=n)  # independent of proba: no real edge
    proba_is, r_is, proba_oos, r_oos = _split_half(proba, r)

    tau = select_act_threshold(proba_is, r_is, DEFAULT_COST_PER_TRADE_R)
    prec, recall, n_acted = precision_recall_at(proba_oos, r_oos, tau)

    assert 0.0 <= tau <= 1.0
    assert 0.0 <= prec <= 1.0
    assert 0.0 <= recall <= 1.0
    assert n_acted >= 0


def test_select_act_threshold_falls_back_to_half_when_no_candidate_qualifies() -> None:
    # Too few rows to satisfy the min-acted floor (max(8, ...)) at any threshold => fallback.
    proba = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    r = np.array([0.5, -0.5, 0.5], dtype=np.float64)
    assert select_act_threshold(proba, r, DEFAULT_COST_PER_TRADE_R) == 0.5


def test_select_act_threshold_empty_is_fallback() -> None:
    empty = np.array([], dtype=np.float64)
    assert select_act_threshold(empty, empty, DEFAULT_COST_PER_TRADE_R) == 0.5


def test_precision_recall_at_no_acted_is_zero_precision() -> None:
    proba = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    r = np.array([1.0, -1.0, 1.0], dtype=np.float64)
    prec, recall, n_acted = precision_recall_at(proba, r, tau=0.9)
    assert n_acted == 0
    assert prec == 0.0
    assert recall == 0.0


def test_precision_recall_at_known_counts() -> None:
    # proba >= 0.5 acts on rows {0.5, 0.7, 0.9} -> R {1.0, -1.0, 2.0}: 2 of 3 profitable.
    # Profitable opportunities overall: rows with R>0 = {1.0(0.5), 2.0(0.9), 0.3(0.2)} = 3.
    proba = np.array([0.5, 0.7, 0.9, 0.2], dtype=np.float64)
    r = np.array([1.0, -1.0, 2.0, 0.3], dtype=np.float64)
    prec, recall, n_acted = precision_recall_at(proba, r, tau=0.5)
    assert n_acted == 3
    assert prec == 2.0 / 3.0
    assert recall == 2.0 / 3.0  # captured 2 of the 3 profitable opportunities


def test_select_act_threshold_is_deterministic() -> None:
    proba, r = _synth_signal_dataset(n=1500, seed=3)
    proba_is, r_is, _, _ = _split_half(proba, r)
    a = select_act_threshold(proba_is, r_is, DEFAULT_COST_PER_TRADE_R)
    b = select_act_threshold(proba_is, r_is, DEFAULT_COST_PER_TRADE_R)
    assert a == b


def test_precision_first_prefers_higher_precision_tau_over_expectancy_max() -> None:
    """A higher tau gives strictly higher (and still profitable) precision; pick it.

    Two distinct proba tiers, INTERLEAVED by index so both IS sub-folds (70/30 positional
    split) contain both tiers and the two-sub-fold robustness check can confirm the choice:
      * LOW tier (proba 0.55): big wins (+3.0) but every 4th is a -1.0 loser -> precision
        ~0.79 yet a *high* mean post-cost expectancy (the fat wins dominate).
      * HIGH tier (proba 0.95): always +0.5 -> precision 1.0 but a *lower* mean expectancy.

    The OLD expectancy-max selector maximizes mean post-cost R, so it grabs the LOW (less
    precise) tier; the new precision-first rule grabs the HIGH (more precise) tier.
    """
    n = 360
    proba_rows: list[float] = []
    r_rows: list[float] = []
    lo_count = 0
    for i in range(n):
        if i % 6 == 0:
            # HIGH tier: precise (always profitable), modest +0.5.
            proba_rows.append(0.95)
            r_rows.append(0.5)
        else:
            # LOW tier: fat +3.0 wins, every 4th a -1.0 loser (lower precision, higher mean).
            proba_rows.append(0.55)
            r_rows.append(-1.0 if lo_count % 4 == 0 else 3.0)
            lo_count += 1
    proba_is = np.asarray(proba_rows, dtype=np.float64)
    r_is = np.asarray(r_rows, dtype=np.float64)

    legacy = _legacy_expectancy_max_threshold(proba_is, r_is, DEFAULT_COST_PER_TRADE_R)
    tau = select_act_threshold(proba_is, r_is, DEFAULT_COST_PER_TRADE_R)

    # Legacy (expectancy-max) grabs the lower, fat-win tier; precision-first goes higher.
    assert legacy <= 0.55 + 1e-9
    assert tau > 0.55

    # And the precision-first choice is strictly more precise on this IS data.
    prec_new, _, _ = precision_recall_at(proba_is, r_is, tau)
    prec_legacy, _, _ = precision_recall_at(proba_is, r_is, legacy)
    assert prec_new > prec_legacy


def test_precise_but_unprofitable_threshold_is_rejected() -> None:
    """A perfectly 'precise' tier that is unprofitable after costs must be rejected.

    The HIGH tier (proba 0.95) has tiny +0.01 wins: precision is 1.0 but the cost-aware
    expectancy is +0.01 - cost < 0, so the profitability constraint must veto it. A lower,
    genuinely profitable tier exists, so the selector must move off the unprofitable tier
    (never returning the precise-but-unprofitable threshold).
    """
    # HIGH tier: 80 rows, all profitable pre-cost but +0.01 << cost -> unprofitable post-cost.
    n_high = 80
    proba_high = np.full(n_high, 0.95)
    r_high = np.full(n_high, 0.01)

    # LOW tier: 200 rows at proba 0.55, clearly profitable after costs (+1.0, all winners).
    n_low = 200
    proba_low = np.full(n_low, 0.55)
    r_low = np.full(n_low, 1.0)

    proba_is = np.concatenate([proba_high, proba_low])
    r_is = np.concatenate([r_high, r_low])

    tau = select_act_threshold(proba_is, r_is, DEFAULT_COST_PER_TRADE_R)

    # The unprofitable-after-cost high tier (tau in (0.55, 0.95]) must not be chosen.
    assert tau <= 0.55 + 1e-9
    # The chosen threshold's cost-aware expectancy on IS is strictly positive.
    acted = proba_is >= tau
    cost_aware = r_is[acted] - DEFAULT_COST_PER_TRADE_R
    assert float(np.mean(cost_aware)) > 0.0


def test_two_subfold_robustness_rejects_sliver_precise_threshold() -> None:
    """A tau precise+profitable only on the FIRST IS sub-fold must be rejected.

    WITHOUT the two-sub-fold guard, a single-fold precision-first selector picks the HIGH
    tier (proba 0.95): on the WHOLE IS half it is precise (more wins than losses) and
    profitable, so it looks great. But the IS half is split 70% / 30% by index and the high
    tier is profitable ONLY in the first 70% — it loses money in the last 30%. The robustness
    guard re-checks the candidate on the confirmation sub-fold, where the high tier's
    cost-aware expectancy is negative, so it is rejected. No other candidate isolates a
    profitable region on BOTH sub-folds here, so the selector falls back to 0.5.
    """
    n = 400
    cut = int(np.floor(0.70 * n))
    rows_proba: list[float] = []
    rows_r: list[float] = []
    lo = 0
    for i in range(n):
        in_first = i < cut
        if i % 2 == 0:
            # LOW tier 0.55: in fold A it has some losers (so its precision there is BELOW the
            # high tier's); in fold B it is always profitable. Can't be isolated from the high
            # tier (0.95 >= any tau the low tier clears), so it can't rescue the choice.
            rows_proba.append(0.55)
            rows_r.append((-1.0 if lo % 3 == 0 else 1.0) if in_first else 1.0)
            lo += 1
        else:
            # HIGH tier 0.95: precise + profitable in the first sub-fold, lossy in the second.
            rows_proba.append(0.95)
            rows_r.append(1.0 if in_first else -1.0)
    proba_is = np.asarray(rows_proba, dtype=np.float64)
    r_is = np.asarray(rows_r, dtype=np.float64)

    cut_is = int(np.floor(0.70 * proba_is.size))
    proba_a, r_a = proba_is[:cut_is], r_is[:cut_is]
    proba_b, r_b = proba_is[cut_is:], r_is[cut_is:]

    # On the SELECTION sub-fold (first 70%) alone, the high tier is the precision winner: it
    # is precise (1.0) AND profitable there, so a one-fold selector would pick it.
    from se_ml.server import _candidate_grid, _rank_precision_first

    grid = _candidate_grid(proba_is)
    min_a = max(8, int(np.ceil(0.10 * proba_a.size)))
    fold_a_ranked = _rank_precision_first(
        proba_a, r_a, grid, DEFAULT_COST_PER_TRADE_R, min_a
    )
    assert fold_a_ranked, "fold-A selector should find a candidate"
    high_tau = fold_a_ranked[0]
    assert high_tau > 0.55, "fold-A precision winner is the (overfit) high tier"

    # The high tier is profitable on the selection sub-fold but LOSES on the confirmation
    # sub-fold (its precision/profit was a 70%-sliver artifact).
    acted_a = proba_a >= high_tau
    acted_b = proba_b >= high_tau
    assert float(np.mean(r_a[acted_a] - DEFAULT_COST_PER_TRADE_R)) > 0.0
    assert float(np.mean(r_b[acted_b] - DEFAULT_COST_PER_TRADE_R)) <= 0.0

    # WITH the two-sub-fold guard, the high tier fails the confirmation fold and no other
    # candidate clears both sub-folds here, so the selector falls back to 0.5 (never the
    # sliver-precise high tier).
    tau = select_act_threshold(proba_is, r_is, DEFAULT_COST_PER_TRADE_R)
    assert tau != high_tau
    assert tau == 0.5, "robustness must reject the sliver-precise high tier and fall back"
