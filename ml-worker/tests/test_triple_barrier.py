"""Golden tests for triple-barrier labeling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from se_ml.labeling.triple_barrier import Outcome, label_events


def _bars(closes, highs=None, lows=None, atr=1.0):
    n = len(closes)
    highs = highs if highs is not None else closes
    lows = lows if lows is not None else closes
    return pd.DataFrame(
        {
            "ts": pd.date_range("2021-01-01", periods=n, freq="D"),
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "atr": [atr] * n,
        }
    )


def test_long_hits_target_first():
    # entry at close=100, atr=1, target_mult=2 -> target 102, stop_mult=1 -> stop 99.
    closes = [100, 100.5, 102.5, 100]
    highs = [100, 101.0, 103.0, 100]
    lows = [100, 100.0, 102.0, 100]
    bars = _bars(closes, highs, lows, atr=1.0)
    events = pd.DataFrame({"ts": [bars["ts"].iloc[0]], "side": [1]})

    out = label_events(bars, events, target_mult=2.0, stop_mult=1.0, max_hold=3)
    row = out.iloc[0]
    assert row["outcome"] == Outcome.TARGET.value
    # target distance is 2 ATR, stop (R) is 1 ATR -> +2R.
    assert np.isclose(row["ret_r"], 2.0)
    assert row["exit_idx"] == 2  # the bar whose high reached 103


def test_long_hits_stop_first():
    closes = [100, 99.5, 98.5, 100]
    highs = [100, 100.0, 99.0, 100]
    lows = [100, 99.0, 98.0, 100]
    bars = _bars(closes, highs, lows, atr=1.0)
    events = pd.DataFrame({"ts": [bars["ts"].iloc[0]], "side": [1]})

    out = label_events(bars, events, target_mult=2.0, stop_mult=1.0, max_hold=3)
    row = out.iloc[0]
    assert row["outcome"] == Outcome.STOP.value
    assert np.isclose(row["ret_r"], -1.0)
    assert row["exit_idx"] == 1


def test_conservative_intrabar_stop_wins_when_both_touched():
    # A single bar straddles BOTH target (102) and stop (99): conservative -> STOP.
    closes = [100, 100]
    highs = [100, 103]
    lows = [100, 98]
    bars = _bars(closes, highs, lows, atr=1.0)
    events = pd.DataFrame({"ts": [bars["ts"].iloc[0]], "side": [1]})

    out = label_events(bars, events, target_mult=2.0, stop_mult=1.0, max_hold=1)
    assert out.iloc[0]["outcome"] == Outcome.STOP.value
    assert np.isclose(out.iloc[0]["ret_r"], -1.0)


def test_time_barrier_when_no_touch():
    closes = [100, 100.2, 100.4, 100.6]
    bars = _bars(closes, atr=1.0)  # high==low==close, narrow drift, never touches +-
    events = pd.DataFrame({"ts": [bars["ts"].iloc[0]], "side": [1]})

    out = label_events(bars, events, target_mult=5.0, stop_mult=5.0, max_hold=3)
    row = out.iloc[0]
    assert row["outcome"] == Outcome.TIME.value
    # exit at close of bar 3 = 100.6; entry 100; risk = 5 ATR = 5 -> 0.6/5.
    assert np.isclose(row["ret_r"], (100.6 - 100.0) / 5.0)
    assert row["exit_idx"] == 3


def test_short_side_target():
    # short: target below entry. entry 100, target_mult 2 -> 98, stop 101.
    closes = [100, 99.0, 97.5, 100]
    highs = [100, 99.5, 98.0, 100]
    lows = [100, 98.5, 97.0, 100]
    bars = _bars(closes, highs, lows, atr=1.0)
    events = pd.DataFrame({"ts": [bars["ts"].iloc[0]], "side": [-1]})

    out = label_events(bars, events, target_mult=2.0, stop_mult=1.0, max_hold=3)
    row = out.iloc[0]
    assert row["outcome"] == Outcome.TARGET.value
    assert np.isclose(row["ret_r"], 2.0)


def test_entry_bar_not_used_for_touch():
    # The entry bar itself has an extreme high but must NOT count (no look-ahead on entry).
    closes = [100, 100.1]
    highs = [200, 100.1]  # entry bar high is huge but ignored
    lows = [100, 100.0]
    bars = _bars(closes, highs, lows, atr=1.0)
    events = pd.DataFrame({"ts": [bars["ts"].iloc[0]], "side": [1]})
    out = label_events(bars, events, target_mult=2.0, stop_mult=1.0, max_hold=1)
    # Should be TIME (bar 1 doesn't reach 102), not TARGET off the entry bar.
    assert out.iloc[0]["outcome"] == Outcome.TIME.value


def test_matches_rust_labeler_convention_across_all_outcomes_and_sides():
    """Cross-language parity: this Python REFERENCE labeler must agree with the authoritative
    Rust `se-labeler` on every outcome x side. The scenarios + expected (outcome, R) mirror the
    Rust sweep `label_one_equals_from_profile_across_all_outcomes_and_sides` exactly
    (target_mult=2, stop_mult=1, atr=1, entry close=100 => long target 102/stop 99, short
    target 98/stop 101; time exits realize signed close-move / R).
    """
    # (name, side, closes, highs, lows, expected_outcome, expected_ret_r)
    scenarios = [
        ("long_target", 1, [100, 101.0], [100, 102.5], [100, 100.5], Outcome.TARGET, 2.0),
        ("long_stop", 1, [100, 99.2], [100, 100.5], [100, 98.5], Outcome.STOP, -1.0),
        (
            "long_time",
            1,
            [100, 100.2, 100.3, 100.4, 100.4, 100.5],
            [100, 101, 101, 101, 101, 101],
            [100, 99.5, 99.5, 99.5, 99.5, 99.5],
            Outcome.TIME,
            0.5,  # (100.5 - 100) / 1
        ),
        ("short_target", -1, [100, 98.0], [100, 100.5], [100, 97.5], Outcome.TARGET, 2.0),
        ("short_stop", -1, [100, 100.8], [100, 101.5], [100, 99.5], Outcome.STOP, -1.0),
        (
            "short_time",
            -1,
            [100, 99.8, 99.7, 99.6, 99.6, 99.5],
            [100, 100.5, 100.5, 100.5, 100.5, 100.5],
            [100, 99, 99, 99, 99, 99],
            Outcome.TIME,
            0.5,  # -(99.5 - 100) / 1
        ),
    ]
    for name, side, closes, highs, lows, want_outcome, want_r in scenarios:
        bars = _bars(closes, highs, lows, atr=1.0)
        events = pd.DataFrame({"ts": [bars["ts"].iloc[0]], "side": [side]})
        out = label_events(bars, events, target_mult=2.0, stop_mult=1.0, max_hold=5)
        row = out.iloc[0]
        assert row["outcome"] == want_outcome.value, f"{name}: outcome"
        assert np.isclose(row["ret_r"], want_r), f"{name}: ret_r={row['ret_r']}"
