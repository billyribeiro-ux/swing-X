"""Triple-barrier labeling (Lopez de Prado), ATR-sized, event-sampled.

For each entry event we place three barriers relative to the entry price:

  * a **profit target**  at ``entry + side · target_mult · ATR``,
  * a **stop loss**      at ``entry - side · stop_mult · ATR``,
  * a **vertical (time)** barrier ``max_hold`` bars after entry.

We walk the bars forward from the entry bar and record the FIRST barrier touched. The
realized return is expressed in **R units** (multiples of the initial risk = the stop
distance), signed by ``side`` so that a winning long and a winning short both yield a
positive R.

Conservative intrabar ordering
------------------------------
Within a single bar we cannot observe the high/low sequence, so when *both* the target
and the stop lie inside a bar's [low, high] range we resolve the touch **against** the
position — the stop is deemed hit first. This avoids optimistic first-touch assumptions
that would inflate backtest edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd


class Outcome(StrEnum):
    TARGET = "target"
    STOP = "stop"
    TIME = "time"


@dataclass(frozen=True)
class BarrierLabel:
    event_idx: int          # index into the bars frame of the entry bar
    entry_ts: pd.Timestamp  # entry timestamp
    exit_idx: int           # index into the bars frame where a barrier was touched
    t1: pd.Timestamp        # barrier-touch timestamp (label window end)
    outcome: Outcome
    ret_r: float            # realized return in R units (signed by side)


def _resolve_bar(
    high: float,
    low: float,
    target_px: float,
    stop_px: float,
    side: int,
) -> Outcome | None:
    """Resolve whether a single bar touches a barrier; None if neither is touched.

    ``side`` is +1 for long, -1 for short. The conservative rule: if both barriers are
    inside the bar range, the stop wins.
    """
    if side == 1:
        hit_target = high >= target_px
        hit_stop = low <= stop_px
    else:
        hit_target = low <= target_px
        hit_stop = high >= stop_px

    if hit_target and hit_stop:
        return Outcome.STOP  # conservative: adverse barrier first
    if hit_stop:
        return Outcome.STOP
    if hit_target:
        return Outcome.TARGET
    return None


def label_events(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    target_mult: float,
    stop_mult: float,
    max_hold: int,
    *,
    ts_col: str = "ts",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_col: str = "atr",
) -> pd.DataFrame:
    """Apply triple-barrier labeling to a set of entry events.

    Parameters
    ----------
    bars
        OHLC(+ATR) bar frame, sorted ascending by ``ts``. Must contain ``ts``, ``high``,
        ``low``, ``close`` and an ``atr`` column (the per-bar volatility unit).
    events
        Entry events with columns ``ts`` (entry bar timestamp, must align to a bar) and
        ``side`` (+1 long / -1 short). An optional ``atr`` column on the event overrides
        the bar ATR at entry.
    target_mult, stop_mult
        ATR multiples for the profit target and stop distance.
    max_hold
        Vertical-barrier horizon in bars (inclusive count of forward bars).

    Returns
    -------
    pandas.DataFrame
        One row per event with columns:
        ``event_idx, entry_ts, exit_idx, t1, outcome, ret_r``.
    """
    if max_hold < 1:
        raise ValueError("max_hold must be >= 1")
    if stop_mult <= 0:
        raise ValueError("stop_mult must be > 0 (defines the R unit)")

    b = bars.reset_index(drop=True)
    ts_to_idx = {ts: i for i, ts in enumerate(b[ts_col].to_numpy())}
    highs = b[high_col].to_numpy(dtype=np.float64)
    lows = b[low_col].to_numpy(dtype=np.float64)
    closes = b[close_col].to_numpy(dtype=np.float64)
    atrs = b[atr_col].to_numpy(dtype=np.float64)
    n_bars = len(b)

    rows: list[BarrierLabel] = []
    for _, ev in events.iterrows():
        ev_ts = ev[ts_col]
        if ev_ts not in ts_to_idx:
            raise ValueError(f"event ts {ev_ts!r} does not align to any bar")
        i0 = ts_to_idx[ev_ts]
        side = int(ev["side"])
        if side not in (1, -1):
            raise ValueError("side must be +1 (long) or -1 (short)")

        entry_px = closes[i0]
        has_ev_atr = "atr" in events.columns and not pd.isna(ev.get("atr"))
        atr = float(ev["atr"]) if has_ev_atr else atrs[i0]
        if atr <= 0:
            raise ValueError("ATR must be > 0 at entry")

        risk = stop_mult * atr  # one R, in price units
        target_px = entry_px + side * target_mult * atr
        stop_px = entry_px - side * stop_mult * atr

        last_idx = min(i0 + max_hold, n_bars - 1)
        resolved: tuple[int, Outcome] | None = None
        # Walk forward; first touchable bar is the NEXT bar after entry (no look-ahead on
        # the entry bar's own range — entry executes at the entry close).
        for j in range(i0 + 1, last_idx + 1):
            out = _resolve_bar(highs[j], lows[j], target_px, stop_px, side)
            if out is not None:
                resolved = (j, out)
                break

        if resolved is None:
            # Time barrier: exit at the close of the vertical barrier bar.
            exit_idx = last_idx
            outcome = Outcome.TIME
            exit_px = closes[exit_idx]
            ret_r = side * (exit_px - entry_px) / risk
        else:
            exit_idx, outcome = resolved
            # +R multiple on a target touch; a stop touch loses exactly one R.
            ret_r = target_mult / stop_mult if outcome is Outcome.TARGET else -1.0

        rows.append(
            BarrierLabel(
                event_idx=i0,
                entry_ts=ev_ts,
                exit_idx=exit_idx,
                t1=b[ts_col].iloc[exit_idx],
                outcome=outcome,
                ret_r=float(ret_r),
            )
        )

    return pd.DataFrame(
        {
            "event_idx": [r.event_idx for r in rows],
            "entry_ts": [r.entry_ts for r in rows],
            "exit_idx": [r.exit_idx for r in rows],
            "t1": [r.t1 for r in rows],
            "outcome": [r.outcome.value for r in rows],
            "ret_r": [r.ret_r for r in rows],
        }
    )
