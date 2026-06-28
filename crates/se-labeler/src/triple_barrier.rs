//! Rust triple-barrier labeling (Lopez de Prado), ATR-sized, event-sampled.
//!
//! Mirrors `ml-worker/src/se_ml/labeling/triple_barrier.py` semantics EXACTLY. For each
//! entry event we place three barriers relative to the entry price:
//!
//!   * a **profit target** at `entry + side · target_atr_mult · ATR`,
//!   * a **stop loss**     at `entry − side · stop_atr_mult · ATR`,
//!   * a **time barrier**  `max_hold_bars` bars after entry.
//!
//! We walk bars forward from the entry bar and record the FIRST barrier touched. The
//! realized return is in **R units** (multiples of the stop distance = one R), signed by
//! `side` so a winning long and a winning short both yield positive R.
//!
//! Conservative intrabar ordering
//! ------------------------------
//! Within one bar we cannot observe the high/low sequence, so when *both* the target and
//! the stop lie inside a bar's `[low, high]` range, the **stop is deemed hit first**. This
//! avoids optimistic first-touch assumptions that would inflate backtest edge.
//!
//! As in the Python reference, on a target touch the return is exactly
//! `target_atr_mult / stop_atr_mult` R, and on a stop touch it is exactly `-1` R; on the
//! time barrier it is the signed close-to-close move divided by the risk (one R in price).

use chrono::{DateTime, Utc};
use se_core::{Bar, HorizonProfile, Side};

/// Which barrier was touched first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Profit target hit first.
    Target,
    /// Stop hit first (includes the conservative both-in-one-bar case).
    Stop,
    /// Time (vertical) barrier reached without a price barrier touch.
    Time,
}

impl Outcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Outcome::Target => "target",
            Outcome::Stop => "stop",
            Outcome::Time => "time",
        }
    }
}

/// A fully resolved label for one entry event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabelEvent {
    /// Entry timestamp (the entry bar's `ts`).
    pub entry_ts: DateTime<Utc>,
    /// Barrier-touch timestamp = `t1` (label-window end), used for CPCV purging.
    pub t1: DateTime<Utc>,
    pub side: Side,
    pub entry_px: f64,
    pub target_px: f64,
    pub stop_px: f64,
    pub outcome: Outcome,
    /// Realized return in R units, signed by `side`.
    pub ret_r: f64,
}

/// Errors from the triple-barrier geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LabelError {
    /// An entry index is out of range of the bar series.
    EntryOutOfRange(usize),
    /// ATR at entry was not strictly positive.
    NonPositiveAtr,
    /// `stop_atr_mult <= 0` (it defines the R unit and must be positive).
    NonPositiveStop,
    /// `max_hold_bars < 1`.
    ZeroMaxHold,
}

impl std::fmt::Display for LabelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LabelError::EntryOutOfRange(i) => write!(f, "entry index {i} out of range"),
            LabelError::NonPositiveAtr => write!(f, "ATR must be > 0 at entry"),
            LabelError::NonPositiveStop => {
                write!(f, "stop_atr_mult must be > 0 (defines the R unit)")
            }
            LabelError::ZeroMaxHold => write!(f, "max_hold_bars must be >= 1"),
        }
    }
}

impl std::error::Error for LabelError {}

impl From<LabelError> for se_core::Error {
    fn from(e: LabelError) -> Self {
        se_core::Error::Validation(e.to_string())
    }
}

/// Triple-barrier geometry parameterized by a [`HorizonProfile`].
///
/// The profile is the single source of barrier widths (`target_atr_mult`, `stop_atr_mult`)
/// and the time barrier (`max_hold_bars`) — and, downstream, the CPCV purge length. Keeping
/// these in one config object is what prevents labeling and cross-validation from desyncing.
#[derive(Debug, Clone, Copy)]
pub struct TripleBarrier {
    profile: HorizonProfile,
}

impl TripleBarrier {
    pub fn new(profile: HorizonProfile) -> Self {
        TripleBarrier { profile }
    }

    pub fn profile(&self) -> &HorizonProfile {
        &self.profile
    }

    /// Resolve a single bar: does it touch a barrier? `None` if neither is touched.
    ///
    /// The conservative rule: if BOTH barriers lie inside the bar range, the stop wins.
    fn resolve_bar(
        high: f64,
        low: f64,
        target_px: f64,
        stop_px: f64,
        side: Side,
    ) -> Option<Outcome> {
        let (hit_target, hit_stop) = match side {
            Side::Long => (high >= target_px, low <= stop_px),
            Side::Short => (low <= target_px, high >= stop_px),
        };
        if hit_target && hit_stop {
            return Some(Outcome::Stop); // adverse barrier first
        }
        if hit_stop {
            return Some(Outcome::Stop);
        }
        if hit_target {
            return Some(Outcome::Target);
        }
        None
    }

    /// Label a single entry: bars (ascending by `ts`), the entry bar index, side, and the
    /// per-entry ATR (the volatility unit that sizes the barriers).
    ///
    /// First touchable bar is the NEXT bar after entry — no look-ahead on the entry bar's
    /// own range (entry executes at the entry close).
    pub fn label_one(
        &self,
        bars: &[Bar],
        entry_idx: usize,
        side: Side,
        atr: f64,
    ) -> Result<LabelEvent, LabelError> {
        let target_mult = self.profile.target_atr_mult;
        let stop_mult = self.profile.stop_atr_mult;
        let max_hold = self.profile.max_hold_bars as usize;

        if max_hold < 1 {
            return Err(LabelError::ZeroMaxHold);
        }
        if stop_mult <= 0.0 {
            return Err(LabelError::NonPositiveStop);
        }
        if entry_idx >= bars.len() {
            return Err(LabelError::EntryOutOfRange(entry_idx));
        }
        // Reject non-positive and NaN ATR (the latter would silently break the barriers).
        if atr <= 0.0 || atr.is_nan() {
            return Err(LabelError::NonPositiveAtr);
        }

        let s = side.sign();
        let entry = bars[entry_idx];
        let entry_px = entry.close;
        let risk = stop_mult * atr; // one R, in price units
        let target_px = entry_px + s * target_mult * atr;
        let stop_px = entry_px - s * stop_mult * atr;

        let n = bars.len();
        let last_idx = (entry_idx + max_hold).min(n - 1);

        // First touchable bar is the next bar after entry, up to (and including) the time
        // barrier. `enumerate` keeps absolute bar indices via the `entry_idx + 1` offset.
        let mut resolved: Option<(usize, Outcome)> = None;
        for (offset, b) in bars[(entry_idx + 1)..=last_idx].iter().enumerate() {
            if let Some(out) = Self::resolve_bar(b.high, b.low, target_px, stop_px, side) {
                resolved = Some((entry_idx + 1 + offset, out));
                break;
            }
        }

        let (exit_idx, outcome, ret_r) = match resolved {
            None => {
                // Time barrier: exit at the close of the vertical-barrier bar.
                let exit_px = bars[last_idx].close;
                let ret_r = s * (exit_px - entry_px) / risk;
                (last_idx, Outcome::Time, ret_r)
            }
            Some((j, Outcome::Target)) => (j, Outcome::Target, target_mult / stop_mult),
            Some((j, Outcome::Stop)) => (j, Outcome::Stop, -1.0),
            // `resolve_bar` never returns Time.
            Some((j, Outcome::Time)) => (j, Outcome::Time, 0.0),
        };

        Ok(LabelEvent {
            entry_ts: entry.ts,
            t1: bars[exit_idx].ts,
            side,
            entry_px,
            target_px,
            stop_px,
            outcome,
            ret_r,
        })
    }

    /// Label a batch of entries. `entries` pairs an entry bar index with its side and ATR.
    pub fn label_events(
        &self,
        bars: &[Bar],
        entries: &[(usize, Side, f64)],
    ) -> Result<Vec<LabelEvent>, LabelError> {
        entries
            .iter()
            .map(|&(idx, side, atr)| self.label_one(bars, idx, side, atr))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use se_core::Ticker;

    fn bar(i: i64, o: f64, h: f64, l: f64, c: f64) -> Bar {
        Bar {
            ticker: Ticker::Spy,
            ts: Utc.timestamp_opt(1_600_000_000 + i * 86_400, 0).unwrap(),
            open: o,
            high: h,
            low: l,
            close: c,
            volume: 1.0,
        }
    }

    /// Profile with target=2, stop=1, max_hold=5, so target touch = +2R, stop = −1R.
    fn profile() -> HorizonProfile {
        let mut p = HorizonProfile::swing();
        p.max_hold_bars = 5;
        p.target_atr_mult = 2.0;
        p.stop_atr_mult = 1.0;
        p
    }

    #[test]
    fn long_hits_target_gives_plus_r() {
        // entry close 100, atr 1 -> target 102, stop 99. Next bar reaches 102.5 high.
        let bars = vec![
            bar(0, 100.0, 100.0, 100.0, 100.0),
            bar(1, 100.0, 102.5, 100.5, 101.0),
        ];
        let tb = TripleBarrier::new(profile());
        let ev = tb.label_one(&bars, 0, Side::Long, 1.0).unwrap();
        assert_eq!(ev.outcome, Outcome::Target);
        assert_eq!(ev.ret_r, 2.0);
        assert_eq!(ev.t1, bars[1].ts);
        assert_eq!(ev.target_px, 102.0);
        assert_eq!(ev.stop_px, 99.0);
    }

    #[test]
    fn long_hits_stop_gives_minus_one_r() {
        let bars = vec![
            bar(0, 100.0, 100.0, 100.0, 100.0),
            bar(1, 100.0, 100.4, 98.5, 99.0),
        ];
        let tb = TripleBarrier::new(profile());
        let ev = tb.label_one(&bars, 0, Side::Long, 1.0).unwrap();
        assert_eq!(ev.outcome, Outcome::Stop);
        assert_eq!(ev.ret_r, -1.0);
    }

    #[test]
    fn both_barriers_in_one_bar_resolves_to_stop() {
        // Bar spans 98..103: touches both target (102) and stop (99) -> conservative STOP.
        let bars = vec![
            bar(0, 100.0, 100.0, 100.0, 100.0),
            bar(1, 100.0, 103.0, 98.0, 100.0),
        ];
        let tb = TripleBarrier::new(profile());
        let ev = tb.label_one(&bars, 0, Side::Long, 1.0).unwrap();
        assert_eq!(ev.outcome, Outcome::Stop);
        assert_eq!(ev.ret_r, -1.0);
    }

    #[test]
    fn time_barrier_uses_signed_close_move_over_risk() {
        // No barrier touched within max_hold; exit at last bar close 100.5.
        // ret_r = side * (101 ... ) actually close move (100.5 - 100)/1 = 0.5R for long.
        let mut bars = vec![bar(0, 100.0, 100.0, 100.0, 100.0)];
        for i in 1..=5 {
            // stay strictly inside (99, 102): highs < 102, lows > 99.
            bars.push(bar(i, 100.0, 101.5, 99.5, 100.0 + 0.1 * i as f64));
        }
        let tb = TripleBarrier::new(profile());
        let ev = tb.label_one(&bars, 0, Side::Long, 1.0).unwrap();
        assert_eq!(ev.outcome, Outcome::Time);
        // last close = 100 + 0.1*5 = 100.5 -> 0.5R
        assert!((ev.ret_r - 0.5).abs() < 1e-9, "ret_r={}", ev.ret_r);
        assert_eq!(ev.t1, bars[5].ts);
    }

    #[test]
    fn short_hits_target_gives_plus_r() {
        // Short entry 100, atr 1 -> target 98 (price falls), stop 101.
        let bars = vec![
            bar(0, 100.0, 100.0, 100.0, 100.0),
            bar(1, 100.0, 100.5, 97.5, 98.0),
        ];
        let tb = TripleBarrier::new(profile());
        let ev = tb.label_one(&bars, 0, Side::Short, 1.0).unwrap();
        assert_eq!(ev.outcome, Outcome::Target);
        assert_eq!(ev.ret_r, 2.0);
        assert_eq!(ev.target_px, 98.0);
        assert_eq!(ev.stop_px, 101.0);
    }

    #[test]
    fn short_both_in_bar_resolves_to_stop() {
        let bars = vec![
            bar(0, 100.0, 100.0, 100.0, 100.0),
            bar(1, 100.0, 101.5, 97.5, 100.0),
        ];
        let tb = TripleBarrier::new(profile());
        let ev = tb.label_one(&bars, 0, Side::Short, 1.0).unwrap();
        assert_eq!(ev.outcome, Outcome::Stop);
        assert_eq!(ev.ret_r, -1.0);
    }

    #[test]
    fn non_positive_atr_is_error() {
        let bars = vec![
            bar(0, 100.0, 100.0, 100.0, 100.0),
            bar(1, 100.0, 100.0, 100.0, 100.0),
        ];
        let tb = TripleBarrier::new(profile());
        assert_eq!(
            tb.label_one(&bars, 0, Side::Long, 0.0),
            Err(LabelError::NonPositiveAtr)
        );
    }

    #[test]
    fn entry_out_of_range_is_error() {
        let bars = vec![bar(0, 100.0, 100.0, 100.0, 100.0)];
        let tb = TripleBarrier::new(profile());
        assert_eq!(
            tb.label_one(&bars, 5, Side::Long, 1.0),
            Err(LabelError::EntryOutOfRange(5))
        );
    }

    #[test]
    fn entry_bar_own_range_does_not_trigger_lookahead() {
        // The entry bar itself spans the target; it must be ignored (no look-ahead). The
        // next bar is flat, so the time barrier fires with ~0 return.
        let mut bars = vec![bar(0, 100.0, 105.0, 95.0, 100.0)];
        for i in 1..=5 {
            bars.push(bar(i, 100.0, 101.0, 99.5, 100.0));
        }
        let tb = TripleBarrier::new(profile());
        let ev = tb.label_one(&bars, 0, Side::Long, 1.0).unwrap();
        assert_eq!(ev.outcome, Outcome::Time);
    }

    #[test]
    fn ratio_return_respects_profile_mults() {
        // target=3, stop=1.5 -> target touch = 2R.
        let mut p = profile();
        p.target_atr_mult = 3.0;
        p.stop_atr_mult = 1.5;
        let bars = vec![
            bar(0, 100.0, 100.0, 100.0, 100.0),
            bar(1, 100.0, 104.5, 100.5, 104.0), // target = 100 + 3*1 = 103
        ];
        let tb = TripleBarrier::new(p);
        let ev = tb.label_one(&bars, 0, Side::Long, 1.0).unwrap();
        assert_eq!(ev.outcome, Outcome::Target);
        assert!((ev.ret_r - 2.0).abs() < 1e-9);
    }
}
