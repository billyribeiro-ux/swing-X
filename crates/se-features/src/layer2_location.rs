//! LAYER 2 — the location feature module.
//!
//! "Where decisions sit." Given the ticker's PIT-safe daily bars, it measures the
//! close's distance to the structures a swing trader cares about: the 50/200-day
//! moving averages, prior-day and prior-week extremes, recent swing pivots, an
//! anchored VWAP, and the close's position within the recent range. Level
//! distances are expressed as **ATR-normalized signed fractions** (positive =
//! close above the level) so they're comparable across names and vol regimes.
//!
//! It NEVER zero-fills: any distance whose inputs are missing (too few bars, zero
//! ATR) is simply skipped — the downstream gate handles missingness.
//!
//! Documented gaps (made explicit, never fabricated):
//!   * Volume profile (POC / VAH / VAL / HVN / LVN), GEX walls (call/put wall),
//!     and max-pain need intraday and/or options data which is
//!     PROPRIETARY/intraday — unavailable in v1, so these are SKIPPED (not
//!     zero-filled). See the marked block below.

use async_trait::async_trait;
use se_core::{AsOf, Bar, Feature, Layer, LeadTimeTag, Result};

use crate::indicators::{atr, sma};
use crate::module::{FeatureContext, FeatureModule};

/// Lookback (daily bars) — long enough for SMA200 plus an ATR window and headroom.
const LOOKBACK: i64 = 260;
/// ATR period used to normalize level distances.
const ATR_PERIOD: usize = 14;
/// Window (bars) for the percent-range-position and anchored-VWAP features.
const RANGE_WINDOW: usize = 20;
/// Fractal half-width for swing-pivot detection (k bars on each side).
const PIVOT_K: usize = 3;
/// N-bar range window for prior-week extremes (a trading week = 5 sessions).
const WEEK_BARS: usize = 5;
/// Rolling Donchian window (bars) for the N-bar high/low exhaustion distances.
const ROLLING_WINDOW: usize = 20;

/// The Layer-2 location feature module.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocationModule;

impl LocationModule {
    pub fn new() -> Self {
        LocationModule
    }
}

/// ATR-normalized signed distance of `close` from `level`: `(close - level) / atr`.
/// Positive when the close is above the level. `None` if `atr <= 0`.
fn atr_dist(close: f64, level: f64, atr: f64) -> Option<f64> {
    if atr <= 0.0 {
        return None;
    }
    Some((close - level) / atr)
}

/// The most recent pivot-high BEFORE the current bar: a bar whose high is strictly
/// greater than the `k` bars on each side. Scans from most-recent backwards so the
/// nearest confirmed pivot wins. Excludes the last `k` bars (unconfirmed).
fn last_swing_high(bars: &[Bar], k: usize) -> Option<f64> {
    if bars.len() < 2 * k + 1 {
        return None;
    }
    // Center index ranges over confirmed pivots only; iterate newest-first.
    for c in (k..bars.len() - k).rev() {
        let h = bars[c].high;
        let is_pivot = (1..=k).all(|d| h > bars[c - d].high && h > bars[c + d].high);
        if is_pivot {
            return Some(h);
        }
    }
    None
}

/// The most recent pivot-low BEFORE the current bar (mirror of [`last_swing_high`]).
fn last_swing_low(bars: &[Bar], k: usize) -> Option<f64> {
    if bars.len() < 2 * k + 1 {
        return None;
    }
    for c in (k..bars.len() - k).rev() {
        let l = bars[c].low;
        let is_pivot = (1..=k).all(|d| l < bars[c - d].low && l < bars[c + d].low);
        if is_pivot {
            return Some(l);
        }
    }
    None
}

/// Anchored VWAP over a slice: `sum(typical_price * volume) / sum(volume)`.
/// `None` if total volume is zero.
fn anchored_vwap(bars: &[Bar]) -> Option<f64> {
    let mut pv = 0.0;
    let mut vol = 0.0;
    for b in bars {
        pv += b.typical_price() * b.volume;
        vol += b.volume;
    }
    if vol <= 0.0 {
        return None;
    }
    Some(pv / vol)
}

/// Rolling (Donchian) high and low of the last `window` bars: the max `high` and
/// min `low` over the window ending at the decision bar (the decision bar itself is
/// included — these are extremes KNOWN AT the decision bar, not future ones).
/// `None` if there are no bars to scan.
fn rolling_high_low(bars: &[Bar], window: usize) -> Option<(f64, f64)> {
    if bars.is_empty() {
        return None;
    }
    let slice = &bars[bars.len().saturating_sub(window)..];
    let hi = slice.iter().fold(f64::MIN, |m, b| m.max(b.high));
    let lo = slice.iter().fold(f64::MAX, |m, b| m.min(b.low));
    if hi.is_finite() && lo.is_finite() {
        Some((hi, lo))
    } else {
        None
    }
}

/// Where `close` sits in the `[low, high]` range of the last `window` bars, in
/// `[0, 1]` (0 = at the window low, 1 = at the window high). `None` if degenerate.
fn pct_range_position(bars: &[Bar], close: f64, window: usize) -> Option<f64> {
    if bars.is_empty() {
        return None;
    }
    let slice = &bars[bars.len().saturating_sub(window)..];
    let hi = slice.iter().fold(f64::MIN, |m, b| m.max(b.high));
    let lo = slice.iter().fold(f64::MAX, |m, b| m.min(b.low));
    if hi <= lo {
        return None;
    }
    Some(((close - lo) / (hi - lo)).clamp(0.0, 1.0))
}

#[async_trait]
impl FeatureModule for LocationModule {
    fn layer(&self) -> Layer {
        Layer::Location
    }

    fn name(&self) -> &str {
        "layer2_location"
    }

    async fn compute(&self, ctx: &FeatureContext<'_>) -> Result<Vec<Feature>> {
        let pit = ctx.pit;
        let decision_ts = pit.decision_ts();
        let as_of = AsOf::new(decision_ts.inner());
        let mut out: Vec<Feature> = Vec::new();

        let push = |out: &mut Vec<Feature>, key: &str, value: f64| {
            if value.is_finite() {
                out.push(Feature::new(
                    key,
                    value,
                    Layer::Location,
                    as_of,
                    LeadTimeTag::EndOfDay,
                    "derived",
                ));
            }
        };

        let bars = pit.bars("daily", LOOKBACK).await?;
        if bars.len() < 2 {
            return Ok(out);
        }
        let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
        let last = *bars.last().unwrap();
        let close = last.close;
        let atr_v = atr(&bars, ATR_PERIOD);

        // --- Moving-average distances (ATR units) ------------------------
        if let Some(a) = atr_v {
            if let Some(ma50) = sma(&closes, 50) {
                if let Some(d) = atr_dist(close, ma50, a) {
                    push(&mut out, "location.dist_50dma", d);
                }
            }
            if let Some(ma200) = sma(&closes, 200) {
                if let Some(d) = atr_dist(close, ma200, a) {
                    push(&mut out, "location.dist_200dma", d);
                }
            }
        }

        // --- Prior-day extremes (ATR units) ------------------------------
        // The "prior day" is the bar before the decision bar.
        if let Some(a) = atr_v {
            let prior = &bars[bars.len() - 2];
            if let Some(d) = atr_dist(close, prior.high, a) {
                push(&mut out, "location.dist_prior_dh", d);
            }
            if let Some(d) = atr_dist(close, prior.low, a) {
                push(&mut out, "location.dist_prior_dl", d);
            }
        }

        // --- Prior-week extremes (high/low of the 5 bars before this one) -
        if let Some(a) = atr_v {
            if bars.len() > WEEK_BARS {
                let prior_week = &bars[bars.len() - 1 - WEEK_BARS..bars.len() - 1];
                let wh = prior_week.iter().fold(f64::MIN, |m, b| m.max(b.high));
                let wl = prior_week.iter().fold(f64::MAX, |m, b| m.min(b.low));
                if let Some(d) = atr_dist(close, wh, a) {
                    push(&mut out, "location.dist_prior_wh", d);
                }
                if let Some(d) = atr_dist(close, wl, a) {
                    push(&mut out, "location.dist_prior_wl", d);
                }
            }
        }

        // --- Swing pivots (k-bar fractal), ATR units ---------------------
        if let Some(a) = atr_v {
            if let Some(sh) = last_swing_high(&bars, PIVOT_K) {
                if let Some(d) = atr_dist(close, sh, a) {
                    push(&mut out, "location.swing_high_dist", d);
                }
            }
            if let Some(sl) = last_swing_low(&bars, PIVOT_K) {
                if let Some(d) = atr_dist(close, sl, a) {
                    push(&mut out, "location.swing_low_dist", d);
                }
            }
        }

        // --- Anchored VWAP from the start of the lookback window (ATR units) -
        if let Some(a) = atr_v {
            if let Some(vwap) = anchored_vwap(&bars) {
                if let Some(d) = atr_dist(close, vwap, a) {
                    push(&mut out, "location.anchored_vwap_dist", d);
                }
            }
        }

        // --- Position within the recent N-bar high-low range (0..1) ------
        if let Some(p) = pct_range_position(&bars, close, RANGE_WINDOW) {
            push(&mut out, "location.pct_range_position", p);
        }

        // --- Distance to the rolling N-bar (Donchian) high / low, ATR units ---
        // Mean-reversion / exhaustion context: how far the close sits above the
        // rolling low and below the rolling high. The window ends at the decision
        // bar (inclusive), so the extremes are known AT that bar — never future.
        if let Some(a) = atr_v {
            if let Some((hi, lo)) = rolling_high_low(&bars, ROLLING_WINDOW) {
                if let Some(d) = atr_dist(close, hi, a) {
                    push(&mut out, "location.dist_rolling_high", d);
                }
                if let Some(d) = atr_dist(close, lo, a) {
                    push(&mut out, "location.dist_rolling_low", d);
                }
            }
        }

        // --- Volume profile / GEX walls / max-pain -----------------------
        // PROPRIETARY/intraday — unavailable in v1. POC/VAH/VAL/HVN/LVN need
        // intraday volume-at-price; call/put walls + max-pain need the options
        // chain. These are intentionally SKIPPED (never zero-filled). When a
        // proprietary feed is wired (ctx.proprietary), the GEX walls would be
        // emitted here as location.call_wall_dist / location.put_wall_dist.

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};
    use se_core::Ticker;

    fn bar(ts_day: i64, o: f64, h: f64, l: f64, c: f64, v: f64) -> Bar {
        Bar {
            ticker: Ticker::SPY,
            ts: Utc.with_ymd_and_hms(2024, 1, 1, 21, 0, 0).unwrap() + Duration::days(ts_day),
            open: o,
            high: h,
            low: l,
            close: c,
            volume: v,
        }
    }

    #[test]
    fn atr_dist_sign() {
        // Close above level -> positive; below -> negative; scaled by ATR.
        assert!((atr_dist(110.0, 100.0, 5.0).unwrap() - 2.0).abs() < 1e-9);
        assert!((atr_dist(95.0, 100.0, 5.0).unwrap() + 1.0).abs() < 1e-9);
        assert_eq!(atr_dist(100.0, 90.0, 0.0), None);
    }

    #[test]
    fn swing_high_low_fractal() {
        // Build a series with a clear pivot high at index 3 (value 110) and a
        // pivot low at index 7 (value 90), k=2 confirmation on each side.
        let highs = [
            100.0, 102.0, 105.0, 110.0, 104.0, 101.0, 95.0, 92.0, 96.0, 98.0,
        ];
        let lows = [
            99.0, 101.0, 104.0, 109.0, 103.0, 100.0, 94.0, 90.0, 95.0, 97.0,
        ];
        let bars: Vec<Bar> = highs
            .iter()
            .zip(lows.iter())
            .enumerate()
            .map(|(i, (&h, &l))| bar(i as i64, l, h, l, (h + l) / 2.0, 1.0))
            .collect();
        // Most recent confirmed pivot high (k=2) is 110 at index 3.
        assert_eq!(last_swing_high(&bars, 2), Some(110.0));
        // Most recent confirmed pivot low (k=2) is 90 at index 7.
        assert_eq!(last_swing_low(&bars, 2), Some(90.0));
    }

    #[test]
    fn rolling_high_low_window() {
        // Highs/lows over 6 bars; with window=4 only the last 4 bars count.
        let bars: Vec<Bar> = [
            (110.0, 90.0),
            (130.0, 70.0), // extreme outside the last-4 window
            (115.0, 95.0),
            (112.0, 96.0),
            (118.0, 93.0),
            (116.0, 94.0),
        ]
        .iter()
        .enumerate()
        .map(|(i, &(h, l))| bar(i as i64, l, h, l, (h + l) / 2.0, 1.0))
        .collect();
        // Last 4 bars: highs {115,112,118,116} -> 118; lows {95,96,93,94} -> 93.
        assert_eq!(rolling_high_low(&bars, 4), Some((118.0, 93.0)));
        // Full window picks up the wide bar.
        assert_eq!(rolling_high_low(&bars, 6), Some((130.0, 70.0)));
        // No bars -> None.
        assert_eq!(rolling_high_low(&[], 4), None);
    }

    #[test]
    fn pct_range_position_endpoints() {
        let bars: Vec<Bar> = (0..5)
            .map(|i| bar(i, 100.0, 110.0, 90.0, 100.0, 1.0))
            .collect();
        // Close at the window low -> 0; at the high -> 1; midpoint -> 0.5.
        assert_eq!(pct_range_position(&bars, 90.0, 5), Some(0.0));
        assert_eq!(pct_range_position(&bars, 110.0, 5), Some(1.0));
        assert_eq!(pct_range_position(&bars, 100.0, 5), Some(0.5));
    }

    #[test]
    fn anchored_vwap_volume_weighted() {
        // Two bars: typical prices 100 and 200, volumes 1 and 3 -> VWAP = 175.
        let b1 = bar(0, 100.0, 100.0, 100.0, 100.0, 1.0);
        let b2 = bar(1, 200.0, 200.0, 200.0, 200.0, 3.0);
        let v = anchored_vwap(&[b1, b2]).unwrap();
        assert!((v - 175.0).abs() < 1e-9);
        // Zero total volume -> None.
        let z = bar(0, 1.0, 1.0, 1.0, 1.0, 0.0);
        assert_eq!(anchored_vwap(&[z]), None);
    }
}
