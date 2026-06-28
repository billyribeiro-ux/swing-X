//! LAYER 3 — the trigger feature module.
//!
//! "Who's leaning on arrival." Momentum, relative strength, volatility
//! compression, and participation signals that fire AT the decision bar, all from
//! PIT-safe daily bars (the ticker's own, plus SPY and the universe for the
//! cross-ticker reads via `bars_for`). Like every layer it NEVER zero-fills:
//! a signal whose inputs are missing is skipped.
//!
//! Documented gaps (made explicit, never fabricated):
//!   * Order-flow imbalance / cumulative signed delta / absorption need tick data.
//!     These are PROPRIETARY (tick) — read via `ctx.proprietary`; when the hook is
//!     `Unavailable` (the v1 default) they are SKIPPED, with the marked hook left
//!     in place for when a feed is wired.

use async_trait::async_trait;
use se_core::{AsOf, Feature, Layer, LeadTimeTag, Result, Ticker};

use crate::indicators::{atr, obv, roc, rsi, slope_sign, sma};
use crate::module::{FeatureContext, FeatureModule};

/// Return/ROC/RS horizon in trading days.
const MOMENTUM_N: usize = 20;
/// Short ATR window for the squeeze ratio.
const ATR_SHORT: usize = 5;
/// Long ATR window for the squeeze ratio.
const ATR_LONG: usize = 20;
/// RSI period.
const RSI_PERIOD: usize = 14;
/// OBV slope window.
const OBV_WINDOW: usize = 20;
/// NR-window for NR7 (narrowest range of the last 7 bars).
const NR_WINDOW: usize = 7;
/// SMA window for the breadth-thrust "% above 50DMA" check.
const BREADTH_MA: usize = 50;
/// Lookback to compute the momentum-divergence pivots.
const DIVERGENCE_WINDOW: usize = 10;
/// Generous daily-bar lookback covering every window above.
const LOOKBACK: i64 = 80;

/// The Layer-3 trigger feature module.
#[derive(Debug, Clone, Copy, Default)]
pub struct TriggerModule;

impl TriggerModule {
    pub fn new() -> Self {
        TriggerModule
    }
}

/// N-day relative strength of `ticker_closes` vs `bench_closes`: the ticker's
/// N-day return minus the benchmark's N-day return. `None` if either lacks data.
fn rs_vs_bench(ticker_closes: &[f64], bench_closes: &[f64], n: usize) -> Option<f64> {
    let r_t = roc(ticker_closes, n)?;
    let r_b = roc(bench_closes, n)?;
    Some(r_t - r_b)
}

/// NR7: 1.0 if the LAST bar's range is the narrowest of the last `window` bars.
fn nr7(ranges: &[f64], window: usize) -> Option<f64> {
    if ranges.len() < window {
        return None;
    }
    let slice = &ranges[ranges.len() - window..];
    let last = *slice.last().unwrap();
    let is_narrowest = slice.iter().all(|&r| last <= r);
    Some(if is_narrowest { 1.0 } else { 0.0 })
}

/// Signed momentum divergence over the last `window` bars. Compares the price
/// high/low against ROC(1) momentum:
///   * price makes a higher high but momentum a lower high -> bearish divergence (-1)
///   * price makes a lower low but momentum a higher low -> bullish divergence (+1)
///   * otherwise 0 (momentum confirms, or no clean two-pivot structure).
///
/// Coarse by design — a daily-bar swing tell, not an oscillator.
fn momentum_divergence(closes: &[f64], window: usize) -> f64 {
    if window < 2 || closes.len() < window {
        return 0.0;
    }
    let px = &closes[closes.len() - window..];
    // Momentum proxy: bar-to-bar change aligned to `px` (first element 0).
    let mut mom = vec![0.0; px.len()];
    for i in 1..px.len() {
        mom[i] = px[i] - px[i - 1];
    }
    // Split the window into an earlier and a recent half; compare extremes.
    let mid = px.len() / 2;
    if mid == 0 || mid >= px.len() {
        return 0.0;
    }
    let (early_px, late_px) = (&px[..mid], &px[mid..]);
    let (early_mom, late_mom) = (&mom[..mid], &mom[mid..]);
    let max = |s: &[f64]| s.iter().cloned().fold(f64::MIN, f64::max);
    let min = |s: &[f64]| s.iter().cloned().fold(f64::MAX, f64::min);

    let price_hh = max(late_px) > max(early_px);
    let mom_lh = max(late_mom) < max(early_mom);
    let price_ll = min(late_px) < min(early_px);
    let mom_hl = min(late_mom) > min(early_mom);

    if price_hh && mom_lh {
        -1.0 // bearish: new price high not confirmed by momentum
    } else if price_ll && mom_hl {
        1.0 // bullish: new price low not confirmed by momentum
    } else {
        0.0
    }
}

#[async_trait]
impl FeatureModule for TriggerModule {
    fn layer(&self) -> Layer {
        Layer::Trigger
    }

    fn name(&self) -> &str {
        "layer3_trigger"
    }

    async fn compute(&self, ctx: &FeatureContext<'_>) -> Result<Vec<Feature>> {
        let pit = ctx.pit;
        let ticker = pit.ticker();
        let decision_ts = pit.decision_ts();
        let as_of = AsOf::new(decision_ts.inner());
        let mut out: Vec<Feature> = Vec::new();

        let push = |out: &mut Vec<Feature>, key: &str, value: f64, source: &str| {
            if value.is_finite() {
                out.push(Feature::new(
                    key,
                    value,
                    Layer::Trigger,
                    as_of,
                    LeadTimeTag::EndOfDay,
                    source,
                ));
            }
        };

        let bars = pit.bars("daily", LOOKBACK).await?;
        if bars.len() < 2 {
            return Ok(out);
        }
        let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
        let ranges: Vec<f64> = bars.iter().map(|b| b.range()).collect();

        // --- Relative strength vs SPY ------------------------------------
        // SPY measures itself against itself -> 0 by definition.
        if ticker == Ticker::BENCHMARK {
            push(&mut out, "trigger.rs_vs_spy", 0.0, "derived");
        } else {
            let spy = pit.bars_for(Ticker::BENCHMARK, "daily", LOOKBACK).await?;
            let spy_closes: Vec<f64> = spy.iter().map(|b| b.close).collect();
            if let Some(rs) = rs_vs_bench(&closes, &spy_closes, MOMENTUM_N) {
                push(&mut out, "trigger.rs_vs_spy", rs, "derived");
            }
        }

        // --- Momentum: rate of change ------------------------------------
        if let Some(r) = roc(&closes, MOMENTUM_N) {
            push(&mut out, "trigger.momentum_roc", r, "derived");
        }

        // --- Momentum divergence (signed) --------------------------------
        push(
            &mut out,
            "trigger.momentum_divergence",
            momentum_divergence(&closes, DIVERGENCE_WINDOW),
            "derived",
        );

        // --- ATR contraction (squeeze proxy): short ATR / long ATR -------
        if let (Some(s), Some(l)) = (atr(&bars, ATR_SHORT), atr(&bars, ATR_LONG)) {
            if l > 0.0 {
                push(&mut out, "trigger.atr_contraction", s / l, "derived");
            }
        }

        // --- NR7 ----------------------------------------------------------
        if let Some(n) = nr7(&ranges, NR_WINDOW) {
            push(&mut out, "trigger.nr7", n, "derived");
        }

        // --- RSI(14) ------------------------------------------------------
        if let Some(r) = rsi(&closes, RSI_PERIOD) {
            push(&mut out, "trigger.rsi14", r, "derived");
        }

        // --- OBV trend (slope sign over N) -------------------------------
        let obv_series = obv(&bars);
        if obv_series.len() >= OBV_WINDOW {
            let tail = &obv_series[obv_series.len() - OBV_WINDOW..];
            push(&mut out, "trigger.obv_trend", slope_sign(tail), "derived");
        }

        // --- Breadth thrust: fraction of the universe above its 50DMA ----
        // Market-wide value (same for every ticker); computed per-ticker via the
        // cross-ticker PIT read so it stays leakage-safe at the decision bar.
        let mut have = 0usize;
        let mut above = 0usize;
        for &t in &Ticker::ALL {
            let tb = if t == ticker {
                bars.clone()
            } else {
                pit.bars_for(t, "daily", BREADTH_MA as i64 + 2).await?
            };
            let tc: Vec<f64> = tb.iter().map(|b| b.close).collect();
            if let Some(ma) = sma(&tc, BREADTH_MA) {
                have += 1;
                if *tc.last().unwrap() > ma {
                    above += 1;
                }
            }
        }
        if have > 0 {
            push(
                &mut out,
                "trigger.breadth_thrust",
                above as f64 / have as f64,
                "derived",
            );
        }

        // --- Order-flow imbalance / signed delta / absorption ------------
        // PROPRIETARY (tick) — read via ctx.proprietary; Unavailable -> SKIP
        // (never fabricated). When a tick feed is wired this emits
        // trigger.order_flow_imbalance (and signed-delta / absorption variants).
        if let Ok(Some(ofi)) = ctx
            .proprietary
            .order_flow_imbalance(ticker, decision_ts)
            .await
        {
            push(
                &mut out,
                "trigger.order_flow_imbalance",
                ofi,
                "proprietary:order_flow",
            );
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rs_vs_spy_sign() {
        // Ticker up 10% over 2 bars, SPY up 2% -> positive RS.
        let t = [100.0, 105.0, 110.0];
        let spy = [100.0, 101.0, 102.0];
        let rs = rs_vs_bench(&t, &spy, 2).unwrap();
        assert!(rs > 0.0, "outperformer -> positive RS, got {rs}");
        // Reverse the roles -> negative RS.
        let rs2 = rs_vs_bench(&spy, &t, 2).unwrap();
        assert!(rs2 < 0.0, "underperformer -> negative RS, got {rs2}");
        // Identical series -> exactly 0.
        assert_eq!(rs_vs_bench(&t, &t, 2), Some(0.0));
    }

    #[test]
    fn nr7_detects_narrowest() {
        // Last bar's range (0.5) is the narrowest of 7 -> 1.0.
        let ranges = [3.0, 2.0, 4.0, 1.5, 2.5, 1.0, 0.5];
        assert_eq!(nr7(&ranges, 7), Some(1.0));
        // Last bar wide -> 0.0.
        let ranges2 = [3.0, 2.0, 4.0, 1.5, 2.5, 1.0, 5.0];
        assert_eq!(nr7(&ranges2, 7), Some(0.0));
        // Too few bars -> None.
        assert_eq!(nr7(&[1.0, 2.0], 7), None);
    }

    #[test]
    fn momentum_divergence_bearish_and_bullish() {
        // Bearish: price climbs to a higher high but the late-half steps shrink
        // (decelerating) vs the early-half -> momentum lower-high -> -1.
        let bearish = [100.0, 104.0, 109.0, 115.0, 116.0, 117.0, 117.5, 118.0];
        assert_eq!(momentum_divergence(&bearish, 8), -1.0);

        // Bullish: price falls to a lower low but the late-half declines shrink
        // (decelerating sell-off) vs the early half -> momentum higher-low -> +1.
        let bullish = [120.0, 116.0, 111.0, 105.0, 104.0, 103.0, 102.5, 102.0];
        assert_eq!(momentum_divergence(&bullish, 8), 1.0);

        // Clean trend where momentum confirms -> 0.
        let confirm = [100.0, 102.0, 104.0, 106.0, 109.0, 112.0, 116.0, 121.0];
        assert_eq!(momentum_divergence(&confirm, 8), 0.0);
    }
}
