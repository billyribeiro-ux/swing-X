//! LAYER 1 — the regime feature module.
//!
//! The conditioner: continuation-vs-fade. It assembles the observable subset of
//! the regime taxonomy from PIT-safe data (the ticker's own bars plus the
//! market-wide macro store) and emits one [`Feature`] per signal it can actually
//! compute. It NEVER zero-fills: an unavailable input (e.g. SKEW, or the FRED
//! credit/liquidity complex when `FRED_API_KEY` is empty) is simply skipped, and
//! the downstream classifier handles the missingness.
//!
//! Documented proxies (gaps made explicit, never fabricated):
//!   * `regime.vrp` uses **VIX as the implied-vol proxy** (VIX − rv20). A true
//!     vol-risk-premium would use the full IV surface, which is unavailable in v1.
//!   * Dealer gamma / charm / vanna / DIX are PROPRIETARY: read via
//!     `ctx.proprietary`; when the hook is `Unavailable` they are skipped and the
//!     classifier approximates short/long-gamma from the vol regime + VVIX.

use async_trait::async_trait;
use se_core::{AsOf, Feature, Layer, LeadTimeTag, Result};

use crate::indicators::{adx, realized_vol};
use crate::module::{FeatureContext, FeatureModule};

/// Number of trading days used to annualize realized vol.
const RV_LOOKBACK: usize = 20;
/// History window (~1y of sessions) for the realized-vol percentile.
const RV_PCTL_WINDOW: i64 = 252;
/// Macro history window for trend features (DXY, copper/gold, oil).
const TREND_WINDOW: i64 = 21;
/// ADX period (Wilder default).
const ADX_PERIOD: usize = 14;

/// The Layer-1 regime feature module.
#[derive(Debug, Clone, Copy, Default)]
pub struct RegimeModule;

impl RegimeModule {
    pub fn new() -> Self {
        RegimeModule
    }
}

/// Percentile (in [0,1]) of `x` within `hist` — fraction of history at or below x.
fn percentile_of(hist: &[f64], x: f64) -> Option<f64> {
    if hist.is_empty() {
        return None;
    }
    let below = hist.iter().filter(|&&v| v <= x).count();
    Some(below as f64 / hist.len() as f64)
}

/// Percent change of a series from first to last point (e.g. 20d DXY trend).
fn pct_change(series: &[f64]) -> Option<f64> {
    if series.len() < 2 {
        return None;
    }
    let first = *series.first().unwrap();
    let last = *series.last().unwrap();
    if first == 0.0 {
        return None;
    }
    Some(last / first - 1.0)
}

#[async_trait]
impl FeatureModule for RegimeModule {
    fn layer(&self) -> Layer {
        Layer::Regime
    }

    fn name(&self) -> &str {
        "layer1_regime"
    }

    async fn compute(&self, ctx: &FeatureContext<'_>) -> Result<Vec<Feature>> {
        let pit = ctx.pit;
        let decision_ts = pit.decision_ts();
        let as_of = AsOf::new(decision_ts.inner());
        let mut out: Vec<Feature> = Vec::new();

        // Helper builders. `derived` = computed here; passthrough series keep
        // their native provenance source ("fmp"/"fred").
        let mk = |key: &str, value: f64, source: &str| {
            Feature::new(
                key,
                value,
                Layer::Regime,
                as_of,
                LeadTimeTag::EndOfDay,
                source,
            )
        };
        let push = |out: &mut Vec<Feature>, key: &str, value: f64, source: &str| {
            if value.is_finite() {
                out.push(mk(key, value, source));
            }
        };

        // --- VIX term structure ------------------------------------------
        // Read each leg PIT-safely from the macro store. Skip a ratio if any leg
        // is missing (never substitute a default).
        let vix = pit.macro_value_as_of("vix").await?;
        let vix9d = pit.macro_value_as_of("vix9d").await?;
        let vix3m = pit.macro_value_as_of("vix3m").await?;
        let vvix = pit.macro_value_as_of("vvix").await?;

        if let (Some(v9), Some(v)) = (vix9d, vix) {
            if v > 0.0 {
                // < 1 backwardation (near-term fear), > 1 contango (calm).
                push(&mut out, "regime.vix9d_vix", v9 / v, "derived");
            }
        }
        if let (Some(v), Some(v3)) = (vix, vix3m) {
            if v3 > 0.0 {
                // < 1 contango (calm), > 1 backwardation (stress).
                push(&mut out, "regime.vix_vix3m", v / v3, "derived");
            }
        }
        if let Some(vv) = vvix {
            // VVIX level (vol-of-vol); passthrough.
            push(&mut out, "regime.vvix", vv, "fmp");
        }
        // SKEW is unavailable on this tier -> intentionally skipped.

        // --- Realized-vol regime -----------------------------------------
        // Need history for both rv20 and its ~1y percentile.
        let bars = pit
            .bars("daily", RV_PCTL_WINDOW + RV_LOOKBACK as i64 + 5)
            .await?;
        if bars.len() > RV_LOOKBACK {
            let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
            let rv20 = realized_vol(&closes, RV_LOOKBACK);
            push(&mut out, "regime.rv20", rv20, "derived");

            // Rolling rv20 history -> percentile of the current rv20.
            let mut rv_hist: Vec<f64> = Vec::new();
            // Compute rv20 ending at each bar we have enough data for.
            for end in (RV_LOOKBACK + 1)..=closes.len() {
                let window = &closes[..end];
                rv_hist.push(realized_vol(window, RV_LOOKBACK));
            }
            if let Some(p) = percentile_of(&rv_hist, rv20) {
                push(&mut out, "regime.rv_percentile", p, "derived");
            }

            // ADX (trend strength).
            if let Some(a) = adx(&bars, ADX_PERIOD) {
                push(&mut out, "regime.adx", a, "derived");
            }

            // --- Vol risk premium (PROXY) --------------------------------
            // VRP = implied vol − realized vol. We use VIX (in vol points, i.e.
            // percent) as the IV proxy; rv20 is a fraction, so scale to points.
            if let Some(v) = vix {
                let vrp = v - rv20 * 100.0;
                push(&mut out, "regime.vrp", vrp, "derived");
            }
        }

        // --- Credit (FRED; skipped entirely when FRED unavailable) -------
        let hy = pit.macro_value_as_of("hy_oas").await?;
        let ig = pit.macro_value_as_of("ig_oas").await?;
        if let Some(h) = hy {
            push(&mut out, "regime.hy_oas", h, "fred");
        }
        if let Some(i) = ig {
            push(&mut out, "regime.ig_oas", i, "fred");
        }
        if let (Some(h), Some(i)) = (hy, ig) {
            if i > 0.0 {
                push(&mut out, "regime.hy_ig_ratio", h / i, "derived");
            }
        }

        // --- Net liquidity = WALCL − WTREGEN − RRPONTSYD (FRED) ----------
        let walcl = pit.macro_value_as_of("fed_balance_sheet").await?;
        let tga = pit.macro_value_as_of("tga").await?;
        let rrp = pit.macro_value_as_of("reverse_repo").await?;
        if let (Some(w), Some(t), Some(r)) = (walcl, tga, rrp) {
            push(&mut out, "regime.net_liquidity", w - t - r, "derived");
        }

        // --- Cross-asset --------------------------------------------------
        // DXY 20d % change.
        let dxy_hist = pit.macro_history("dxy", TREND_WINDOW).await?;
        let dxy_vals: Vec<f64> = dxy_hist.iter().map(|(_, v)| *v).collect();
        if let Some(c) = pct_change(&dxy_vals) {
            push(&mut out, "regime.dxy_trend", c, "derived");
        }

        // 10y − 2y curve.
        let ust2y = pit.macro_value_as_of("ust2y").await?;
        let ust10y = pit.macro_value_as_of("ust10y").await?;
        if let (Some(t2), Some(t10)) = (ust2y, ust10y) {
            push(&mut out, "regime.twos_tens", t10 - t2, "derived");
        }

        // Copper/gold ratio trend (risk appetite): 20d % change of the ratio.
        let copper_hist = pit.macro_history("copper", TREND_WINDOW).await?;
        let gold_hist = pit.macro_history("gold", TREND_WINDOW).await?;
        let ratio: Vec<f64> = copper_hist
            .iter()
            .zip(gold_hist.iter())
            .filter(|((_, _c), (_, g))| *g > 0.0)
            .map(|((_, c), (_, g))| c / g)
            .collect();
        if let Some(c) = pct_change(&ratio) {
            push(&mut out, "regime.copper_gold", c, "derived");
        }

        // Oil 20d % change.
        let oil_hist = pit.macro_history("oil", TREND_WINDOW).await?;
        let oil_vals: Vec<f64> = oil_hist.iter().map(|(_, v)| *v).collect();
        if let Some(c) = pct_change(&oil_vals) {
            push(&mut out, "regime.oil_trend", c, "derived");
        }

        // --- Dealer gamma / charm / vanna / DIX (PROPRIETARY hook) -------
        // Read via ctx.proprietary; Unavailable -> skip (never fabricated). When
        // wired, this would emit regime.gex_sign / regime.gamma_flip_dist / etc.
        let ticker = pit.ticker();
        if let Ok(Some(snap)) = ctx.proprietary.gex(ticker, decision_ts).await {
            push(&mut out, "regime.gex_net", snap.net_gex, "proprietary:gex");
            push(
                &mut out,
                "regime.gex_sign",
                if snap.net_gex >= 0.0 { 1.0 } else { -1.0 },
                "proprietary:gex",
            );
        }
        if let Ok(Some(dix)) = ctx.proprietary.dix(ticker, decision_ts).await {
            push(&mut out, "regime.dix", dix, "proprietary:dix");
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_basic() {
        let hist = [0.1, 0.2, 0.3, 0.4, 0.5];
        // 0.3 has 3 of 5 <= it.
        assert!((percentile_of(&hist, 0.3).unwrap() - 0.6).abs() < 1e-9);
        assert_eq!(percentile_of(&[], 1.0), None);
    }

    #[test]
    fn pct_change_basic() {
        assert!((pct_change(&[100.0, 110.0]).unwrap() - 0.10).abs() < 1e-9);
        assert_eq!(pct_change(&[1.0]), None);
        assert_eq!(pct_change(&[0.0, 1.0]), None);
    }

    #[test]
    fn vix_term_structure_ratios() {
        // Backwardation: vix9d/vix < 1 means near-term LESS than spot? No —
        // vix9d/vix < 1 means 9-day vol below 30-day spot vol (calm front).
        // vix_vix3m > 1 means spot above 3-month -> backwardation (stress).
        let vix = 30.0;
        let vix9d = 33.0;
        let vix3m = 25.0;
        // Front-month elevated vs 3m -> backwardation.
        assert!(vix / vix3m > 1.0);
        // 9d above 30d spot -> acute near-term fear.
        assert!(vix9d / vix > 1.0);

        // Calm: contango.
        let (cvix, cvix3m) = (15.0, 18.0);
        assert!(cvix / cvix3m < 1.0);
    }
}
