//! LAYER 0 — the tradeability gate.
//!
//! Only scan names with a large hand leaning on them: meaningful liquidity, flow
//! capacity, and (ideally) observable dealer positioning. Reject the rest, with
//! reasons. In v1 the dealer-positioning signal (|GEX|) is a `PROPRIETARY_FEATURE`
//! hook; when it's unavailable the gate falls back to a liquidity-implied proxy and
//! marks the large-hand component as *unobserved* rather than fabricating it.

use async_trait::async_trait;
use se_core::{AsOf, Feature, Layer, LeadTimeTag, Result};

use crate::indicators::{dollar_adv, realized_vol};
use crate::module::{FeatureContext, FeatureModule};

fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

/// Linear interpolation in log space between a floor (->0) and a "good" level (->1).
fn log_scale(value: f64, floor: f64, good: f64) -> f64 {
    if value <= 0.0 || floor <= 0.0 || good <= floor {
        return 0.0;
    }
    clamp01((value.ln() - floor.ln()) / (good.ln() - floor.ln()))
}

#[derive(Debug, Clone, Copy)]
pub struct TradeabilityConfig {
    pub adv_floor: f64,
    pub adv_good: f64,
    pub aum_floor: f64,
    pub aum_good: f64,
    pub min_holdings: i64,
    pub w_liquidity: f64,
    pub w_flow: f64,
    pub w_large_hand: f64,
}

impl Default for TradeabilityConfig {
    fn default() -> Self {
        TradeabilityConfig {
            adv_floor: 50_000_000.0,    // $50M average daily dollar volume
            adv_good: 1_000_000_000.0,  // $1B
            aum_floor: 500_000_000.0,   // $500M
            aum_good: 10_000_000_000.0, // $10B
            min_holdings: 10,
            w_liquidity: 0.40,
            w_flow: 0.35,
            w_large_hand: 0.25,
        }
    }
}

/// Raw inputs the gate scores. Assembled from bars + ETF profile + (optional)
/// proprietary positioning.
#[derive(Debug, Clone)]
pub struct TradeabilityInput {
    pub symbol: String,
    pub dollar_adv: f64,
    pub aum: Option<f64>,
    pub holdings_count: Option<i64>,
    /// |dealer GEX| if a proprietary feed is wired; `None` in v1.
    pub abs_gex: Option<f64>,
    /// Options open interest proxy if available; `None` in v1.
    pub options_oi: Option<f64>,
    pub realized_vol: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct TradeabilityComponents {
    pub liquidity: f64,
    pub flow_capacity: f64,
    pub large_hand: f64,
    /// False when |GEX|/options were unavailable and a proxy was used instead.
    pub large_hand_observed: bool,
}

#[derive(Debug, Clone)]
pub struct TradeabilityScore {
    pub symbol: String,
    pub score: f64,
    pub passed: bool,
    pub components: TradeabilityComponents,
    pub reasons: Vec<String>,
}

/// The pure, well-tested scorer. No I/O.
#[derive(Debug, Clone, Copy, Default)]
pub struct TradeabilityGate {
    pub cfg: TradeabilityConfig,
}

impl TradeabilityGate {
    pub fn new(cfg: TradeabilityConfig) -> Self {
        TradeabilityGate { cfg }
    }

    pub fn evaluate(&self, input: &TradeabilityInput) -> TradeabilityScore {
        let c = &self.cfg;
        let mut reasons = Vec::new();

        let liquidity = log_scale(input.dollar_adv, c.adv_floor, c.adv_good);
        let aum = input.aum.unwrap_or(0.0);
        let flow_capacity = log_scale(aum, c.aum_floor, c.aum_good);

        // Large-hand signal: prefer |GEX|, then options OI, else a capped proxy.
        let (large_hand, large_hand_observed) = match (input.abs_gex, input.options_oi) {
            (Some(gex), _) => (clamp01(gex / 5.0e9), true), // ~$5B |GEX| -> saturated
            (None, Some(oi)) => (clamp01(oi / 2.0e6), true),
            (None, None) => (liquidity.min(0.5), false),
        };

        let score = clamp01(
            c.w_liquidity * liquidity + c.w_flow * flow_capacity + c.w_large_hand * large_hand,
        );

        // Hard gates -> rejection with explicit reasons.
        if input.dollar_adv < c.adv_floor {
            reasons.push(format!(
                "dollar ADV ${:.1}M < ${:.0}M floor",
                input.dollar_adv / 1e6,
                c.adv_floor / 1e6
            ));
        }
        match input.aum {
            Some(a) if a < c.aum_floor => reasons.push(format!(
                "AUM ${:.0}M < ${:.0}M floor",
                a / 1e6,
                c.aum_floor / 1e6
            )),
            None => reasons.push("AUM unavailable".into()),
            _ => {}
        }
        // Holdings count is NOT a hard gate: provider ETF holdings counts are
        // unreliable (e.g. FMP reports 1 for IWM). It's kept as an ingested raw
        // feature for later layers, but the tradeability gate keys only on the
        // reliable signals — liquidity and flow capacity.
        if !large_hand_observed {
            reasons.push(
                "large hand unobserved (GEX/options unavailable) — using liquidity proxy".into(),
            );
        }

        // Pass requires the liquidity + flow floors (the large-hand note is informational in v1).
        let hard_fail =
            input.dollar_adv < c.adv_floor || input.aum.map(|a| a < c.aum_floor).unwrap_or(true);

        TradeabilityScore {
            symbol: input.symbol.clone(),
            score,
            passed: !hard_fail,
            components: TradeabilityComponents {
                liquidity,
                flow_capacity,
                large_hand,
                large_hand_observed,
            },
            reasons,
        }
    }
}

/// `FeatureModule` wrapper: assembles a [`TradeabilityInput`] from PIT-safe data +
/// proprietary hooks and emits the tradeability features.
pub struct TradeabilityModule {
    pub gate: TradeabilityGate,
    pub lookback: usize,
}

impl Default for TradeabilityModule {
    fn default() -> Self {
        TradeabilityModule {
            gate: TradeabilityGate::default(),
            lookback: 20,
        }
    }
}

#[async_trait]
impl FeatureModule for TradeabilityModule {
    fn layer(&self) -> Layer {
        Layer::Tradeability
    }
    fn name(&self) -> &str {
        "layer0_tradeability"
    }

    async fn compute(&self, ctx: &FeatureContext<'_>) -> Result<Vec<Feature>> {
        let pit = ctx.pit;
        let ticker = pit.ticker();
        let decision_ts = pit.decision_ts();
        let bars = pit.bars("daily", (self.lookback as i64) + 2).await?;
        if bars.is_empty() {
            return Ok(vec![]);
        }
        let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();

        // AUM / holdings come from raw features ingested from the ETF profile.
        let aum = pit.feature_value("tradeability.aum_raw").await?;
        let holdings = pit
            .feature_value("tradeability.holdings_raw")
            .await?
            .map(|v| v as i64);

        // Proprietary positioning (Unavailable -> None, never fabricated).
        let abs_gex = match ctx.proprietary.gex(ticker, decision_ts).await {
            Ok(Some(snap)) => Some(snap.abs_gex),
            _ => None,
        };

        let input = TradeabilityInput {
            symbol: ticker.to_string(),
            dollar_adv: dollar_adv(&bars, self.lookback),
            aum,
            holdings_count: holdings,
            abs_gex,
            options_oi: None,
            realized_vol: realized_vol(&closes, self.lookback),
        };
        let score = self.gate.evaluate(&input);

        let as_of = AsOf::new(decision_ts.inner());
        let mk = |key: &str, value: f64| {
            Feature::new(
                key,
                value,
                Layer::Tradeability,
                as_of,
                LeadTimeTag::EndOfDay,
                "derived",
            )
        };
        Ok(vec![
            mk("tradeability.dollar_adv", input.dollar_adv),
            mk("tradeability.liquidity", score.components.liquidity),
            mk("tradeability.flow_capacity", score.components.flow_capacity),
            mk("tradeability.large_hand", score.components.large_hand),
            mk("tradeability.score", score.score),
            mk("tradeability.passed", if score.passed { 1.0 } else { 0.0 }),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn liquid_etf(symbol: &str) -> TradeabilityInput {
        TradeabilityInput {
            symbol: symbol.into(),
            dollar_adv: 20_000_000_000.0,
            aum: Some(40_000_000_000.0),
            holdings_count: Some(500),
            abs_gex: None,
            options_oi: None,
            realized_vol: 0.12,
        }
    }

    #[test]
    fn liquid_name_passes() {
        let gate = TradeabilityGate::default();
        let s = gate.evaluate(&liquid_etf("SPY"));
        assert!(s.passed);
        assert!(s.score > 0.5);
    }

    #[test]
    fn thin_name_rejected_with_reasons() {
        let gate = TradeabilityGate::default();
        let thin = TradeabilityInput {
            symbol: "THIN".into(),
            dollar_adv: 5_000_000.0,
            aum: Some(80_000_000.0),
            holdings_count: Some(40),
            abs_gex: None,
            options_oi: None,
            realized_vol: 0.4,
        };
        let s = gate.evaluate(&thin);
        assert!(!s.passed);
        assert!(s.reasons.iter().any(|r| r.contains("dollar ADV")));
        assert!(s.reasons.iter().any(|r| r.contains("AUM")));
    }

    #[test]
    fn zero_holdings_is_unknown_not_a_fail() {
        // Providers report 0 for "not populated"; a liquid ETF with holdings=0
        // must still pass on liquidity + flow.
        let gate = TradeabilityGate::default();
        let mut input = liquid_etf("IWM");
        input.holdings_count = Some(0);
        let s = gate.evaluate(&input);
        assert!(
            s.passed,
            "holdings=0 must be treated as unknown, not a hard fail"
        );
    }

    #[test]
    fn missing_aum_rejected() {
        let gate = TradeabilityGate::default();
        let mut input = liquid_etf("NOAUM");
        input.aum = None;
        let s = gate.evaluate(&input);
        assert!(!s.passed);
        assert!(s.reasons.iter().any(|r| r.contains("AUM unavailable")));
    }
}
