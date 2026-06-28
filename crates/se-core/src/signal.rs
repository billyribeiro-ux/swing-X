//! Executable signal output + paper-trade journal + monitor action types.
//!
//! No setup surfaces without entry, stop, target, and attribution — that invariant
//! is enforced by [`Signal::new`] returning a `Result`.

use serde::{Deserialize, Serialize};

use crate::{
    DecisionTs, Error, Horizon, Layer, RegimeLabel, Result, Side, SignalId, StrategyId, Ticker,
    TradeId,
};

/// One driver behind a signal — a feature that contributed, by layer, with weight.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Driver {
    pub layer: Layer,
    pub key: String,
    pub contribution: f64,
    pub detail: String,
}

/// A concrete, executable setup. Every field needed to act is present and validated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Signal {
    pub id: SignalId,
    pub strategy_id: StrategyId,
    pub ticker: Ticker,
    pub side: Side,
    pub decision_ts: DecisionTs,
    pub horizon: Horizon,
    pub entry: f64,
    pub stop: f64,
    pub target1: f64,
    pub target2: Option<f64>,
    pub rr1: f64,
    pub rr2: Option<f64>,
    /// Calibrated probability from the regime-matched cohort.
    pub conviction: f64,
    pub cohort_n: u32,
    pub regime: RegimeLabel,
    pub regime_desc: String,
    pub why: Vec<Driver>,
    /// Hard invalidation in human terms (e.g. "daily close < 120.0").
    pub invalidation: String,
    pub cohort_expectancy: f64,
    pub cvar5: f64,
    pub lead_time: String,
}

impl Signal {
    /// Build a signal, refusing to surface anything missing entry/stop/target/attribution
    /// or with degenerate geometry.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        strategy_id: StrategyId,
        ticker: Ticker,
        side: Side,
        decision_ts: DecisionTs,
        horizon: Horizon,
        entry: f64,
        stop: f64,
        target1: f64,
        target2: Option<f64>,
        conviction: f64,
        cohort_n: u32,
        regime: RegimeLabel,
        regime_desc: impl Into<String>,
        why: Vec<Driver>,
        invalidation: impl Into<String>,
        cohort_expectancy: f64,
        cvar5: f64,
        lead_time: impl Into<String>,
    ) -> Result<Self> {
        if ![entry, stop, target1].iter().all(|x| x.is_finite()) {
            return Err(Error::Validation("signal price not finite".into()));
        }
        if why.is_empty() {
            return Err(Error::Validation("signal has no attribution".into()));
        }
        let risk = (entry - stop).abs();
        if risk <= 0.0 {
            return Err(Error::Validation("signal risk (entry-stop) is zero".into()));
        }
        // Geometry must be directionally consistent.
        let ok_dir = match side {
            Side::Long => stop < entry && target1 > entry,
            Side::Short => stop > entry && target1 < entry,
        };
        if !ok_dir {
            return Err(Error::Validation(
                "signal geometry inconsistent with side".into(),
            ));
        }
        let rr1 = (target1 - entry).abs() / risk;
        let rr2 = target2.map(|t| (t - entry).abs() / risk);
        Ok(Signal {
            id: SignalId::new(),
            strategy_id,
            ticker,
            side,
            decision_ts,
            horizon,
            entry,
            stop,
            target1,
            target2,
            rr1,
            rr2,
            conviction,
            cohort_n,
            regime,
            regime_desc: regime_desc.into(),
            why,
            invalidation: invalidation.into(),
            cohort_expectancy,
            cvar5,
            lead_time: lead_time.into(),
        })
    }

    /// Human-readable rendering (the §6 format).
    pub fn to_human(&self) -> String {
        let t2 = self
            .target2
            .map(|t| format!("   Target 2: {t:.2}"))
            .unwrap_or_default();
        let rr2 = self.rr2.map(|r| format!(" / {r:.1}")).unwrap_or_default();
        let why = self
            .why
            .iter()
            .map(|d| format!("{} ({:+.2})", d.key, d.contribution))
            .collect::<Vec<_>>()
            .join("; ");
        format!(
            "{} — {} ({})\nEntry: {:.2}   Stop: {:.2}\nTarget 1: {:.2}{}   Risk:Reward {:.1}{}\n\
             Conviction: {:.2} (calibrated, regime-matched cohort n={})\nRegime: {}\nWhy: {}\n\
             Invalidation: {}\nExpectancy(cohort): {:+.2}R   CVaR(5%): {:.1}R   Lead-time: {}",
            self.ticker,
            if self.side == Side::Long { "LONG" } else { "SHORT" },
            self.horizon.as_str(),
            self.entry,
            self.stop,
            self.target1,
            t2,
            self.rr1,
            rr2,
            self.conviction,
            self.cohort_n,
            self.regime_desc,
            why,
            self.invalidation,
            self.cohort_expectancy,
            self.cvar5,
            self.lead_time,
        )
    }
}

/// Paper or live mode for a journaled trade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TradeMode {
    Paper,
    Live,
}

impl TradeMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            TradeMode::Paper => "paper",
            TradeMode::Live => "live",
        }
    }
}

/// A journaled trade with realistic fills (next-bar-open or worse) and attribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trade {
    pub id: TradeId,
    pub signal_id: Option<SignalId>,
    pub strategy_id: Option<StrategyId>,
    pub ticker: Ticker,
    pub side: Side,
    pub mode: TradeMode,
    pub entry_ts: DecisionTs,
    pub fill_px: f64,
    pub fill_ts: DecisionTs,
    pub exit_ts: Option<DecisionTs>,
    pub exit_px: Option<f64>,
    pub pnl_r: Option<f64>,
    pub cost_frac: f64,
}

/// Automatic action the forward-adaptation monitor can take. detect -> act -> log -> alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MonitorAction {
    ShrinkSize,
    Quarantine,
    ForceRefit,
    Recalibrate,
    Suppress,
    Disable,
    Demote,
    Retire,
    SkipDegraded,
    Alert,
}

impl MonitorAction {
    pub const fn as_str(self) -> &'static str {
        match self {
            MonitorAction::ShrinkSize => "shrink_size",
            MonitorAction::Quarantine => "quarantine",
            MonitorAction::ForceRefit => "force_refit",
            MonitorAction::Recalibrate => "recalibrate",
            MonitorAction::Suppress => "suppress",
            MonitorAction::Disable => "disable",
            MonitorAction::Demote => "demote",
            MonitorAction::Retire => "retire",
            MonitorAction::SkipDegraded => "skip_degraded",
            MonitorAction::Alert => "alert",
        }
    }
}
