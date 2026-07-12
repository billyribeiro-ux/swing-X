//! `se-signal` (P7) — turn a PROMOTED strategy that fires at the latest decision bar into a
//! concrete, executable [`se_core::Signal`] and persist it.
//!
//! Geometry: entry at the latest close (a conservative last-close convention; the journal then
//! fills next-bar-open-or-worse). Stop/target1/target2 come from the genome's evolved
//! [`se_core::RiskModel`] (the geometry the OOS scoreboard kept) resolved against the current
//! ATR + entry — ATR multiples, fixed dollars, percent, or R-multiples. Conviction is a
//! clearly-labeled cohort hit-rate proxy from the strategy's OOS expectancy (see
//! [`conviction`]) — we do not invent a calibrated probability. `why` renders the genome's
//! predicates as [`se_core::Driver`]s. Cohort stats (n, expectancy, CVaR) come from the
//! strategy's persisted OOS score.
//!
//! [`se_core::Signal::new`] enforces the hard invariant: no entry/stop/target/attribution, or
//! degenerate geometry, means no signal is emitted (we propagate the refusal, never fabricate).

pub mod conviction;

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use se_core::{Driver, HorizonProfile, Result, Scanner, Side, Signal, Strategy, Ticker};
use se_features::indicators::atr;
use se_features::{
    EventOverlay, FeatureContext, FeatureModule, LocationModule, RegimeModule, TradeabilityModule,
    TriggerModule,
};
use se_provider::NullProprietary;
use se_regime::RegimeClassifier;
use se_search::persist::{latest_oos_score, load_promoted, StoredOosScore};
use se_search::{wilson_lower_bound, MIN_ACTED_TO_PROMOTE, WILSON_Z_95};
use se_store::Store;

use crate::conviction::{from_cohort, from_oos_precision};

/// Live defence-in-depth precision floor: never surface a single-name signal whose strategy's
/// validated OUT-OF-SAMPLE precision (P(profit | acted) at τ\*) is below this, even though
/// promotion already optimized the acting threshold for cost-aware OOS expectancy. Low-precision
/// fades are the most fragile to single-name earnings/gap risk, so we hold the live bar higher
/// than the search's promote bar. Only applied when the validator measured precision over a
/// non-trivial acted cohort (`n_acted >= MIN_ACTED_TO_PROMOTE`); legacy scores (NULL precision)
/// fall back to the cohort-implied conviction and are unaffected.
pub const MIN_LIVE_PRECISION: f64 = 0.40;

/// Reasons a promoted strategy did not produce a signal at the latest bar (for reporting).
#[derive(Debug, Clone, PartialEq)]
pub enum NoSignal {
    NoBars,
    NotFiring,
    RegimeNotTradeable,
    NoAtr,
    NoOosScore,
    GeometryRefused(String),
    /// An earnings release falls inside the would-be holding window (single-name gap risk).
    EarningsBlackout(chrono::NaiveDate),
    /// The strategy's validated OOS precision (P(profit | acted)) is below the live floor.
    LowPrecision(f64),
}

impl std::fmt::Display for NoSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NoSignal::NoBars => write!(f, "no bars at/under the decision cutoff"),
            NoSignal::NotFiring => write!(f, "genome did not fire at the latest bar"),
            NoSignal::RegimeNotTradeable => write!(f, "regime not tradeable (suppressed)"),
            NoSignal::NoAtr => write!(f, "insufficient history for ATR"),
            NoSignal::NoOosScore => write!(f, "no persisted OOS score for the strategy"),
            NoSignal::GeometryRefused(why) => write!(f, "signal refused: {why}"),
            NoSignal::EarningsBlackout(d) => {
                write!(f, "earnings blackout (release {d} inside holding window)")
            }
            NoSignal::LowPrecision(p) => {
                write!(
                    f,
                    "validated OOS precision {p:.3} below live floor {MIN_LIVE_PRECISION:.2}"
                )
            }
        }
    }
}

/// Is there an earnings release inside the would-be holding window? Returns the offending date if
/// so. The window is the horizon's max holding period converted from trading bars to calendar days
/// (`bars × 7 ÷ 5`, ceil). ETFs have no earnings rows, so this is a no-op for the ETF scanner.
async fn earnings_in_holding_window(
    store: &Store,
    ticker: Ticker,
    decision_ts: DateTime<Utc>,
    profile: &HorizonProfile,
) -> Result<Option<chrono::NaiveDate>> {
    let decision_date = decision_ts.date_naive();
    // Trading bars -> calendar days (ceil of bars × 7/5), at least 1.
    let blackout_days = ((profile.max_hold_bars as i64 * 7) + 4) / 5;
    let blackout_days = blackout_days.max(1);
    let until = decision_date + chrono::Duration::days(blackout_days);
    let row: Option<(chrono::NaiveDate,)> = se_store::sqlx::query_as(
        "SELECT date FROM earnings WHERE ticker = $1 AND date > $2 AND date <= $3 \
         ORDER BY date ASC LIMIT 1",
    )
    .bind(ticker.as_str())
    .bind(decision_date)
    .bind(until)
    .fetch_optional(store.pool())
    .await
    .map_err(|e| se_core::Error::Store(e.to_string()))?;
    Ok(row.map(|(d,)| d))
}

/// The outcome of attempting to build a signal for one strategy at one ticker.
pub enum SignalAttempt {
    /// A built signal (boxed — `Signal` is large relative to the skip variant).
    Emitted(Box<Signal>),
    Skipped(NoSignal),
}

/// Build the merged PIT-safe feature map (all layers, dotted keys) at `decision_ts` for
/// `ticker`, plus the latest bar and the ATR at that bar.
async fn features_at_latest(
    store: &Store,
    ticker: Ticker,
    profile: &HorizonProfile,
) -> Result<Option<(BTreeMap<String, f64>, se_core::Bar, Option<f64>)>> {
    // Find the latest stored daily bar timestamp for this ticker.
    let latest_ts: Option<(DateTime<Utc>,)> = se_store::sqlx::query_as(
        "SELECT ts FROM bars WHERE ticker = $1 AND cadence = 'daily' ORDER BY ts DESC LIMIT 1",
    )
    .bind(ticker.as_str())
    .fetch_optional(store.pool())
    .await
    .map_err(|e| se_core::Error::Store(e.to_string()))?;
    let Some((ts,)) = latest_ts else {
        return Ok(None);
    };

    let decision = se_core::DecisionTs::new(ts);
    let pit = store.pit(ticker, decision);
    let Some(bar) = pit.latest_bar("daily").await? else {
        return Ok(None);
    };
    let bars = pit.bars("daily", (profile.atr_lookback as i64) + 5).await?;
    let atr_val = atr(&bars, profile.atr_lookback as usize).filter(|a| *a > 0.0 && a.is_finite());

    let prop = NullProprietary;
    let modules: [&dyn FeatureModule; 5] = [
        &TradeabilityModule::default(),
        &RegimeModule,
        &LocationModule::new(),
        &TriggerModule::new(),
        &EventOverlay::new(),
    ];
    let ctx = FeatureContext::new(&pit, &prop, *profile);
    let mut features = BTreeMap::new();
    for m in modules {
        for f in m.compute(&ctx).await? {
            if f.value.is_finite() {
                features.insert(f.key, f.value);
            }
        }
    }
    Ok(Some((features, bar, atr_val)))
}

/// Render a genome's predicates as attribution [`Driver`]s. Contribution is a small signed
/// magnitude derived from the predicate direction (the genome is a conjunction; each clause
/// contributes equally), and the detail carries the human predicate.
fn drivers_for(genome: &se_core::Genome, features: &BTreeMap<String, f64>) -> Vec<Driver> {
    let n = genome.predicates.len().max(1) as f64;
    genome
        .predicates
        .iter()
        .map(|p| {
            let observed = features.get(&p.feature_key).copied();
            let sign = match p.op {
                se_core::CmpOp::Gt | se_core::CmpOp::Gte => 1.0,
                se_core::CmpOp::Lt | se_core::CmpOp::Lte => -1.0,
            };
            let detail = match observed {
                Some(v) => format!("{} (observed {:.4})", p.describe(), v),
                None => p.describe(),
            };
            Driver {
                layer: p.layer,
                key: p.feature_key.clone(),
                contribution: sign / n,
                detail,
            }
        })
        .collect()
}

/// Build a human invalidation rule from the stop geometry.
fn invalidation_rule(side: Side, stop: f64) -> String {
    match side {
        Side::Long => format!("daily close < {stop:.2} (stop)"),
        Side::Short => format!("daily close > {stop:.2} (stop)"),
    }
}

/// Attempt to build a signal for one promoted `strategy` at one `ticker`'s latest decision bar.
pub async fn build_signal_for(
    store: &Store,
    strategy: &Strategy,
    ticker: Ticker,
    profile: &HorizonProfile,
) -> Result<SignalAttempt> {
    let Some((features, bar, atr_opt)) = features_at_latest(store, ticker, profile).await? else {
        return Ok(SignalAttempt::Skipped(NoSignal::NoBars));
    };

    if !strategy.genome.fires(&features) {
        return Ok(SignalAttempt::Skipped(NoSignal::NotFiring));
    }

    // Regime gate at the bar.
    let assessment = RegimeClassifier::default().classify(&features);
    if !assessment.label.is_tradeable() {
        return Ok(SignalAttempt::Skipped(NoSignal::RegimeNotTradeable));
    }

    // Earnings blackout: never open a new position into a release inside the holding window
    // (single-name gap risk that the stop can't honor). No-op for ETFs (no earnings rows).
    if let Some(d) = earnings_in_holding_window(store, ticker, bar.ts, profile).await? {
        return Ok(SignalAttempt::Skipped(NoSignal::EarningsBlackout(d)));
    }

    let Some(atr_val) = atr_opt else {
        return Ok(SignalAttempt::Skipped(NoSignal::NoAtr));
    };

    // Cohort stats from the persisted OOS score (fail-closed: no score -> no signal).
    let Some(score) = latest_oos_score(store, strategy.id).await? else {
        return Ok(SignalAttempt::Skipped(NoSignal::NoOosScore));
    };

    // Live precision floor (defence-in-depth): when the validator measured this strategy's OOS
    // precision (P(net profit | acted) at τ*) over a non-trivial acted cohort, hold the live
    // single-name bar above the search's promote bar. We gate on the WILSON LOWER BOUND of the
    // precision, not the point estimate, so a small-n high-precision fluke cannot slip through on
    // sampling luck. Legacy scores (NULL precision) skip this check.
    if let (Some(p), Some(n)) = (score.precision_oos, score.n_acted) {
        let lb = wilson_lower_bound(p, n, WILSON_Z_95);
        if n as usize >= MIN_ACTED_TO_PROMOTE && lb < MIN_LIVE_PRECISION {
            return Ok(SignalAttempt::Skipped(NoSignal::LowPrecision(lb)));
        }
    }

    let side = strategy.genome.side;
    let entry = bar.close;
    // Geometry comes from the genome's evolved risk model (stop/target1/target2), not the
    // profile mults — so the signal honors exactly the geometry the OOS scoreboard kept.
    let risk = &strategy.genome.risk;
    let stop = risk.stop_price(entry, atr_val, side);
    let (target1, target2) = risk.target_prices(entry, atr_val, side);
    let target1_dist = (target1 - entry).abs();

    let rr1 = if (entry - stop).abs() > 0.0 {
        target1_dist / (entry - stop).abs()
    } else {
        0.0
    };
    // Conviction: prefer the strategy's OUT-OF-SAMPLE measured precision at the meta-labeling
    // acting threshold τ* (a directly-measured P(profit | acted)) over the expectancy-implied
    // cohort proxy — but only when measured over a non-trivial acted cohort.
    let conviction = match (score.precision_oos, score.n_acted) {
        (Some(p), Some(n)) if n as usize >= MIN_ACTED_TO_PROMOTE => from_oos_precision(p, n),
        _ => from_cohort(score_expectancy(&score), rr1),
    };

    let drivers = drivers_for(&strategy.genome, &features);
    let invalidation = invalidation_rule(side, stop);
    let cohort_n = score_cohort_n(&score);
    let tau = score.act_threshold.unwrap_or(0.5);
    let lead_time = format!(
        "next-bar-open fill; conviction {} (acting τ*={tau:.2})",
        conviction.label
    );

    match Signal::new(
        strategy.id,
        ticker,
        side,
        se_core::DecisionTs::new(bar.ts),
        profile.horizon,
        entry,
        stop,
        target1,
        target2,
        conviction.value,
        cohort_n,
        assessment.label,
        assessment.label.as_str(),
        drivers,
        invalidation,
        score_expectancy(&score),
        score.cvar5.unwrap_or(0.0),
        lead_time,
    ) {
        Ok(signal) => Ok(SignalAttempt::Emitted(Box::new(signal))),
        Err(e) => Ok(SignalAttempt::Skipped(NoSignal::GeometryRefused(
            e.to_string(),
        ))),
    }
}

fn score_expectancy(s: &StoredOosScore) -> f64 {
    s.oos_expectancy_cost_aware.unwrap_or(0.0)
}

/// Cohort size: the true entry count that fed the OOS validation, recovered from the persisted
/// `fold_spec` JSON (`n_entries`).
fn score_cohort_n(s: &StoredOosScore) -> u32 {
    s.n_entries
}

/// Generate signals across the universe from all promoted strategies for the active horizon.
/// Returns the emitted signals; persists each to the `signals` table. Skips (with reasons
/// logged) are not persisted.
pub async fn generate_signals(
    store: &Store,
    profile: &HorizonProfile,
    universe: &[Ticker],
    scanner: Scanner,
) -> Result<Vec<Signal>> {
    let promoted = load_promoted(store, profile.horizon.as_str(), scanner).await?;
    let mut out = Vec::new();
    for strategy in &promoted {
        for &ticker in universe {
            match build_signal_for(store, strategy, ticker, profile).await? {
                SignalAttempt::Emitted(sig) => {
                    let sig = *sig;
                    persist_signal(store, &sig, scanner).await?;
                    out.push(sig);
                }
                SignalAttempt::Skipped(reason) => {
                    tracing::debug!(strategy = %strategy.id, %ticker, reason = %reason, "no signal");
                }
            }
        }
    }
    Ok(out)
}

/// Persist one signal to the `signals` table (idempotent on `signal_id`).
pub async fn persist_signal(store: &Store, sig: &Signal, scanner: Scanner) -> Result<()> {
    let why_json =
        serde_json::to_value(&sig.why).map_err(|e| se_core::Error::Store(e.to_string()))?;
    let payload_json =
        serde_json::to_value(sig).map_err(|e| se_core::Error::Store(e.to_string()))?;
    let payload_human = sig.to_human();
    let side = match sig.side {
        Side::Long => "long",
        Side::Short => "short",
    };

    se_store::sqlx::query(
        "INSERT INTO signals \
            (signal_id, strategy_id, ticker, side, decision_ts, horizon, entry, stop, \
             target1, target2, rr1, rr2, conviction, cohort_n, regime_desc, why, invalidation, \
             cohort_expectancy, cvar5, lead_time, payload_json, payload_human, scanner) \
         VALUES \
            ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23) \
         ON CONFLICT (signal_id) DO NOTHING",
    )
    .bind(sig.id.inner())
    .bind(sig.strategy_id.inner())
    .bind(sig.ticker.as_str())
    .bind(side)
    .bind(sig.decision_ts.inner())
    .bind(sig.horizon.as_str())
    .bind(sig.entry)
    .bind(sig.stop)
    .bind(sig.target1)
    .bind(sig.target2)
    .bind(sig.rr1)
    .bind(sig.rr2)
    .bind(sig.conviction)
    .bind(sig.cohort_n as i32)
    .bind(&sig.regime_desc)
    .bind(why_json)
    .bind(&sig.invalidation)
    .bind(sig.cohort_expectancy)
    .bind(sig.cvar5)
    .bind(&sig.lead_time)
    .bind(payload_json)
    .bind(payload_human)
    .bind(scanner.as_str())
    .execute(store.pool())
    .await
    .map_err(|e| se_core::Error::Store(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use se_core::{CmpOp, Genome, Horizon, Layer, Predicate};

    fn genome() -> Genome {
        Genome::new(
            Side::Long,
            Horizon::Swing,
            vec![
                Predicate {
                    layer: Layer::Trigger,
                    feature_key: "trigger.rsi14".into(),
                    op: CmpOp::Gt,
                    threshold: 55.0,
                },
                Predicate {
                    layer: Layer::Location,
                    feature_key: "location.dist_50dma".into(),
                    op: CmpOp::Lt,
                    threshold: 0.02,
                },
            ],
        )
    }

    #[test]
    fn drivers_render_predicates_with_layers() {
        let g = genome();
        let mut feats = BTreeMap::new();
        feats.insert("trigger.rsi14".to_string(), 60.0);
        feats.insert("location.dist_50dma".to_string(), 0.0);
        let ds = drivers_for(&g, &feats);
        assert_eq!(ds.len(), 2);
        assert_eq!(ds[0].layer, Layer::Trigger);
        assert!(ds[0].detail.contains("rsi14"));
        assert!(ds[0].contribution > 0.0); // Gt -> positive
        assert!(ds[1].contribution < 0.0); // Lt -> negative
    }

    #[test]
    fn invalidation_is_directional() {
        assert!(invalidation_rule(Side::Long, 120.0).contains("< 120"));
        assert!(invalidation_rule(Side::Short, 120.0).contains("> 120"));
    }

    #[test]
    fn signal_geometry_honors_genome_risk_model() {
        use se_core::{RegimeLabel, RiskModel, SignalId, StopSpec, StrategyId, TargetSpec, Ticker};

        // A genome with a fixed $5 stop and 2R/3R targets.
        let g = Genome::with_risk(
            Side::Long,
            Horizon::Swing,
            vec![Predicate {
                layer: Layer::Trigger,
                feature_key: "trigger.rsi14".into(),
                op: CmpOp::Gt,
                threshold: 55.0,
            }],
            RiskModel::new(
                StopSpec::fixed(5.0),
                TargetSpec::r_multiple(2.0),
                Some(TargetSpec::r_multiple(3.0)),
            ),
        );
        // Resolve geometry exactly as build_signal_for does.
        let entry = 100.0;
        let atr = 2.0; // irrelevant for a fixed stop
        let stop = g.risk.stop_price(entry, atr, g.side);
        let (t1, t2) = g.risk.target_prices(entry, atr, g.side);
        assert!((stop - 95.0).abs() < 1e-9); // $5 below
        assert!((t1 - 110.0).abs() < 1e-9); // 2R = $10 above
        assert!((t2.unwrap() - 115.0).abs() < 1e-9); // 3R = $15 above

        // Signal::new must accept this geometry (directionally consistent).
        let sig = Signal::new(
            StrategyId::new(),
            Ticker::SPY,
            g.side,
            se_core::DecisionTs::new(Utc::now()),
            g.horizon,
            entry,
            stop,
            t1,
            t2,
            0.5,
            50,
            RegimeLabel::RiskOn,
            "risk_on",
            drivers_for(&g, &BTreeMap::from([("trigger.rsi14".to_string(), 60.0)])),
            invalidation_rule(g.side, stop),
            0.2,
            -1.0,
            "test",
        )
        .expect("fixed-dollar risk geometry must build a valid signal");
        assert!((sig.rr1 - 2.0).abs() < 1e-9, "rr1={}", sig.rr1);
        let _ = SignalId::new();
    }
}
