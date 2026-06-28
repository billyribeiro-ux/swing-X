//! Pure detector math — no I/O. Every function here takes plain synthetic-able
//! inputs (vectors of realized R-multiples, predicted/realized pairs, timestamps)
//! and returns an `Option<Decision>`: `None` when the metric is within tolerance
//! or there is not enough data to judge, `Some` when the threshold is breached and
//! an action is warranted.
//!
//! The DB-facing layer in `lib.rs` is responsible for *gathering* these inputs and
//! *acting* on the returned [`Decision`]s (writing `monitor_events`, flipping
//! strategy status). Keeping the thresholds and arithmetic here makes them unit
//! testable against hand-computed expectations with zero database.

use se_core::MonitorAction;

/// Default thresholds for the forward-adaptation detectors. Centralized so the
/// numbers live in exactly one place and tests can assert against them.
#[derive(Debug, Clone, Copy)]
pub struct Thresholds {
    /// backtest-vs-live: |live − oos| expectancy gap (R) that triggers ShrinkSize.
    pub divergence_shrink_r: f64,
    /// backtest-vs-live: gap (R) that escalates to Quarantine.
    pub divergence_quarantine_r: f64,
    /// minimum realized trades before a divergence verdict is trusted.
    pub divergence_min_trades: usize,
    /// drawdown: per-strategy CVaR(5%) floor in R (more negative => breach).
    pub drawdown_cvar_floor_r: f64,
    /// drawdown: max-drawdown floor in R (more negative => breach).
    pub drawdown_maxdd_floor_r: f64,
    /// drawdown: minimum trades before a drawdown verdict is trusted.
    pub drawdown_min_trades: usize,
    /// calibration: reliability gap |mean_conviction − hit_rate| that triggers Recalibrate.
    pub calibration_gap: f64,
    /// calibration: minimum trades with a known outcome before judging.
    pub calibration_min_trades: usize,
    /// staleness: max age (hours) of the latest bar/feature before SkipDegraded.
    pub staleness_max_age_hours: f64,
    /// regime OOD: count of recent out_of_distribution labels that triggers Suppress.
    pub ood_count: i64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds {
            divergence_shrink_r: 0.25,
            divergence_quarantine_r: 0.50,
            divergence_min_trades: 10,
            drawdown_cvar_floor_r: -2.0,
            drawdown_maxdd_floor_r: -6.0,
            drawdown_min_trades: 10,
            calibration_gap: 0.15,
            calibration_min_trades: 20,
            staleness_max_age_hours: 48.0,
            ood_count: 3,
        }
    }
}

/// The outcome of a fired detector: the metric value observed, the threshold it
/// breached, and the action to take. The DB layer turns this into a `monitor_events`
/// row and any side effect (status flip, alert).
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub metric_value: f64,
    pub threshold: f64,
    pub action: MonitorAction,
    /// Short machine/human note, copied into the event `detail` jsonb.
    pub note: String,
}

/// Mean of a slice, or `None` for an empty slice.
pub fn mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    Some(xs.iter().sum::<f64>() / xs.len() as f64)
}

/// CVaR (expected shortfall) at the given tail fraction `alpha` (e.g. 0.05),
/// computed on realized R-multiples. Returns the mean of the worst `alpha` share
/// of outcomes (always at least one observation). Lower (more negative) is worse.
/// `None` when there is no data.
pub fn cvar(returns: &[f64], alpha: f64) -> Option<f64> {
    if returns.is_empty() {
        return None;
    }
    let mut sorted: Vec<f64> = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let alpha = alpha.clamp(0.0, 1.0);
    let k = ((sorted.len() as f64) * alpha).ceil() as usize;
    let k = k.max(1).min(sorted.len());
    let tail = &sorted[..k];
    Some(tail.iter().sum::<f64>() / k as f64)
}

/// Maximum peak-to-trough drawdown of the cumulative-R equity curve, in R units.
/// Returns `<= 0.0` (0 means monotonically non-decreasing). `None` for empty input.
pub fn max_drawdown(returns: &[f64]) -> Option<f64> {
    if returns.is_empty() {
        return None;
    }
    let mut cum = 0.0_f64;
    let mut peak = 0.0_f64;
    let mut max_dd = 0.0_f64;
    for &r in returns {
        cum += r;
        if cum > peak {
            peak = cum;
        }
        let dd = cum - peak;
        if dd < max_dd {
            max_dd = dd;
        }
    }
    Some(max_dd)
}

/// **backtest-vs-live divergence**. Compares rolling realized expectancy (mean R
/// over recent live/paper trades) against the strategy's OOS expectancy. A large
/// shortfall of live below backtest means the edge is not surviving forward.
///
/// * gap >= quarantine threshold -> Quarantine
/// * gap >= shrink threshold     -> ShrinkSize
/// * otherwise / insufficient data -> None
pub fn detect_divergence(
    realized_r: &[f64],
    oos_expectancy: f64,
    t: &Thresholds,
) -> Option<Decision> {
    if realized_r.len() < t.divergence_min_trades {
        return None;
    }
    let live = mean(realized_r)?;
    // Shortfall: how far live underperforms the backtested expectancy.
    let gap = oos_expectancy - live;
    if gap >= t.divergence_quarantine_r {
        Some(Decision {
            metric_value: gap,
            threshold: t.divergence_quarantine_r,
            action: MonitorAction::Quarantine,
            note: format!(
                "live expectancy {live:.3}R vs OOS {oos_expectancy:.3}R (gap {gap:.3}R) — quarantine"
            ),
        })
    } else if gap >= t.divergence_shrink_r {
        Some(Decision {
            metric_value: gap,
            threshold: t.divergence_shrink_r,
            action: MonitorAction::ShrinkSize,
            note: format!(
                "live expectancy {live:.3}R vs OOS {oos_expectancy:.3}R (gap {gap:.3}R) — shrink size"
            ),
        })
    } else {
        None
    }
}

/// **drawdown breach**. Flags when realized CVaR(5%) or max-drawdown falls through
/// the configured floor. Returns `Disable` (the DB layer also emits a paired `Alert`).
pub fn detect_drawdown(realized_r: &[f64], t: &Thresholds) -> Option<Decision> {
    if realized_r.len() < t.drawdown_min_trades {
        return None;
    }
    let cv = cvar(realized_r, 0.05)?;
    let mdd = max_drawdown(realized_r)?;
    if cv <= t.drawdown_cvar_floor_r {
        return Some(Decision {
            metric_value: cv,
            threshold: t.drawdown_cvar_floor_r,
            action: MonitorAction::Disable,
            note: format!(
                "CVaR(5%) {cv:.2}R breached floor {:.2}R",
                t.drawdown_cvar_floor_r
            ),
        });
    }
    if mdd <= t.drawdown_maxdd_floor_r {
        return Some(Decision {
            metric_value: mdd,
            threshold: t.drawdown_maxdd_floor_r,
            action: MonitorAction::Disable,
            note: format!(
                "max drawdown {mdd:.2}R breached floor {:.2}R",
                t.drawdown_maxdd_floor_r
            ),
        });
    }
    None
}

/// **calibration break**. Compares mean predicted conviction against the realized
/// hit-rate (fraction of trades with positive R). A persistent reliability gap means
/// the conviction model is miscalibrated -> Recalibrate.
///
/// `convictions[i]` pairs with `wins[i]` (true iff that trade closed positive R).
pub fn detect_calibration(convictions: &[f64], wins: &[bool], t: &Thresholds) -> Option<Decision> {
    let n = convictions.len().min(wins.len());
    if n < t.calibration_min_trades {
        return None;
    }
    let predicted = mean(&convictions[..n])?;
    let realized = wins[..n].iter().filter(|w| **w).count() as f64 / n as f64;
    let gap = (predicted - realized).abs();
    if gap >= t.calibration_gap {
        Some(Decision {
            metric_value: gap,
            threshold: t.calibration_gap,
            action: MonitorAction::Recalibrate,
            note: format!(
                "reliability gap {gap:.3} (predicted {predicted:.3} vs hit-rate {realized:.3}) over {n} trades"
            ),
        })
    } else {
        None
    }
}

/// **data outage / staleness**. `age_hours` is now − latest bar/feature timestamp
/// for a source. Beyond the freshness budget the source is degraded -> SkipDegraded
/// (paired with an Alert by the DB layer).
pub fn detect_staleness(age_hours: f64, t: &Thresholds) -> Option<Decision> {
    if age_hours > t.staleness_max_age_hours {
        Some(Decision {
            metric_value: age_hours,
            threshold: t.staleness_max_age_hours,
            action: MonitorAction::SkipDegraded,
            note: format!(
                "latest data {age_hours:.1}h old (> {:.1}h budget) — skipping degraded source",
                t.staleness_max_age_hours
            ),
        })
    } else {
        None
    }
}

/// **regime OOD**. Counts recent `out_of_distribution` regime labels; once the count
/// crosses the threshold the engine suppresses signals rather than guessing.
pub fn detect_regime_ood(ood_count: i64, t: &Thresholds) -> Option<Decision> {
    if ood_count >= t.ood_count {
        Some(Decision {
            metric_value: ood_count as f64,
            threshold: t.ood_count as f64,
            action: MonitorAction::Suppress,
            note: format!(
                "{ood_count} out-of-distribution regime labels (>= {}) — suppressing signals",
                t.ood_count
            ),
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_and_cvar_basics() {
        assert_eq!(mean(&[]), None);
        assert_eq!(mean(&[1.0, 3.0]), Some(2.0));
        // tail of worst 5% with 10 obs => ceil(0.5)=1 => the single worst value.
        let rs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -4.0];
        assert_eq!(cvar(&rs, 0.05), Some(-4.0));
        assert_eq!(cvar(&[], 0.05), None);
    }

    #[test]
    fn cvar_averages_the_tail_share() {
        // 20 obs, alpha 0.10 => ceil(2)=2 worst values averaged.
        let mut rs = vec![1.0; 18];
        rs.push(-3.0);
        rs.push(-5.0);
        assert_eq!(cvar(&rs, 0.10), Some(-4.0));
    }

    #[test]
    fn max_drawdown_tracks_peak_to_trough() {
        assert_eq!(max_drawdown(&[]), None);
        // equity: +1, +2, -1(=1), -3(=-2) ... peak 2, trough -2 => dd -4.
        let rs = [1.0, 1.0, -3.0, -1.0];
        assert_eq!(max_drawdown(&rs), Some(-4.0));
        // monotonic up => 0 drawdown.
        assert_eq!(max_drawdown(&[1.0, 1.0, 1.0]), Some(0.0));
    }

    #[test]
    fn divergence_escalates_with_gap() {
        let t = Thresholds::default();
        // not enough trades -> None
        assert!(detect_divergence(&[0.0; 5], 0.5, &t).is_none());
        // live ~0.4, oos 0.5 => gap 0.1 < shrink(0.25) -> None
        assert!(detect_divergence(&[0.4; 12], 0.5, &t).is_none());
        // gap 0.30 in [0.25,0.50) -> ShrinkSize
        let d = detect_divergence(&[0.2; 12], 0.5, &t).unwrap();
        assert_eq!(d.action, MonitorAction::ShrinkSize);
        // gap 0.60 >= 0.50 -> Quarantine
        let d = detect_divergence(&[-0.1; 12], 0.5, &t).unwrap();
        assert_eq!(d.action, MonitorAction::Quarantine);
    }

    #[test]
    fn drawdown_breach_disables() {
        let t = Thresholds::default();
        // healthy small wins -> None
        assert!(detect_drawdown(&[0.3; 12], &t).is_none());
        // a brutal tail trips CVaR floor (-2.0). 12 obs => ceil(0.6)=1 worst => -5.
        let mut rs = vec![0.5; 11];
        rs.push(-5.0);
        let d = detect_drawdown(&rs, &t).unwrap();
        assert_eq!(d.action, MonitorAction::Disable);
        assert!(d.metric_value <= t.drawdown_cvar_floor_r);
    }

    #[test]
    fn maxdd_breach_disables_even_with_ok_cvar() {
        let t = Thresholds {
            drawdown_cvar_floor_r: -100.0, // make CVaR un-trippable
            ..Thresholds::default()
        };
        // steady bleed: cumulative goes deeply negative -> maxdd floor (-6) breached.
        let rs = vec![-1.0; 12];
        let d = detect_drawdown(&rs, &t).unwrap();
        assert_eq!(d.action, MonitorAction::Disable);
        assert!(d.metric_value <= t.drawdown_maxdd_floor_r);
    }

    #[test]
    fn calibration_gap_triggers_recalibrate() {
        let t = Thresholds::default();
        // 25 trades, predicted ~0.8 but only ~40% win => gap ~0.4 -> Recalibrate.
        let conv = vec![0.8; 25];
        let mut wins = vec![true; 10];
        wins.extend(vec![false; 15]);
        let d = detect_calibration(&conv, &wins, &t).unwrap();
        assert_eq!(d.action, MonitorAction::Recalibrate);
        // well-calibrated => None
        let conv = vec![0.5; 25];
        let wins: Vec<bool> = (0..25).map(|i| i % 2 == 0).collect();
        assert!(detect_calibration(&conv, &wins, &t).is_none());
        // too few trades => None
        assert!(detect_calibration(&[0.9; 5], &[false; 5], &t).is_none());
    }

    #[test]
    fn staleness_and_ood() {
        let t = Thresholds::default();
        assert!(detect_staleness(10.0, &t).is_none());
        let d = detect_staleness(72.0, &t).unwrap();
        assert_eq!(d.action, MonitorAction::SkipDegraded);

        assert!(detect_regime_ood(2, &t).is_none());
        let d = detect_regime_ood(4, &t).unwrap();
        assert_eq!(d.action, MonitorAction::Suppress);
    }
}
