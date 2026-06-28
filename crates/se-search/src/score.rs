//! The OOS scoreboard.
//!
//! THE SCOREBOARD RANKS ONLY ON OUT-OF-SAMPLE METRICS. There is no in-sample number anywhere in
//! the ranking key: an in-sample-fit edge means nothing here. This is enforced structurally by
//! [`OosScore`] — the only thing the search may sort on — which is constructed *exclusively*
//! from a [`ValidationResult`] (the worker's purged+embargoed CPCV output) and the Rust-side
//! [`GateDecision`]. There is deliberately no constructor that accepts an in-sample metric.
//!
//! Ranking order: gate-passing strategies first, then by cost-aware OOS expectancy, then by
//! DSR — both OOS quantities. `win_rate` is never present (it is not even a field on
//! `ValidationResult`).

use std::cmp::Ordering;

use se_core::{HorizonProfile, StrategyId};
use se_mlclient::{DatasetRow, ValidationResult};
use se_validation::{GateDecision, ValidationHarness};

/// The minimum number of labeled entries a genome must produce before it is even sent to the
/// OOS validator. Below this the CPCV cannot form meaningful folds; we skip and log rather than
/// crash (see [`crate::population`]).
pub const MIN_ENTRIES_TO_VALIDATE: usize = 40;

/// An out-of-sample score for one strategy. The ONLY sortable summary the search keeps.
///
/// Constructed only from OOS validation output + the gate. The fields below are all OOS or
/// gate-derived; none is an in-sample fit metric.
#[derive(Debug, Clone)]
pub struct OosScore {
    pub strategy_id: StrategyId,
    /// Deflated Sharpe ratio (OOS, deflated for the trial count).
    pub dsr: f64,
    /// Probability of backtest overfit (CSCV).
    pub pbo: f64,
    /// Cost-aware OOS expectancy in R — the headline ranking quantity.
    pub oos_expectancy_cost_aware: f64,
    pub profit_factor: f64,
    pub cvar5: f64,
    pub mar: f64,
    /// Per-regime OOS contribution map.
    pub regime_contrib: std::collections::BTreeMap<String, f64>,
    pub n_regimes_positive: i64,
    /// Whether the hard promotion gate passed (DSR>0, PBO<0.5, OOS exp>0, >=2 regimes).
    pub passed_gate: bool,
    /// The full gate decision (per-condition booleans + reasons), for reporting.
    pub gate: GateDecision,
    /// Number of labeled entries that fed the validation (cohort size).
    pub n_entries: usize,
}

impl OosScore {
    /// Build the score from a finished validation. This is the ONLY constructor, and it takes
    /// no in-sample input — enforcing OOS-only ranking at the type level.
    pub fn from_validation(
        strategy_id: StrategyId,
        validation: &ValidationResult,
        gate: GateDecision,
        n_entries: usize,
    ) -> Self {
        OosScore {
            strategy_id,
            dsr: validation.dsr,
            pbo: validation.pbo,
            oos_expectancy_cost_aware: validation.oos_expectancy_cost_aware,
            profit_factor: validation.profit_factor,
            cvar5: validation.cvar5,
            mar: validation.mar,
            regime_contrib: validation.regime_contrib.clone(),
            n_regimes_positive: validation.n_regimes_positive,
            passed_gate: gate.passed,
            gate,
            n_entries,
        }
    }

    /// Survivor rule: a strategy is kept if the hard gate passed, OR it has a positive
    /// cost-aware OOS expectancy with a leakage-clean signature (DSR>0 and PBO<0.5). The second
    /// arm keeps "promising but not yet promotable" genomes to mutate, while still demanding the
    /// OOS expectancy be positive and the overfit signature be absent.
    pub fn is_survivor(&self) -> bool {
        if self.passed_gate {
            return true;
        }
        self.oos_expectancy_cost_aware > 0.0 && self.dsr > 0.0 && self.pbo < 0.5
    }

    /// Total order for the leaderboard: gate-passers first, then higher cost-aware OOS
    /// expectancy, then higher DSR, then lower PBO. Every tiebreaker is an OOS quantity.
    pub fn rank_key(&self) -> impl PartialOrd + Clone {
        // Higher is better for the first three; lower PBO is better (negate it).
        (
            self.passed_gate as u8 as f64,
            self.oos_expectancy_cost_aware,
            self.dsr,
            -self.pbo,
        )
    }

    /// Compare two scores by the OOS ranking, best first.
    pub fn cmp_best_first(&self, other: &Self) -> Ordering {
        let a = (
            self.passed_gate as u8,
            self.oos_expectancy_cost_aware,
            self.dsr,
            -self.pbo,
        );
        let b = (
            other.passed_gate as u8,
            other.oos_expectancy_cost_aware,
            other.dsr,
            -other.pbo,
        );
        // Descending (best first); fall back to Equal on NaN.
        b.partial_cmp(&a).unwrap_or(Ordering::Equal)
    }
}

/// CPCV / DSR shaping for the OOS validation. Modest defaults that work for single-ticker
/// cohorts of a few dozen to a few hundred entries.
#[derive(Debug, Clone, Copy)]
pub struct ScoreConfig {
    pub n_groups: u32,
    pub k_test_groups: u32,
    pub n_trials: u32,
}

impl Default for ScoreConfig {
    fn default() -> Self {
        ScoreConfig {
            n_groups: 6,
            k_test_groups: 2,
            n_trials: 12,
        }
    }
}

/// Score a genome's labeled dataset out-of-sample. Returns `Ok(None)` when the dataset is too
/// small to validate (the caller logs + skips); returns `Err` only on a real transport/IO
/// failure (treated as fail-closed by the caller).
pub async fn score_oos(
    harness: &ValidationHarness,
    strategy_id: StrategyId,
    rows: &[DatasetRow],
    profile: &HorizonProfile,
    cfg: ScoreConfig,
) -> se_core::Result<Option<OosScore>> {
    if rows.len() < MIN_ENTRIES_TO_VALIDATE {
        return Ok(None);
    }
    // Guard: n_groups must not exceed the number of observations.
    let n_groups = cfg.n_groups.min(rows.len() as u32).max(2);
    let k_test = cfg.k_test_groups.min(n_groups.saturating_sub(1)).max(1);

    let name = format!("search_{strategy_id}.parquet");
    let outcome = harness
        .evaluate(rows, profile, &name, n_groups, k_test, cfg.n_trials)
        .await?;

    Ok(Some(OosScore::from_validation(
        strategy_id,
        &outcome.validation,
        outcome.decision,
        rows.len(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn vr(dsr: f64, pbo: f64, exp: f64, n_pos: i64) -> ValidationResult {
        ValidationResult {
            dsr,
            pbo,
            oos_expectancy_cost_aware: exp,
            profit_factor: 1.2,
            cvar5: -0.5,
            mar: 0.3,
            regime_contrib: BTreeMap::new(),
            n_regimes_positive: n_pos,
            passed_gate: false,
        }
    }

    #[test]
    fn ranking_is_oos_only_and_gate_first() {
        let pass = OosScore::from_validation(
            StrategyId::new(),
            &vr(0.5, 0.1, 0.05, 3),
            se_validation::PromotionGate::evaluate(&vr(0.5, 0.1, 0.05, 3)),
            80,
        );
        let high_exp_no_gate = OosScore::from_validation(
            StrategyId::new(),
            &vr(0.1, 0.6, 0.20, 1), // higher expectancy but fails gate
            se_validation::PromotionGate::evaluate(&vr(0.1, 0.6, 0.20, 1)),
            80,
        );
        // Gate-passer ranks ahead of the higher-expectancy gate-failer.
        assert_eq!(pass.cmp_best_first(&high_exp_no_gate), Ordering::Less);
        let mut v = [high_exp_no_gate.clone(), pass.clone()];
        v.sort_by(|a, b| a.cmp_best_first(b));
        assert!(v[0].passed_gate);
    }

    #[test]
    fn survivor_rule() {
        let promoted = OosScore::from_validation(
            StrategyId::new(),
            &vr(0.5, 0.1, 0.05, 3),
            se_validation::PromotionGate::evaluate(&vr(0.5, 0.1, 0.05, 3)),
            80,
        );
        assert!(promoted.is_survivor() && promoted.passed_gate);

        // Positive OOS expectancy, clean signature, but only 1 regime -> not promoted but kept.
        let promising = OosScore::from_validation(
            StrategyId::new(),
            &vr(0.3, 0.2, 0.03, 1),
            se_validation::PromotionGate::evaluate(&vr(0.3, 0.2, 0.03, 1)),
            80,
        );
        assert!(!promising.passed_gate && promising.is_survivor());

        // Negative OOS expectancy -> killed.
        let dead = OosScore::from_validation(
            StrategyId::new(),
            &vr(-0.2, 0.7, -0.04, 0),
            se_validation::PromotionGate::evaluate(&vr(-0.2, 0.7, -0.04, 0)),
            80,
        );
        assert!(!dead.is_survivor());
    }
}
