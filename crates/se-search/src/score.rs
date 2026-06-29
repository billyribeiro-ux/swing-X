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

/// The minimum number of OOS trades a genome must ACT ON (at the acting threshold τ\*) before it
/// may be promoted. A genome that fires (and so produces labeled entries) but acts on almost no
/// OOS trades has no real out-of-sample track record to promote on; it may still be kept as a
/// survivor/candidate to mutate, but it is not promotable. Paired with the actionable-predicate
/// guardrail in [`genome_has_actionable_predicate`] this kills degenerate regime/tradeability-only
/// promotions. See [`crate::population`].
pub const MIN_ACTED_TO_PROMOTE: usize = 8;

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
    /// OOS precision at τ\* — the north-star metric. Surfaced and persisted, NOT a ranking key
    /// (cost-aware OOS expectancy is already precision-conditioned upstream).
    pub precision_oos: f64,
    /// OOS recall at τ\*.
    pub recall_oos: f64,
    /// τ\* — the acting threshold in [0,1] the meta-label classifier acts at.
    pub act_threshold: f64,
    /// Count of OOS trades acted on at τ\* (the acted cohort). Gates promotability via
    /// [`MIN_ACTED_TO_PROMOTE`]; distinct from `n_entries` (total labeled).
    pub n_acted_oos: i64,
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
            precision_oos: validation.precision_oos,
            recall_oos: validation.recall_oos,
            act_threshold: validation.act_threshold,
            n_acted_oos: validation.n_acted_oos,
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

/// Whether a genome carries at least one ACTIONABLE entry condition — a predicate on the
/// [`Layer::Trigger`] or [`Layer::Location`] layer. Regime / tradeability / event predicates only
/// *condition* an entry; on their own they fire trivially (e.g. "we're in a bull regime") without
/// any real entry trigger, which is the root cause of degenerate promotions. A genome that lacks
/// any trigger/location predicate must not be promoted (it may still be kept as a survivor).
pub fn genome_has_actionable_predicate(genome: &se_core::Genome) -> bool {
    genome
        .predicates
        .iter()
        .any(|p| matches!(p.layer, se_core::Layer::Trigger | se_core::Layer::Location))
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
            precision_oos: 0.0,
            recall_oos: 0.0,
            act_threshold: 0.5,
            n_acted_oos: 0,
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

    #[test]
    fn from_validation_copies_precision_fields() {
        let mut v = vr(0.5, 0.1, 0.05, 3);
        v.precision_oos = 0.62;
        v.recall_oos = 0.41;
        v.act_threshold = 0.58;
        v.n_acted_oos = 23;
        let s = OosScore::from_validation(
            StrategyId::new(),
            &v,
            se_validation::PromotionGate::evaluate(&v),
            80,
        );
        assert_eq!(s.precision_oos, 0.62);
        assert_eq!(s.recall_oos, 0.41);
        assert_eq!(s.act_threshold, 0.58);
        assert_eq!(s.n_acted_oos, 23);
    }

    #[test]
    fn actionable_predicate_detection() {
        use se_core::{CmpOp, Genome, Horizon, Layer, Predicate, Side};

        let pred = |layer: Layer| Predicate {
            layer,
            feature_key: "k".into(),
            op: CmpOp::Gt,
            threshold: 0.0,
        };

        // Regime/tradeability/event only -> NOT actionable.
        let regime_only = Genome::new(
            Side::Long,
            Horizon::Swing,
            vec![pred(Layer::Regime), pred(Layer::Tradeability)],
        );
        assert!(!genome_has_actionable_predicate(&regime_only));

        // A trigger predicate -> actionable.
        let with_trigger = Genome::new(
            Side::Long,
            Horizon::Swing,
            vec![pred(Layer::Regime), pred(Layer::Trigger)],
        );
        assert!(genome_has_actionable_predicate(&with_trigger));

        // A location predicate -> actionable.
        let with_location = Genome::new(Side::Long, Horizon::Swing, vec![pred(Layer::Location)]);
        assert!(genome_has_actionable_predicate(&with_location));
    }
}
