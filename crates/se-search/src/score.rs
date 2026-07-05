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

/// z for a one-sided ~95% Wilson lower bound on a binomial proportion.
pub const WILSON_Z_95: f64 = 1.96;

/// The minimum WILSON LOWER BOUND on OOS precision a genome must clear to be promoted. Gating on
/// the point estimate is textbook optimizer's-curse: the search maximizes precision over many
/// genomes on tiny acted cohorts, so a "best 0.73 on n=8" is statistically consistent with the
/// ~0.5 mean (Wilson half-width ≈ ±0.25-0.30). Requiring the *lower bound* of the (net-of-cost)
/// precision interval to clear this floor keeps only genomes whose edge survives sampling
/// uncertainty — a large-n 0.55 passes, a small-n 0.73 fluke does not. Tune jointly with history
/// depth (a strict floor on a shallow single-regime window can starve promotions).
pub const MIN_PROMOTE_PRECISION_LB: f64 = 0.45;

/// One-sided Wilson score-interval lower bound for a binomial proportion `p` observed over `n`
/// trials, at confidence `z` (see [`WILSON_Z_95`]). Unlike the raw `p ± z·SE` Wald interval, the
/// Wilson bound is well-behaved for small `n` and near 0/1 — exactly the tiny-cohort regime where
/// the search's best-of-many precision is most optimistically biased. Returns `0.0` for `n <= 0`
/// (no evidence) and clamps to `[0, 1]`.
pub fn wilson_lower_bound(p: f64, n: i64, z: f64) -> f64 {
    if n <= 0 || !p.is_finite() {
        return 0.0;
    }
    let n = n as f64;
    let p = p.clamp(0.0, 1.0);
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let centre = p + z2 / (2.0 * n);
    let margin = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
    ((centre - margin) / denom).clamp(0.0, 1.0)
}

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
    /// Precision on a strict time-ordered forward holdout (train earliest 70%, measure latest 30%).
    /// The durability metric — separates a real forward edge from bull-window regime-fitting.
    pub precision_forward: f64,
    /// Cost-aware expectancy (R) on the forward holdout.
    pub expectancy_forward: f64,
    /// Acted-trade count in the forward holdout (small ⇒ low-confidence durability estimate).
    pub n_forward: i64,
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
            precision_forward: validation.precision_forward,
            expectancy_forward: validation.expectancy_forward,
            n_forward: validation.n_forward,
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

/// Threshold rounding precision (decimal places) for the genome signature. A stable, coarse
/// grid so two genomes whose thresholds differ only in noise below this precision collapse to
/// the same signature. Matches the 4-decimal display precision of [`se_core::Predicate::describe`]
/// so signature-equal genomes also read identically by eye.
const SIGNATURE_THRESHOLD_DECIMALS: i32 = 4;

/// Round a threshold to the stable signature grid. `-0.0` is normalized to `0.0` so the two zeros
/// never produce different signatures.
fn round_threshold(t: f64) -> f64 {
    if !t.is_finite() {
        // NaN/inf can't fire meaningfully, but keep the signature total/deterministic rather than
        // panicking: map every non-finite threshold to a single sentinel.
        return 0.0;
    }
    let f = 10f64.powi(SIGNATURE_THRESHOLD_DECIMALS);
    let r = (t * f).round() / f;
    if r == 0.0 {
        0.0
    } else {
        r
    }
}

/// A canonical, ORDER-INDEPENDENT signature for a genome: two genomes that would fire and manage
/// risk identically map to the same string. It is a stable normalization of
/// `(side, horizon, sorted normalized predicates (layer, feature_key, op, rounded threshold),
/// risk model)`.
///
/// This is the deduplication key for the search. It is intentionally stronger than
/// [`se_core::Genome::describe`], which joins predicates in their stored vector order (so the same
/// conjunction written in a different order yields a different string). Because predicates form a
/// conjunction, their order does not affect firing; the signature sorts them so it does not affect
/// the key either. Thresholds are rounded to [`SIGNATURE_THRESHOLD_DECIMALS`] so genomes that
/// differ only by sub-grid threshold noise (which would fire identically on the empirical grid)
/// collapse together. Side, horizon and the full risk geometry are included so genuinely different
/// genomes stay distinct.
///
/// Deterministic and side-effect-free; safe to use as a `BTreeSet<String>`/`HashMap` key.
pub fn genome_signature(genome: &se_core::Genome) -> String {
    // Normalize each predicate to a canonical, comparable token, then sort so order can't matter.
    let mut preds: Vec<String> = genome
        .predicates
        .iter()
        .map(|p| {
            format!(
                "{}|{}|{}|{:.*}",
                p.layer.as_str(),
                p.feature_key,
                p.op.as_str(),
                SIGNATURE_THRESHOLD_DECIMALS as usize,
                round_threshold(p.threshold),
            )
        })
        .collect();
    preds.sort();
    format!(
        "{}::{}::[{}]::{}",
        genome.side.sign() as i64, // Long=+1, Short=-1: a stable, side-distinguishing token.
        genome.horizon.as_str(),
        preds.join("&"),
        // The risk geometry is part of what fires/manages a trade, so it is part of identity.
        // `describe` is a stable, human-readable normalization of the full stop/target1/target2.
        genome.risk.describe(),
    )
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
            precision_forward: 0.0,
            expectancy_forward: 0.0,
            n_forward: 0,
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
    fn genome_signature_is_order_independent() {
        use se_core::{CmpOp, Genome, Horizon, Layer, Predicate, Side};

        let p_trigger = Predicate {
            layer: Layer::Trigger,
            feature_key: "trigger.rsi14".into(),
            op: CmpOp::Gt,
            threshold: 55.0,
        };
        let p_regime = Predicate {
            layer: Layer::Regime,
            feature_key: "regime.adx".into(),
            op: CmpOp::Lt,
            threshold: 20.0,
        };
        // Same predicate SET (a conjunction), stored in two different orders.
        let a = Genome::new(
            Side::Long,
            Horizon::Swing,
            vec![p_trigger.clone(), p_regime.clone()],
        );
        let b = Genome::new(Side::Long, Horizon::Swing, vec![p_regime, p_trigger]);
        assert_ne!(
            a.describe(),
            b.describe(),
            "describe is order-DEPENDENT (this is exactly why we need a canonical signature)"
        );
        assert_eq!(
            genome_signature(&a),
            genome_signature(&b),
            "signature must be order-independent for the same conjunction"
        );
    }

    #[test]
    fn genome_signature_distinguishes_genuinely_different_genomes() {
        use se_core::{CmpOp, Genome, Horizon, Layer, Predicate, Side};

        let base = Genome::new(
            Side::Long,
            Horizon::Swing,
            vec![Predicate {
                layer: Layer::Trigger,
                feature_key: "trigger.rsi14".into(),
                op: CmpOp::Gt,
                threshold: 55.0,
            }],
        );
        let base_sig = genome_signature(&base);

        // Different side.
        let diff_side = Genome::new(Side::Short, Horizon::Swing, base.predicates.clone());
        assert_ne!(genome_signature(&diff_side), base_sig, "side must matter");

        // Different horizon (also changes the profile-derived risk, doubly distinct).
        let diff_horizon = Genome::new(Side::Long, Horizon::Day, base.predicates.clone());
        assert_ne!(
            genome_signature(&diff_horizon),
            base_sig,
            "horizon must matter"
        );

        // Different op.
        let mut lt = base.predicates.clone();
        lt[0].op = CmpOp::Lt;
        let diff_op = Genome::new(Side::Long, Horizon::Swing, lt);
        assert_ne!(genome_signature(&diff_op), base_sig, "op must matter");

        // Materially different threshold (well beyond the rounding grid).
        let mut thr = base.predicates.clone();
        thr[0].threshold = 40.0;
        let diff_thr = Genome::new(Side::Long, Horizon::Swing, thr);
        assert_ne!(
            genome_signature(&diff_thr),
            base_sig,
            "a materially different threshold must matter"
        );

        // Different risk geometry, everything else equal.
        let diff_risk = se_core::Genome::with_risk(
            Side::Long,
            Horizon::Swing,
            base.predicates.clone(),
            se_core::RiskModel::new(
                se_core::StopSpec::atr(2.0),
                se_core::TargetSpec::r_multiple(4.0),
                None,
            ),
        );
        assert_ne!(
            genome_signature(&diff_risk),
            base_sig,
            "risk geometry must matter"
        );
    }

    #[test]
    fn genome_signature_collapses_subgrid_threshold_noise() {
        use se_core::{CmpOp, Genome, Horizon, Layer, Predicate, Side};

        let mk = |thr: f64| {
            Genome::new(
                Side::Long,
                Horizon::Swing,
                vec![Predicate {
                    layer: Layer::Trigger,
                    feature_key: "trigger.rsi14".into(),
                    op: CmpOp::Gt,
                    threshold: thr,
                }],
            )
        };
        // Two thresholds equal well below the 4-decimal rounding grid map to the same signature.
        assert_eq!(
            genome_signature(&mk(55.000_001)),
            genome_signature(&mk(55.000_002)),
            "sub-grid threshold noise must not create a distinct signature"
        );
        // A -0.0 threshold must not differ from +0.0.
        assert_eq!(genome_signature(&mk(-0.0)), genome_signature(&mk(0.0)));
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

    #[test]
    fn wilson_lower_bound_penalizes_small_n_and_tightens_with_n() {
        // Degenerate / no-evidence inputs.
        assert_eq!(wilson_lower_bound(0.7, 0, WILSON_Z_95), 0.0);
        assert_eq!(wilson_lower_bound(f64::NAN, 100, WILSON_Z_95), 0.0);

        // The lower bound is always <= the point estimate, and rises toward it as n grows.
        let lb_small = wilson_lower_bound(0.7, 8, WILSON_Z_95);
        let lb_big = wilson_lower_bound(0.7, 400, WILSON_Z_95);
        assert!(lb_small < 0.7 && lb_big < 0.7);
        assert!(
            lb_big > lb_small,
            "more evidence -> tighter (higher) lower bound"
        );

        // The optimizer's-curse case is caught: 0.70 on n=8 falls below the promote floor, while
        // a genuine 0.55 on a large cohort clears it.
        assert!(
            lb_small < MIN_PROMOTE_PRECISION_LB,
            "small-n 0.70 fluke is rejected"
        );
        assert!(
            wilson_lower_bound(0.55, 400, WILSON_Z_95) >= MIN_PROMOTE_PRECISION_LB,
            "large-n 0.55 edge clears the floor"
        );

        // Bounded in [0, 1] even at extremes.
        assert!((0.0..=1.0).contains(&wilson_lower_bound(1.0, 5, WILSON_Z_95)));
        assert!((0.0..=1.0).contains(&wilson_lower_bound(0.0, 5, WILSON_Z_95)));
    }
}
