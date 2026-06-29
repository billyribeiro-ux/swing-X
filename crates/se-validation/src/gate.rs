//! The hard promotion gate — the Rust-side authoritative re-check.
//!
//! A candidate is promotable ONLY when ALL of:
//!
//! ```text
//! dsr > 0  &&  pbo < 0.5  &&  oos_expectancy_cost_aware > 0  &&  n_regimes_positive >= 2
//! ```
//!
//! These thresholds match `ml-worker/src/se_ml/gates.py` byte-for-byte, and the gate is
//! re-derived here independently as defence-in-depth: the Rust side never trusts the
//! worker's own `passed_gate` flag, it re-evaluates from the raw metrics.
//!
//! `win_rate` deliberately does NOT appear here, nor anywhere as a selection input — it is
//! a misleading objective (a high win-rate, negative-expectancy strategy must be rejected).
//!
//! Fail-closed: see [`PromotionGate::evaluate_opt`] — a `None` (absent/errored validation)
//! is NOT passed.

use se_mlclient::ValidationResult;

/// Promotion thresholds. Constants so both language sides agree exactly.
pub const DSR_MIN: f64 = 0.0;
pub const PBO_MAX: f64 = 0.5;
pub const OOS_EXPECTANCY_MIN: f64 = 0.0;
pub const MIN_POSITIVE_REGIMES: i64 = 2;

/// The gate's decision, with each sub-condition exposed for attribution/logging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateDecision {
    /// Overall: true only if every sub-condition holds.
    pub passed: bool,
    pub dsr_ok: bool,
    pub pbo_ok: bool,
    pub expectancy_ok: bool,
    pub regime_ok: bool,
    /// Human-readable reasons for any FAILED sub-condition (empty when `passed`).
    pub reasons: Vec<String>,
}

impl GateDecision {
    /// A decision that fails closed with a single explanatory reason.
    pub fn fail_closed(reason: impl Into<String>) -> Self {
        GateDecision {
            passed: false,
            dsr_ok: false,
            pbo_ok: false,
            expectancy_ok: false,
            regime_ok: false,
            reasons: vec![reason.into()],
        }
    }
}

/// The stateless promotion gate.
#[derive(Debug, Clone, Copy, Default)]
pub struct PromotionGate;

impl PromotionGate {
    /// Evaluate the gate against a concrete [`ValidationResult`].
    ///
    /// Re-derives `passed` from the raw metrics; the worker's own `passed_gate` is ignored
    /// here on purpose (defence-in-depth).
    pub fn evaluate(result: &ValidationResult) -> GateDecision {
        let dsr_ok = result.dsr > DSR_MIN;
        let pbo_ok = result.pbo < PBO_MAX;
        let expectancy_ok = result.oos_expectancy_cost_aware > OOS_EXPECTANCY_MIN;
        let regime_ok = result.n_regimes_positive >= MIN_POSITIVE_REGIMES;

        let mut reasons = Vec::new();
        if !dsr_ok {
            reasons.push(format!("dsr {:.4} <= {DSR_MIN}", result.dsr));
        }
        if !pbo_ok {
            reasons.push(format!("pbo {:.4} >= {PBO_MAX}", result.pbo));
        }
        if !expectancy_ok {
            reasons.push(format!(
                "oos_expectancy_cost_aware {:.4} <= {OOS_EXPECTANCY_MIN}",
                result.oos_expectancy_cost_aware
            ));
        }
        if !regime_ok {
            reasons.push(format!(
                "n_regimes_positive {} < {MIN_POSITIVE_REGIMES}",
                result.n_regimes_positive
            ));
        }

        let passed = dsr_ok && pbo_ok && expectancy_ok && regime_ok;
        GateDecision {
            passed,
            dsr_ok,
            pbo_ok,
            expectancy_ok,
            regime_ok,
            reasons,
        }
    }

    /// Fail-closed evaluation: `None` (an absent or errored validation) is NOT passed.
    pub fn evaluate_opt(result: Option<&ValidationResult>) -> GateDecision {
        match result {
            Some(r) => Self::evaluate(r),
            None => GateDecision::fail_closed("no validation result (fail-closed)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn result(dsr: f64, pbo: f64, exp: f64, n_pos: i64) -> ValidationResult {
        ValidationResult {
            dsr,
            pbo,
            oos_expectancy_cost_aware: exp,
            profit_factor: 1.0,
            cvar5: 0.0,
            mar: 0.0,
            regime_contrib: BTreeMap::new(),
            n_regimes_positive: n_pos,
            passed_gate: false, // deliberately wrong: gate must ignore it.
            precision_oos: 0.0,
            recall_oos: 0.0,
            act_threshold: 0.5,
            n_acted_oos: 0,
        }
    }

    #[test]
    fn all_conditions_met_passes() {
        let d = PromotionGate::evaluate(&result(0.5, 0.1, 0.04, 3));
        assert!(d.passed);
        assert!(d.dsr_ok && d.pbo_ok && d.expectancy_ok && d.regime_ok);
        assert!(d.reasons.is_empty());
    }

    #[test]
    fn ignores_worker_passed_flag() {
        // Worker says passed_gate=false but metrics are good -> gate independently passes.
        let mut r = result(0.5, 0.1, 0.04, 3);
        r.passed_gate = false;
        assert!(PromotionGate::evaluate(&r).passed);
    }

    #[test]
    fn dsr_boundary_zero_fails() {
        let d = PromotionGate::evaluate(&result(0.0, 0.1, 0.04, 3));
        assert!(!d.passed && !d.dsr_ok);
    }

    #[test]
    fn pbo_boundary_half_fails() {
        let d = PromotionGate::evaluate(&result(0.5, 0.5, 0.04, 3));
        assert!(!d.passed && !d.pbo_ok);
    }

    #[test]
    fn expectancy_zero_fails() {
        let d = PromotionGate::evaluate(&result(0.5, 0.1, 0.0, 3));
        assert!(!d.passed && !d.expectancy_ok);
    }

    #[test]
    fn one_regime_fails() {
        let d = PromotionGate::evaluate(&result(0.5, 0.1, 0.04, 1));
        assert!(!d.passed && !d.regime_ok);
        assert!(d.reasons.iter().any(|s| s.contains("n_regimes_positive")));
    }

    #[test]
    fn leaky_signature_dsr_nonpositive_or_pbo_high_is_rejected() {
        // The leak collapse signature: DSR deflates to <= 0 and OOS expectancy <= 0.
        let d = PromotionGate::evaluate(&result(-0.2, 0.7, -0.03, 1));
        assert!(!d.passed);
        assert!(!d.dsr_ok && !d.pbo_ok && !d.expectancy_ok && !d.regime_ok);
    }

    #[test]
    fn none_is_fail_closed() {
        let d = PromotionGate::evaluate_opt(None);
        assert!(!d.passed);
        assert!(!d.reasons.is_empty());
    }

    #[test]
    fn win_rate_is_not_referenced() {
        // Compile-time/structural guard: ValidationResult has no win_rate field, and the
        // gate reads only the four authorized metrics. This test documents the invariant.
        let r = result(0.5, 0.1, 0.04, 3);
        let _ = (
            r.dsr,
            r.pbo,
            r.oos_expectancy_cost_aware,
            r.n_regimes_positive,
        );
        assert!(PromotionGate::evaluate(&r).passed);
    }
}
