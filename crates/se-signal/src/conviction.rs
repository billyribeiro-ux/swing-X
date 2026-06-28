//! Conviction derivation — honest, clearly-labeled, never invented.
//!
//! We do NOT fit a per-signal probability calibrator in v1. Instead conviction is a COHORT
//! HIT-RATE PROXY derived from the strategy's out-of-sample cost-aware expectancy and the
//! reward:risk geometry of the trade. The label string returned alongside makes the proxy
//! explicit so a reader never mistakes it for a fitted calibrated probability.
//!
//! Derivation: for a fixed-R setup with reward:risk `b` (= target distance / stop distance),
//! a per-trade expectancy `E` in R implies a break-even-consistent hit rate
//! `p = (E + 1) / (b + 1)` (from `E = p·b − (1−p)·1`). We clamp to `[0, 1]`. This is a
//! cohort-level implied probability, not a calibrated per-instance one.

/// A conviction value plus the human label describing exactly how it was derived.
#[derive(Debug, Clone, PartialEq)]
pub struct Conviction {
    /// Implied cohort hit-rate in `[0, 1]`.
    pub value: f64,
    /// Provenance label, surfaced in the signal (e.g. in `lead_time`/notes).
    pub label: String,
}

/// Derive conviction from the OOS cohort expectancy (R) and the trade's reward:risk ratio.
///
/// `rr` is the first-target reward:risk (`target_atr_mult / stop_atr_mult`). A non-finite or
/// non-positive `rr` falls back to `1.0`. The result is clamped to `[0, 1]`.
pub fn from_cohort(oos_expectancy_r: f64, rr: f64) -> Conviction {
    let b = if rr.is_finite() && rr > 0.0 { rr } else { 1.0 };
    let raw = (oos_expectancy_r + 1.0) / (b + 1.0);
    let value = raw.clamp(0.0, 1.0);
    Conviction {
        value,
        label: "cohort-implied (OOS expectancy / R-geometry; not a fitted per-signal calibrator)"
            .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_expectancy_maps_to_breakeven_hit_rate() {
        // rr=2: break-even hit rate is 1/(2+1) = 0.333..., expectancy 0 -> p = (0+1)/3.
        let c = from_cohort(0.0, 2.0);
        assert!((c.value - 1.0 / 3.0).abs() < 1e-9);
        assert!(c.label.contains("not a fitted"));
    }

    #[test]
    fn positive_expectancy_raises_conviction() {
        let low = from_cohort(0.0, 2.0).value;
        let high = from_cohort(0.4, 2.0).value;
        assert!(high > low);
        assert!((0.0..=1.0).contains(&high));
    }

    #[test]
    fn clamps_and_handles_bad_rr() {
        assert_eq!(from_cohort(5.0, 2.0).value, 1.0); // clamps high
        assert_eq!(from_cohort(-5.0, 2.0).value, 0.0); // clamps low
                                                       // rr <= 0 falls back to 1.0 -> p = (0+1)/2 = 0.5.
        assert!((from_cohort(0.0, -1.0).value - 0.5).abs() < 1e-9);
    }
}
