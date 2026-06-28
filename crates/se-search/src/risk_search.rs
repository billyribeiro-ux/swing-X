//! Making the risk geometry LEARNABLE.
//!
//! The operator sets a starting [`RiskModel`] (the ground rules). Unless the search is locked to
//! it, each seeded genome draws a risk geometry from a configurable space, mutation perturbs it,
//! and crossover mixes parents' models. The OOS scoreboard then keeps whatever maximizes
//! cost-aware out-of-sample expectancy — so the best stop/target trade-off wins on its own.
//!
//! All draws come from the same deterministic [`crate::rng::Rng`] used for predicate search, so
//! the whole search stays reproducible.

use se_core::{RiskModel, StopSpec, TargetSpec};

use crate::rng::Rng;

/// The space the search samples risk geometries from. Defaults match the design: stop ATR
/// multiples in `{0.5, 0.75, 1.0, 1.5, 2.0}`, targets as R-multiples in `{1.5, 2.0, 2.5, 3.0}`.
#[derive(Debug, Clone)]
pub struct RiskSpace {
    /// Candidate stop ATR multiples (used when the operator's stop is ATR-kind or unspecified).
    pub stop_atr_mults: Vec<f64>,
    /// Candidate target1 R-multiples.
    pub target_r_mults: Vec<f64>,
    /// The operator's configured ground-rule model, seeded around / mutated from.
    pub operator: RiskModel,
}

impl RiskSpace {
    /// Build the sampling space around the operator's ground-rule model.
    pub fn new(operator: RiskModel) -> Self {
        RiskSpace {
            stop_atr_mults: vec![0.5, 0.75, 1.0, 1.5, 2.0],
            target_r_mults: vec![1.5, 2.0, 2.5, 3.0],
            operator,
        }
    }

    /// Sample a risk model for a freshly seeded genome.
    ///
    /// If the operator pinned a non-ATR stop (fixed dollars / percent), we keep that stop KIND
    /// and jitter its magnitude a little, so seeding stays anchored to the operator's intent;
    /// the target R-multiple is always explored. If the operator's stop is ATR (or default), we
    /// sample a stop ATR multiple from the grid.
    pub fn sample(&self, rng: &mut Rng) -> RiskModel {
        let stop = match self.operator.stop {
            StopSpec::Fixed { dollars } => StopSpec::Fixed {
                dollars: jitter(dollars, rng),
            },
            StopSpec::Percent { pct } => StopSpec::Percent {
                pct: jitter(pct, rng),
            },
            StopSpec::Atr { .. } => StopSpec::Atr {
                mult: *rng.choice(&self.stop_atr_mults).unwrap_or(&1.0),
            },
        };
        let r1 = *rng.choice(&self.target_r_mults).unwrap_or(&2.0);
        let target1 = TargetSpec::RMultiple { r: r1 };
        // Second target a step beyond the first (keeps t2 > t1), capped at a sane ceiling.
        let target2 = Some(TargetSpec::RMultiple {
            r: (r1 + 1.0).min(5.0),
        });
        RiskModel::new(stop, target1, target2)
    }

    /// Mutate a risk model: nudge the stop ATR multiple, occasionally swap the stop KIND, and/or
    /// change the target R-multiple. Returns a new model. Non-ATR operator stops keep their kind
    /// unless a deliberate kind-swap fires.
    pub fn mutate(&self, rm: &RiskModel, rng: &mut Rng) -> RiskModel {
        let mut out = *rm;
        match rng.below(3) {
            // 0: perturb the stop.
            0 => {
                out.stop = self.mutate_stop(out.stop, rng);
            }
            // 1: change the target1 R-multiple.
            1 => {
                let r1 = *rng.choice(&self.target_r_mults).unwrap_or(&2.0);
                out.target1 = TargetSpec::RMultiple { r: r1 };
                if out.target2.is_some() {
                    out.target2 = Some(TargetSpec::RMultiple {
                        r: (r1 + 1.0).min(5.0),
                    });
                }
            }
            // 2: swap the stop kind (ATR <-> fixed <-> percent), keeping a comparable magnitude.
            _ => {
                out.stop = self.swap_stop_kind(out.stop, rng);
            }
        }
        out
    }

    /// Mix two parents' risk models: take each parent's stop or target independently.
    pub fn crossover(&self, a: &RiskModel, b: &RiskModel, rng: &mut Rng) -> RiskModel {
        let stop = if rng.chance(0.5) { a.stop } else { b.stop };
        let target1 = if rng.chance(0.5) {
            a.target1
        } else {
            b.target1
        };
        let target2 = if rng.chance(0.5) {
            a.target2
        } else {
            b.target2
        };
        RiskModel::new(stop, target1, target2)
    }

    fn mutate_stop(&self, stop: StopSpec, rng: &mut Rng) -> StopSpec {
        match stop {
            StopSpec::Atr { .. } => StopSpec::Atr {
                mult: *rng.choice(&self.stop_atr_mults).unwrap_or(&1.0),
            },
            StopSpec::Fixed { dollars } => StopSpec::Fixed {
                dollars: jitter(dollars, rng),
            },
            StopSpec::Percent { pct } => StopSpec::Percent {
                pct: jitter(pct, rng),
            },
        }
    }

    fn swap_stop_kind(&self, stop: StopSpec, rng: &mut Rng) -> StopSpec {
        // Cycle through kinds, drawing a sensible magnitude for the new kind.
        match (stop, rng.below(3)) {
            (StopSpec::Atr { .. }, 0) => StopSpec::Atr {
                mult: *rng.choice(&self.stop_atr_mults).unwrap_or(&1.0),
            },
            (StopSpec::Atr { .. }, 1) => StopSpec::Percent {
                pct: 1.0 + 2.0 * rng.uniform(),
            },
            (StopSpec::Atr { .. }, _) => StopSpec::Fixed {
                dollars: 2.0 + 6.0 * rng.uniform(),
            },
            (StopSpec::Fixed { dollars }, 0) => StopSpec::Fixed {
                dollars: jitter(dollars, rng),
            },
            (StopSpec::Fixed { .. }, _) => StopSpec::Atr {
                mult: *rng.choice(&self.stop_atr_mults).unwrap_or(&1.0),
            },
            (StopSpec::Percent { pct }, 0) => StopSpec::Percent {
                pct: jitter(pct, rng),
            },
            (StopSpec::Percent { .. }, _) => StopSpec::Atr {
                mult: *rng.choice(&self.stop_atr_mults).unwrap_or(&1.0),
            },
        }
    }
}

/// Jitter a positive magnitude by +/- ~25%, keeping it strictly positive.
fn jitter(v: f64, rng: &mut Rng) -> f64 {
    let factor = 0.75 + 0.5 * rng.uniform();
    (v.abs() * factor).max(1e-6)
}

/// Decide a seeded genome's risk geometry: pin to the operator's model when `lock_risk`, else
/// sample a fresh geometry from the space. Pure helper so the locked-vs-explore policy is unit
/// testable without a DB/worker.
pub fn seed_risk(space: &RiskSpace, lock_risk: bool, rng: &mut Rng) -> RiskModel {
    if lock_risk {
        space.operator
    } else {
        space.sample(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn space() -> RiskSpace {
        RiskSpace::new(RiskModel::default_const())
    }

    #[test]
    fn sample_is_deterministic_and_in_space() {
        let s = space();
        let a = {
            let mut r = Rng::seeded(7, 0);
            s.sample(&mut r)
        };
        let b = {
            let mut r = Rng::seeded(7, 0);
            s.sample(&mut r)
        };
        assert_eq!(a, b, "same seed must reproduce the same risk model");
        match a.stop {
            StopSpec::Atr { mult } => assert!(s.stop_atr_mults.contains(&mult)),
            other => panic!("default operator is ATR, got {other:?}"),
        }
        match a.target1 {
            TargetSpec::RMultiple { r } => assert!(s.target_r_mults.contains(&r)),
            other => panic!("target1 must be an R-multiple, got {other:?}"),
        }
    }

    #[test]
    fn sampling_explores_distinct_geometries() {
        let s = space();
        let mut r = Rng::seeded(42, 0);
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..40 {
            seen.insert(s.sample(&mut r).describe());
        }
        assert!(seen.len() > 1, "sampling should produce variety: {seen:?}");
    }

    #[test]
    fn fixed_operator_stop_keeps_kind_on_seed() {
        let op = RiskModel::new(StopSpec::fixed(5.35), TargetSpec::r_multiple(2.0), None);
        let s = RiskSpace::new(op);
        let mut r = Rng::seeded(3, 0);
        for _ in 0..20 {
            let rm = s.sample(&mut r);
            assert!(
                matches!(rm.stop, StopSpec::Fixed { .. }),
                "fixed operator stop must stay fixed when seeding, got {:?}",
                rm.stop
            );
        }
    }

    #[test]
    fn mutate_changes_something_and_stays_valid() {
        let s = space();
        let mut r = Rng::seeded(11, 0);
        let base = s.sample(&mut r);
        let mut changed = false;
        for _ in 0..50 {
            let m = s.mutate(&base, &mut r);
            // Stop distance must stay positive (it defines the R unit downstream).
            assert!(m.risk_distance(100.0, 2.0) > 0.0);
            if m != base {
                changed = true;
            }
        }
        assert!(changed, "mutation should eventually change the model");
    }

    #[test]
    fn crossover_picks_from_parents() {
        let s = space();
        let a = RiskModel::new(StopSpec::atr(0.5), TargetSpec::r_multiple(1.5), None);
        let b = RiskModel::new(StopSpec::atr(2.0), TargetSpec::r_multiple(3.0), None);
        let mut r = Rng::seeded(99, 0);
        let c = s.crossover(&a, &b, &mut r);
        assert!(c.stop == a.stop || c.stop == b.stop);
        assert!(c.target1 == a.target1 || c.target1 == b.target1);
    }

    #[test]
    fn locked_pins_every_genome_to_operator_model() {
        let op = RiskModel::new(StopSpec::fixed(5.0), TargetSpec::r_multiple(2.0), None);
        let s = RiskSpace::new(op);
        let mut r = Rng::seeded(1, 0);
        for _ in 0..20 {
            // lock_risk = true => always the operator's exact model, no exploration.
            assert_eq!(seed_risk(&s, true, &mut r), op);
        }
    }

    #[test]
    fn unlocked_explores_multiple_geometries() {
        let s = space();
        let mut r = Rng::seeded(1, 0);
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..40 {
            seen.insert(seed_risk(&s, false, &mut r).describe());
        }
        assert!(
            seen.len() > 1,
            "unlocked search must explore >1 risk geometry, saw {seen:?}"
        );
    }
}
