//! Strategy genome — the mutable unit the search loop evolves.
//!
//! A genome is a conjunction of feature predicates across the conditional layers
//! (regime × location × trigger, gated by tradeability, modified by events), plus a
//! side and horizon. It "fires" at a decision bar when every predicate holds against
//! the PIT-safe feature values at that bar. Search mutates predicates (perturb a
//! threshold, swap a feature, add/remove a condition); the OOS scoreboard — never the
//! in-sample fit — decides what survives.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::{Horizon, Layer, RiskModel, Side, StrategyId};

/// Comparison operator for a feature predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CmpOp {
    Gt,
    Lt,
    Gte,
    Lte,
}

impl CmpOp {
    pub fn apply(self, value: f64, threshold: f64) -> bool {
        match self {
            CmpOp::Gt => value > threshold,
            CmpOp::Lt => value < threshold,
            CmpOp::Gte => value >= threshold,
            CmpOp::Lte => value <= threshold,
        }
    }
    pub const fn as_str(self) -> &'static str {
        match self {
            CmpOp::Gt => ">",
            CmpOp::Lt => "<",
            CmpOp::Gte => ">=",
            CmpOp::Lte => "<=",
        }
    }
}

/// One condition on a single feature.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Predicate {
    pub layer: Layer,
    pub feature_key: String,
    pub op: CmpOp,
    pub threshold: f64,
}

impl Predicate {
    /// True iff `value` satisfies the predicate.
    pub fn holds(&self, value: f64) -> bool {
        self.op.apply(value, self.threshold)
    }

    pub fn describe(&self) -> String {
        format!(
            "{} {} {:.4}",
            self.feature_key,
            self.op.as_str(),
            self.threshold
        )
    }
}

/// The evolvable genome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Genome {
    pub side: Side,
    pub horizon: Horizon,
    pub predicates: Vec<Predicate>,
    /// Stop/target geometry. The operator sets a starting model; the search evolves it (unless
    /// locked) and the OOS scoreboard keeps the best risk geometry.
    ///
    /// Old genomes persisted before this field existed have no `risk` key in their jsonb, so we
    /// supply a fixed sensible default ([`RiskModel::default_const`] — a 1-ATR stop with 2R/3R
    /// targets) on deserialize. Serde defaults can't read sibling fields, so this is horizon
    /// agnostic by necessity; freshly seeded genomes get a horizon-aware model from the search.
    #[serde(default = "genome_default_risk")]
    pub risk: RiskModel,
}

/// The serde default for [`Genome::risk`] when an old persisted genome lacks the field.
fn genome_default_risk() -> RiskModel {
    RiskModel::default_const()
}

impl Genome {
    pub fn new(side: Side, horizon: Horizon, predicates: Vec<Predicate>) -> Self {
        Genome {
            side,
            horizon,
            predicates,
            risk: RiskModel::from_profile(&crate::HorizonProfile::for_horizon(horizon)),
        }
    }

    /// Construct a genome with an explicit risk model (used by the search when seeding/mutating
    /// risk geometry).
    pub fn with_risk(
        side: Side,
        horizon: Horizon,
        predicates: Vec<Predicate>,
        risk: RiskModel,
    ) -> Self {
        Genome {
            side,
            horizon,
            predicates,
            risk,
        }
    }

    /// Fire iff EVERY predicate holds against the provided feature map. A predicate whose
    /// feature is absent (unavailable/stale) does NOT hold — the genome will not fire on
    /// missing data, never guessing.
    pub fn fires(&self, features: &BTreeMap<String, f64>) -> bool {
        !self.predicates.is_empty()
            && self
                .predicates
                .iter()
                .all(|p| features.get(&p.feature_key).is_some_and(|v| p.holds(*v)))
    }

    /// The set of feature keys this genome reads (for attribution + decay monitoring).
    pub fn feature_keys(&self) -> Vec<&str> {
        self.predicates
            .iter()
            .map(|p| p.feature_key.as_str())
            .collect()
    }

    pub fn describe(&self) -> String {
        let conds = self
            .predicates
            .iter()
            .map(|p| p.describe())
            .collect::<Vec<_>>()
            .join(" AND ");
        format!(
            "{:?} {} :: {} [{}]",
            self.side,
            self.horizon.as_str(),
            conds,
            self.risk.describe()
        )
    }
}

/// Lifecycle status of a strategy in the population.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategyStatus {
    Candidate,
    Promoted,
    Quarantined,
    Demoted,
    Retired,
}

impl StrategyStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            StrategyStatus::Candidate => "candidate",
            StrategyStatus::Promoted => "promoted",
            StrategyStatus::Quarantined => "quarantined",
            StrategyStatus::Demoted => "demoted",
            StrategyStatus::Retired => "retired",
        }
    }
    /// Whether a strategy in this status may surface live signals.
    pub const fn can_surface(self) -> bool {
        matches!(self, StrategyStatus::Promoted)
    }
}

/// A strategy = genome + identity + lifecycle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Strategy {
    pub id: StrategyId,
    pub genome: Genome,
    pub status: StrategyStatus,
    pub generation: u32,
    pub parent: Option<StrategyId>,
}

impl Strategy {
    pub fn new(genome: Genome) -> Self {
        Strategy {
            id: StrategyId::new(),
            genome,
            status: StrategyStatus::Candidate,
            generation: 0,
            parent: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HorizonProfile, RiskModel, StopSpec, TargetSpec};

    fn pred() -> Predicate {
        Predicate {
            layer: Layer::Trigger,
            feature_key: "trigger.rsi14".into(),
            op: CmpOp::Gt,
            threshold: 55.0,
        }
    }

    #[test]
    fn new_genome_gets_profile_risk() {
        let g = Genome::new(Side::Long, Horizon::Swing, vec![pred()]);
        assert_eq!(
            g.risk,
            RiskModel::from_profile(&HorizonProfile::for_horizon(Horizon::Swing))
        );
    }

    #[test]
    fn old_genome_without_risk_field_still_deserializes() {
        // Exactly the jsonb shape persisted before the `risk` field existed.
        let legacy = serde_json::json!({
            "side": "Long",
            "horizon": "swing",
            "predicates": [
                {"layer": "trigger", "feature_key": "trigger.rsi14", "op": "gt", "threshold": 55.0}
            ]
        });
        let g: Genome = serde_json::from_value(legacy).expect("legacy genome must deserialize");
        // Falls back to the fixed sensible default.
        assert_eq!(g.risk, RiskModel::default_const());
        assert_eq!(g.predicates.len(), 1);
    }

    #[test]
    fn genome_round_trips_with_risk() {
        let g = Genome::with_risk(
            Side::Short,
            Horizon::Day,
            vec![pred()],
            RiskModel::new(StopSpec::fixed(5.35), TargetSpec::r_multiple(2.0), None),
        );
        let json = serde_json::to_value(&g).unwrap();
        let back: Genome = serde_json::from_value(json).unwrap();
        assert_eq!(g, back);
    }

    #[test]
    fn describe_includes_risk() {
        let g = Genome::with_risk(
            Side::Long,
            Horizon::Swing,
            vec![pred()],
            RiskModel::new(StopSpec::atr(1.0), TargetSpec::r_multiple(2.0), None),
        );
        assert!(g.describe().contains("stop=1ATR"), "{}", g.describe());
    }
}
