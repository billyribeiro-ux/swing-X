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

use crate::{Horizon, Layer, Side, StrategyId};

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
        format!("{} {} {:.4}", self.feature_key, self.op.as_str(), self.threshold)
    }
}

/// The evolvable genome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Genome {
    pub side: Side,
    pub horizon: Horizon,
    pub predicates: Vec<Predicate>,
}

impl Genome {
    pub fn new(side: Side, horizon: Horizon, predicates: Vec<Predicate>) -> Self {
        Genome {
            side,
            horizon,
            predicates,
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
        self.predicates.iter().map(|p| p.feature_key.as_str()).collect()
    }

    pub fn describe(&self) -> String {
        let conds = self
            .predicates
            .iter()
            .map(|p| p.describe())
            .collect::<Vec<_>>()
            .join(" AND ");
        format!("{:?} {} :: {}", self.side, self.horizon.as_str(), conds)
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
