//! Canonical regime taxonomy (v1). The regime classifier maps the Layer-1 feature
//! vector onto one of these, with a calibrated probability. The set is deliberately
//! small and interpretable so it can be sanity-checked against known historical
//! events (COVID crash, 2022 bear, OPEX clusters) in P2.
//!
//! `OutOfDistribution` is first-class: when the nearest-cohort distance is too large
//! the system suppresses signals and logs rather than guessing.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

use crate::error::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegimeLabel {
    /// Dealers net short gamma → trend / continuation.
    ShortGamma,
    /// Dealers net long gamma → mean-revert / fade.
    LongGamma,
    /// Realized + implied vol expanding.
    VolExpansion,
    /// Vol compressing / coiling.
    VolCompression,
    /// Risk-off: credit widening, breadth weak, cross-asset stress.
    RiskOff,
    /// Risk-on: credit tight, breadth strong.
    RiskOn,
    /// Ambiguous / regime in transition.
    Transition,
    /// No comparable cohort — suppress signals, do not guess.
    OutOfDistribution,
}

impl RegimeLabel {
    pub const ALL: [RegimeLabel; 8] = [
        RegimeLabel::ShortGamma,
        RegimeLabel::LongGamma,
        RegimeLabel::VolExpansion,
        RegimeLabel::VolCompression,
        RegimeLabel::RiskOff,
        RegimeLabel::RiskOn,
        RegimeLabel::Transition,
        RegimeLabel::OutOfDistribution,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            RegimeLabel::ShortGamma => "short_gamma",
            RegimeLabel::LongGamma => "long_gamma",
            RegimeLabel::VolExpansion => "vol_expansion",
            RegimeLabel::VolCompression => "vol_compression",
            RegimeLabel::RiskOff => "risk_off",
            RegimeLabel::RiskOn => "risk_on",
            RegimeLabel::Transition => "transition",
            RegimeLabel::OutOfDistribution => "out_of_distribution",
        }
    }

    /// Whether signals may be surfaced under this regime at all.
    pub const fn is_tradeable(self) -> bool {
        !matches!(self, RegimeLabel::OutOfDistribution)
    }
}

impl fmt::Display for RegimeLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for RegimeLabel {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let low = s.trim().to_ascii_lowercase();
        RegimeLabel::ALL
            .into_iter()
            .find(|r| r.as_str() == low)
            .ok_or_else(|| Error::Parse(format!("unknown regime: {s}")))
    }
}
