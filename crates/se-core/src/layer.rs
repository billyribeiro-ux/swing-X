//! The conditional feature taxonomy is `regime × location × trigger`, gated by
//! tradeability and modified by events. Every [`crate::Feature`] is tagged with the
//! [`Layer`] it belongs to, so the store can partition by layer and the gate can
//! reason about regime-conditional contribution.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

use crate::error::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Layer {
    // `alias` accepts the legacy PascalCase form so genomes persisted before Layer
    // adopted snake_case still deserialize. New writes use snake_case (rename_all).
    /// L0 — only scan names with a large hand leaning on them.
    #[serde(alias = "Tradeability")]
    Tradeability,
    /// L1 — the conditioner: continuation-vs-fade.
    #[serde(alias = "Regime")]
    Regime,
    /// L2 — where decisions sit.
    #[serde(alias = "Location")]
    Location,
    /// L3 — who's leaning on arrival.
    #[serde(alias = "Trigger")]
    Trigger,
    /// Overlay — event modifiers / sizing constraints, never standalone signals.
    #[serde(alias = "Event")]
    Event,
}

impl Layer {
    pub const ALL: [Layer; 5] = [
        Layer::Tradeability,
        Layer::Regime,
        Layer::Location,
        Layer::Trigger,
        Layer::Event,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Layer::Tradeability => "tradeability",
            Layer::Regime => "regime",
            Layer::Location => "location",
            Layer::Trigger => "trigger",
            Layer::Event => "event",
        }
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Layer {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let low = s.trim().to_ascii_lowercase();
        Layer::ALL
            .into_iter()
            .find(|l| l.as_str() == low)
            .ok_or_else(|| Error::Parse(format!("unknown layer: {s}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_snake_case_with_legacy_alias() {
        // Writes are snake_case...
        assert_eq!(serde_json::to_string(&Layer::Regime).unwrap(), "\"regime\"");
        // ...reads accept both snake_case and legacy PascalCase (persisted genomes).
        assert_eq!(
            serde_json::from_str::<Layer>("\"regime\"").unwrap(),
            Layer::Regime
        );
        assert_eq!(
            serde_json::from_str::<Layer>("\"Regime\"").unwrap(),
            Layer::Regime
        );
    }
}
