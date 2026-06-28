//! The unit a feature module emits. Carries its PIT provenance so the store can
//! persist `{as_of, source, lead_time, layer}` alongside the value.

use serde::{Deserialize, Serialize};

use crate::layer::Layer;
use crate::time::{AsOf, LeadTimeTag};

/// A single computed feature value with full provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Feature {
    /// Stable identifier, e.g. `regime.gex_sign`, `location.anchored_vwap_dist`.
    pub key: String,
    pub value: f64,
    pub layer: Layer,
    /// When this value became knowable — the spine of leakage prevention.
    pub as_of: AsOf,
    pub lead_time: LeadTimeTag,
    /// Where it came from, e.g. `fmp`, `fred`, `derived`, `proprietary:gex`.
    pub source: String,
}

impl Feature {
    pub fn new(
        key: impl Into<String>,
        value: f64,
        layer: Layer,
        as_of: AsOf,
        lead_time: LeadTimeTag,
        source: impl Into<String>,
    ) -> Self {
        Feature {
            key: key.into(),
            value,
            layer,
            as_of,
            lead_time,
            source: source.into(),
        }
    }

    pub fn is_finite(&self) -> bool {
        self.value.is_finite()
    }
}
