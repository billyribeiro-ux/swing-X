//! `se-regime` — the regime layer: feature assembly + transparent classifier +
//! window-labeling engine.
//!
//! Three pieces:
//! * [`RegimeClassifier`] — a rule-based, interpretable v1 that maps the Layer-1
//!   regime feature vector to a [`se_core::RegimeLabel`] with a softmax probability
//!   map, a confidence, and an observed-input count. `OutOfDistribution` is
//!   first-class: too few signals (or one wildly out of range) suppresses rather
//!   than guesses.
//! * [`RegimeEngine`] — walks stored bars over a window, runs the PIT-safe
//!   [`se_features::RegimeModule`], persists its features, classifies, and stores
//!   the regime. Provider-independent (reads from the store only).
//! * [`RegimeAssessment`] — the classifier output, persistable as JSON.
//!
//! Documented proxies live in the classifier docs: VIX-as-IV for the vol-risk
//! premium, and gamma-from-vol when the proprietary GEX feed is unavailable.

mod classifier;
mod engine;

pub use classifier::{ClassifierConfig, RegimeAssessment, RegimeClassifier};
pub use engine::{RegimeEngine, StoredRegime};
