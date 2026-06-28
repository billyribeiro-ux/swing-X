//! `se-features` — the conditional feature engine (regime × location × trigger),
//! gated by tradeability and modified by events.
//!
//! Every layer implements [`FeatureModule`] and reads only through a
//! [`FeatureContext`] (PIT-safe). Each layer also exposes a `PROPRIETARY_FEATURE`
//! path via the context's proprietary hooks, whose output passes the same gates.
//!
//! v1 implements LAYER 0 (tradeability). Layers 1–3 + the event overlay land in
//! their build phases.

pub mod indicators;
mod layer0_tradeability;
mod module;

pub use layer0_tradeability::{
    TradeabilityComponents, TradeabilityConfig, TradeabilityGate, TradeabilityInput,
    TradeabilityModule, TradeabilityScore,
};
pub use module::{FeatureContext, FeatureModule};
