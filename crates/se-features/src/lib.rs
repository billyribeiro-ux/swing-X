//! `se-features` — the conditional feature engine (regime × location × trigger),
//! gated by tradeability and modified by events.
//!
//! Every layer implements [`FeatureModule`] and reads only through a
//! [`FeatureContext`] (PIT-safe). Each layer also exposes a `PROPRIETARY_FEATURE`
//! path via the context's proprietary hooks, whose output passes the same gates.
//!
//! v1 implements LAYER 0 (tradeability), LAYER 1 (regime), LAYER 2 (location),
//! LAYER 3 (trigger) and the event overlay.

mod event_overlay;
pub mod indicators;
mod layer0_tradeability;
mod layer1_regime;
mod layer2_location;
mod layer3_trigger;
mod module;

pub use event_overlay::EventOverlay;
pub use layer0_tradeability::{
    TradeabilityComponents, TradeabilityConfig, TradeabilityGate, TradeabilityInput,
    TradeabilityModule, TradeabilityScore,
};
pub use layer1_regime::RegimeModule;
pub use layer2_location::LocationModule;
pub use layer3_trigger::TriggerModule;
pub use module::{FeatureContext, FeatureModule};
