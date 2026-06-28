//! `se-core` — shared domain primitives for the swing-X engine.
//!
//! No I/O. Every other crate depends on these types. The most important
//! invariants of the system are encoded here as types:
//!
//! * [`AsOf`] / [`DecisionTs`] make point-in-time (PIT) timestamps explicit so a
//!   feature value can never be silently read "from the future".
//! * [`Layer`] tags every feature with the conditional layer it belongs to
//!   (tradeability / regime / location / trigger / event).
//! * [`Horizon`] + [`HorizonProfile`] keep ALL swing constants (barrier widths,
//!   time barriers, sampling cadence, costs) as *config*, never hardcoded —
//!   which is what makes horizon generalization (P8) possible.

pub mod error;
pub mod feature;
pub mod horizon;
pub mod ids;
pub mod layer;
pub mod regime;
pub mod risk;
pub mod scanner;
pub mod signal;
pub mod strategy;
pub mod ticker;
pub mod time;

pub use error::{Error, Result};
pub use feature::Feature;
pub use horizon::{Cadence, CostModel, Horizon, HorizonProfile};
pub use ids::{LabelId, ModelId, SignalId, StrategyId, TradeId};
pub use layer::Layer;
pub use regime::RegimeLabel;
pub use risk::{RiskModel, StopSpec, TargetSpec};
pub use scanner::Scanner;
pub use signal::{Driver, MonitorAction, Signal, Trade, TradeMode};
pub use strategy::{CmpOp, Genome, Predicate, Strategy, StrategyStatus};
pub use ticker::Ticker;
pub use time::{AsOf, Bar, DecisionTs, LeadTimeTag, Side};
