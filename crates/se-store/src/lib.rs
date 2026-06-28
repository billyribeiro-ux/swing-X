//! `se-store` — the point-in-time (PIT) feature store.
//!
//! Two responsibilities, one invariant:
//! 1. Persist bars / features / regimes / labels / strategies / signals / etc.
//! 2. Read features *as of a decision instant* such that a value is returned only
//!    when its `as_of <= decision_ts`.
//!
//! The invariant is enforced by [`PitContext`]: it is the ONLY way to read features
//! for decision-making, and every query it issues hard-codes the `as_of <= $decision`
//! predicate. Callers cannot pass their own timestamp filter, so leakage cannot be
//! introduced by a careless query elsewhere in the system.

mod models;
mod pit;
mod store;

pub use models::{BarRow, FeatureWrite, MacroWrite};
pub use pit::PitContext;
pub use store::Store;

/// Re-export sqlx for downstream crates that need the pool type.
pub use sqlx;
