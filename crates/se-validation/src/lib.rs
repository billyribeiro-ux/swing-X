//! `se-validation` — the hard promotion gate and validation orchestration.
//!
//!   * [`gate::PromotionGate`] — the authoritative Rust-side re-check. Promotes ONLY when
//!     `dsr > 0 && pbo < 0.5 && oos_expectancy_cost_aware > 0 && n_regimes_positive >= 2`.
//!     `win_rate` is never a selection input. Fail-closed: a `None`/errored validation is
//!     not passed.
//!   * [`harness::ValidationHarness`] — takes a labeled dataset, writes Parquet via
//!     `se-mlclient`, calls `POST /validate`, and re-evaluates the gate, returning the
//!     decision plus the raw [`se_mlclient::ValidationResult`].
//!
//! The P4 checkpoint integration test (`tests/p4_checkpoint.rs`) proves leakage is caught
//! end-to-end against the live worker: a leaky (look-ahead) dataset is REJECTED while a
//! genuine-edge dataset PASSES.

pub mod fixtures;
pub mod gate;
pub mod harness;

pub use gate::{
    GateDecision, PromotionGate, DSR_MIN, MIN_POSITIVE_REGIMES, OOS_EXPECTANCY_MIN, PBO_MAX,
};
pub use harness::{HarnessOutcome, ValidationHarness};
