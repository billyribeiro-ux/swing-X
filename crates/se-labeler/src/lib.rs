//! `se-labeler` — the Rust triple-barrier labeler (geometry + dataset assembly).
//!
//! Two responsibilities:
//!
//!   * [`triple_barrier::TripleBarrier`] — pure, well-tested triple-barrier geometry over a
//!     [`se_core::HorizonProfile`]. First-touch outcome in {target, stop, time} with
//!     conservative intrabar ordering (both-in-one-bar => stop), realized return in R units.
//!     Mirrors the Python `labeling/triple_barrier.py` semantics exactly.
//!   * [`assemble`] — turns label events + per-entry feature maps into the
//!     [`se_mlclient::DatasetRow`]s the Parquet writer expects, setting `t1` to the barrier
//!     end so CPCV purges correctly (purge length == label horizon via the profile).

pub mod assemble;
pub mod triple_barrier;

pub use assemble::{assemble_dataset, LabeledEntry};
pub use triple_barrier::{LabelError, LabelEvent, Outcome, TripleBarrier};
