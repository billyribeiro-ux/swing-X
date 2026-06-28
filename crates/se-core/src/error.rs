//! Shared error type. Crates may define richer local errors and convert into
//! this at boundaries; many carry a human string for logging/attribution.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("unknown ticker: {0}")]
    UnknownTicker(String),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("provider error: {0}")]
    Provider(String),

    #[error("store error: {0}")]
    Store(String),

    #[error("validation error: {0}")]
    Validation(String),

    /// A feature could not be produced — e.g. a `PROPRIETARY_FEATURE` hook that
    /// is not configured, or a stale/missing upstream feed. NEVER fabricate.
    #[error("feature `{0}` unavailable (proprietary, stale, or degraded)")]
    FeatureUnavailable(String),

    /// The current context is outside any regime cohort the model has seen.
    #[error("out-of-distribution: {0}")]
    OutOfDistribution(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("{0}")]
    Message(String),
}

impl Error {
    pub fn msg(s: impl Into<String>) -> Self {
        Error::Message(s.into())
    }
}
