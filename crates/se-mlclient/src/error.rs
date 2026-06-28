//! The client's local error type, fail-closed by construction.
//!
//! EVERY transport failure, HTTP non-2xx, timeout, or decode error becomes an
//! [`MlError`]. There is no variant that a caller could mistake for "validation passed".
//! Callers convert into the shared [`se_core::Error`] at their boundary; a failed
//! validation is therefore *always* a non-promotion (never a silent default-pass).

use thiserror::Error;

/// Errors raised by the ML client and dataset writer.
#[derive(Debug, Error)]
pub enum MlError {
    /// Transport-level failure (connection refused, DNS, timeout, TLS, ...).
    #[error("ml transport error calling {endpoint}: {source}")]
    Transport {
        endpoint: String,
        #[source]
        source: reqwest::Error,
    },

    /// The worker responded with a non-2xx status. `body` is truncated diagnostic text.
    #[error("ml worker returned HTTP {status} for {endpoint}: {body}")]
    Http {
        endpoint: String,
        status: u16,
        body: String,
    },

    /// The response body could not be decoded into the expected contract type.
    #[error("ml response decode error for {endpoint}: {source}")]
    Decode {
        endpoint: String,
        #[source]
        source: reqwest::Error,
    },

    /// Configuration problem (e.g. a malformed `ML_WORKER_URL`).
    #[error("ml client config error: {0}")]
    Config(String),

    /// Error building or writing the dataset Parquet file.
    #[error("dataset error: {0}")]
    Dataset(String),
}

impl From<MlError> for se_core::Error {
    fn from(e: MlError) -> Self {
        // Fail-closed: any ML error maps to a validation error, which callers must treat
        // as NOT promotable. There is deliberately no "ok-ish" conversion.
        se_core::Error::Validation(e.to_string())
    }
}
