//! `se-mlclient` — the clean Rust<->Python ML job/result boundary.
//!
//! This crate is the **only** place in the workspace that knows the Python `se_ml`
//! sidecar exists. It owns three things:
//!
//!   * [`contract`] — request/response structs mirroring `se_ml.contract` EXACTLY
//!     (snake_case field names), so wire compatibility is enforced by the type system.
//!   * [`client::MlClient`] — an async, **fail-closed** HTTP client. Any transport error,
//!     non-2xx status, timeout, or decode failure becomes an [`MlError`]; there is no
//!     code path by which a failed validation could be read as a pass.
//!   * [`dataset`] — writes a labeled feature matrix to Parquet matching the worker's
//!     on-disk schema (`ts`, `t1`, `label`, optional `regime`, `layer__feature` columns),
//!     with stable feature-column ordering.
//!
//! Bulk data crosses the boundary as a Parquet file referenced by `dataset_uri`; the JSON
//! bodies carry only metadata + metrics.

pub mod client;
pub mod contract;
pub mod dataset;
pub mod error;

pub use client::{MlClient, DEFAULT_BASE_URL};
pub use contract::{
    CalibrationMap, CalibrationResult, FitResult, FoldSpec, HealthResponse, ImportanceResult,
    ImportanceScore, InSampleMetrics, MlCalibrateRequest, MlFitRequest, MlImportanceRequest,
    MlValidateRequest, ReliabilityPoint, ValidationResult,
};
pub use dataset::{path_to_uri, write_dataset, DatasetRow, LABEL_COL, REGIME_COL, T1_COL, TS_COL};
pub use error::MlError;
