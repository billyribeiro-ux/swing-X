//! Request/response structs mirroring the Python `se_ml.contract` models EXACTLY.
//!
//! Field names are snake_case and identical to the pydantic models in
//! `ml-worker/src/se_ml/contract.py`. Bulk data (features + labels) is never embedded
//! here; it crosses the boundary as a Parquet file referenced by `dataset_uri`. These
//! bodies carry only metadata + metrics.
//!
//! `se-mlclient` is the ONLY crate in the workspace that knows the Python sidecar's
//! wire format. Everything downstream consumes the typed results, never raw JSON.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// `GET /health` -> `{status, version}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

// --------------------------------------------------------------------------- //
// /fit
// --------------------------------------------------------------------------- //

/// `POST /fit` request body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MlFitRequest {
    /// Parquet path or `file://` URI for the labeled feature matrix.
    pub dataset_uri: String,
    /// Horizon profile id, e.g. `"swing"` (use [`se_core::Horizon::as_str`]).
    pub horizon: String,
    /// Free-form LightGBM hyperparameters; empty means the worker's defaults.
    #[serde(default)]
    pub model_params: serde_json::Map<String, serde_json::Value>,
    /// Deterministic seed.
    pub seed: i64,
}

/// In-sample edge metrics returned by `/fit` (on the realized R labels).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InSampleMetrics {
    pub expectancy: f64,
    pub profit_factor: f64,
    pub sharpe: f64,
    pub cvar5: f64,
    pub mar: f64,
    pub n: i64,
}

/// `POST /fit` response body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitResult {
    pub model_id: String,
    pub artifact_uri: String,
    pub in_sample_metrics: InSampleMetrics,
}

// --------------------------------------------------------------------------- //
// /validate
// --------------------------------------------------------------------------- //

/// CPCV fold specification. Mirrors `se_ml.contract.FoldSpec`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoldSpec {
    /// Number of contiguous time groups (>= 2).
    pub n_groups: u32,
    /// Groups held out as test per CPCV combination (>= 1).
    pub k_test_groups: u32,
    /// Bars embargoed AFTER each test block.
    pub embargo_bars: u32,
    /// Purge train labels overlapping any test span.
    pub purge: bool,
}

impl FoldSpec {
    /// Construct a [`FoldSpec`] whose `embargo_bars` and (via the horizon) purge length
    /// are derived from a [`se_core::HorizonProfile`], so labeling and cross-validation
    /// cannot desync. `purge` is always on; `n_groups`/`k_test_groups` are CV-shape choices.
    pub fn from_profile(
        profile: &se_core::HorizonProfile,
        n_groups: u32,
        k_test_groups: u32,
    ) -> Self {
        FoldSpec {
            n_groups,
            k_test_groups,
            embargo_bars: profile.embargo_bars,
            purge: true,
        }
    }
}

impl Default for FoldSpec {
    /// Matches the Python `FoldSpec()` defaults: 8 groups, 2 test, 5 embargo, purge on.
    fn default() -> Self {
        FoldSpec {
            n_groups: 8,
            k_test_groups: 2,
            embargo_bars: 5,
            purge: true,
        }
    }
}

/// `POST /validate` request body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MlValidateRequest {
    pub dataset_uri: String,
    pub horizon: String,
    #[serde(default)]
    pub fold_spec: FoldSpec,
    /// Number of strategy trials for DSR deflation / PBO (>= 1).
    pub n_trials: u32,
}

/// `POST /calibrate` request body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlCalibrateRequest {
    pub dataset_uri: String,
    pub model_id: String,
}

/// `POST /importance` request body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlImportanceRequest {
    pub dataset_uri: String,
    pub model_id: String,
}

/// `POST /validate` response body. This is the authoritative input to the promotion gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationResult {
    pub dsr: f64,
    pub pbo: f64,
    pub oos_expectancy_cost_aware: f64,
    pub profit_factor: f64,
    pub cvar5: f64,
    pub mar: f64,
    pub regime_contrib: BTreeMap<String, f64>,
    pub n_regimes_positive: i64,
    pub passed_gate: bool,
}

// --------------------------------------------------------------------------- //
// /calibrate
// --------------------------------------------------------------------------- //

/// A single reliability-curve point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReliabilityPoint {
    pub predicted: f64,
    pub realized: f64,
    pub count: i64,
}

/// The fitted calibration map (e.g. isotonic).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationMap {
    pub method: String,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

/// `POST /calibrate` response body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationResult {
    pub calibration_map: CalibrationMap,
    pub reliability_points: Vec<ReliabilityPoint>,
    pub brier: f64,
}

// --------------------------------------------------------------------------- //
// /importance
// --------------------------------------------------------------------------- //

/// SHAP + permutation importance for a single feature or layer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ImportanceScore {
    pub shap: f64,
    pub permutation: f64,
}

/// `POST /importance` response body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportanceResult {
    pub per_feature: BTreeMap<String, ImportanceScore>,
    pub per_layer: BTreeMap<String, ImportanceScore>,
}
