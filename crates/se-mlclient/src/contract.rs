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

/// Default acting threshold τ\* (0.5) for `ValidationResult::act_threshold` when a worker
/// response omits it (e.g. an older worker or a hand-built test fixture).
fn default_act_threshold() -> f64 {
    0.5
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
    /// OOS precision: fraction of OOS trades ACTED ON (prob >= τ\*) that were profitable. The
    /// north-star meta-labeling metric. Surfaced and persisted; never a ranking key here (the
    /// cost-aware OOS expectancy is already precision-conditioned upstream).
    #[serde(default)]
    pub precision_oos: f64,
    /// OOS recall at τ\*: fraction of profitable OOS trades that were acted on.
    #[serde(default)]
    pub recall_oos: f64,
    /// τ\* — the acting threshold in [0,1] the meta-label classifier acts at. Defaults to 0.5
    /// when absent so older worker responses / fixtures still deserialize.
    #[serde(default = "default_act_threshold")]
    pub act_threshold: f64,
    /// Count of OOS trades acted on at τ\* (the acted cohort size). Drives the search guardrail
    /// that refuses to promote genomes that act on too few OOS trades.
    #[serde(default)]
    pub n_acted_oos: i64,
    /// Precision on a STRICT time-ordered forward holdout: fit + threshold on the earliest 70% of
    /// rows, measure precision on the latest 30% (never shuffled). Distinguishes a durable edge
    /// from regime-fitting that shuffled CPCV folds can flatter. Reported, never a ranking key.
    #[serde(default)]
    pub precision_forward: f64,
    /// Cost-aware expectancy (R) on the same forward holdout at the forward acting threshold.
    #[serde(default)]
    pub expectancy_forward: f64,
    /// Count of acted trades in the forward holdout (small = a weak, low-confidence estimate).
    #[serde(default)]
    pub n_forward: i64,
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
