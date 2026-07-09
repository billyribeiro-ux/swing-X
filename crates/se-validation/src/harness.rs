//! The validation harness — the orchestration that turns a labeled dataset into a
//! promotion decision.
//!
//! Flow:
//!   1. take a labeled dataset (rows from `se-labeler` / `se-mlclient`),
//!   2. write it to Parquet (`se-mlclient::write_dataset`),
//!   3. call `POST /validate` on the worker (`se-mlclient::MlClient`),
//!   4. re-evaluate the promotion gate on the returned metrics ([`crate::gate`]).
//!
//! Fail-closed throughout: any I/O, transport, or HTTP error yields a non-promotion (a
//! [`GateDecision`] with `passed == false`) — never a default pass.

use std::path::PathBuf;

use se_core::{Error, HorizonProfile, Result};
use se_mlclient::{
    write_dataset, DatasetRow, FoldSpec, MlClient, MlValidateRequest, ValidationResult,
};

use crate::gate::{GateDecision, PromotionGate};

/// The harness binds an [`MlClient`] to a working directory for dataset Parquet files.
#[derive(Debug, Clone)]
pub struct ValidationHarness {
    client: MlClient,
    work_dir: PathBuf,
}

/// The harness output: the gate decision plus the raw [`ValidationResult`] it was derived
/// from (so callers can log/store the full metric set).
#[derive(Debug, Clone)]
pub struct HarnessOutcome {
    pub decision: GateDecision,
    pub validation: ValidationResult,
    /// The `dataset_uri` handed to the worker (useful for debugging / re-runs).
    pub dataset_uri: String,
}

/// Request-shaping options for [`ValidationHarness::evaluate_with`]. The classic
/// [`ValidationHarness::evaluate`] signature covers only the CPCV shape + `n_trials`; this
/// struct adds the honest search-multiplicity count and the explicit forward-holdout boundary
/// without breaking that signature. `Default` reproduces the legacy behavior exactly
/// (`FoldSpec::default()`-shaped CV, a single trial, no search multiplicity, 70/30 forward split).
#[derive(Debug, Clone)]
pub struct EvaluateOptions {
    /// CPCV shape: number of contiguous time groups (>= 2).
    pub n_groups: u32,
    /// CPCV shape: groups held out as test per combination (>= 1).
    pub k_test_groups: u32,
    /// Per-validation trial-grid size for DSR deflation + PBO (>= 1).
    pub n_trials: u32,
    /// Cumulative count of DISTINCT genomes the search has evaluated so far in this run.
    /// The worker deflates DSR against `max(n_trials, n_search_trials)`; `1` (the default)
    /// means "no search multiplicity beyond the trial grid" — the legacy behavior.
    pub n_search_trials: u32,
    /// Optional RFC3339 boundary for the worker's strict forward holdout: fit on `ts <`
    /// boundary, measure on `ts >=` boundary. `None` (the default) keeps the 70/30 row split.
    pub forward_boundary_ts: Option<String>,
}

impl Default for EvaluateOptions {
    fn default() -> Self {
        let fold = FoldSpec::default();
        EvaluateOptions {
            n_groups: fold.n_groups,
            k_test_groups: fold.k_test_groups,
            n_trials: 1,
            n_search_trials: 1,
            forward_boundary_ts: None,
        }
    }
}

impl ValidationHarness {
    /// Construct a harness with an explicit client and a directory for Parquet datasets.
    pub fn new(client: MlClient, work_dir: impl Into<PathBuf>) -> Self {
        ValidationHarness {
            client,
            work_dir: work_dir.into(),
        }
    }

    /// Construct from `ML_WORKER_URL` (or the default), writing datasets under the system
    /// temp directory.
    pub fn from_env() -> Result<Self> {
        let client = MlClient::from_env().map_err(Error::from)?;
        Ok(ValidationHarness::new(client, std::env::temp_dir()))
    }

    /// Run the full pipeline for `rows`, using `profile` to derive the fold/purge geometry.
    ///
    /// `dataset_name` names the Parquet file under the work dir. `n_groups`/`k_test_groups`
    /// are the CPCV shape; `n_trials` sizes the search space for DSR deflation + PBO.
    ///
    /// The classic entry point, kept signature-stable: delegates to [`Self::evaluate_with`]
    /// with the legacy defaults (`n_search_trials = 1`, default 70/30 forward split).
    ///
    /// On ANY error (write/transport/HTTP) returns the error; the caller treats that as a
    /// non-promotion. For an explicit fail-closed *decision* (rather than `Err`), use
    /// [`Self::evaluate_or_fail_closed`].
    pub async fn evaluate(
        &self,
        rows: &[DatasetRow],
        profile: &HorizonProfile,
        dataset_name: &str,
        n_groups: u32,
        k_test_groups: u32,
        n_trials: u32,
    ) -> Result<HarnessOutcome> {
        self.evaluate_with(
            rows,
            profile,
            dataset_name,
            &EvaluateOptions {
                n_groups,
                k_test_groups,
                n_trials,
                ..EvaluateOptions::default()
            },
        )
        .await
    }

    /// Like [`Self::evaluate`] but with the full option set: the honest cumulative
    /// `n_search_trials` count (so the worker's DSR deflation reflects the search's real
    /// multiple-comparisons burden) and an optional explicit `forward_boundary_ts` for the
    /// worker's strict forward holdout (fit strictly before the boundary, measure at/after it).
    ///
    /// Same fail-closed semantics as [`Self::evaluate`]: any write/transport/HTTP error is
    /// returned as `Err`, never a defaulted pass.
    pub async fn evaluate_with(
        &self,
        rows: &[DatasetRow],
        profile: &HorizonProfile,
        dataset_name: &str,
        opts: &EvaluateOptions,
    ) -> Result<HarnessOutcome> {
        let path = self.work_dir.join(dataset_name);
        let written = write_dataset(rows, &path).map_err(Error::from)?;
        let dataset_uri = written.to_string_lossy().into_owned();

        let fold_spec = FoldSpec::from_profile(profile, opts.n_groups, opts.k_test_groups);
        let req = MlValidateRequest {
            dataset_uri: dataset_uri.clone(),
            horizon: profile.horizon.as_str().to_string(),
            fold_spec,
            n_trials: opts.n_trials,
            n_search_trials: opts.n_search_trials.max(1),
            forward_boundary_ts: opts.forward_boundary_ts.clone(),
        };

        let validation = self.client.validate(req).await.map_err(Error::from)?;
        let decision = PromotionGate::evaluate(&validation);

        Ok(HarnessOutcome {
            decision,
            validation,
            dataset_uri,
        })
    }

    /// Like [`Self::evaluate`] but converts any error into a fail-closed [`GateDecision`]
    /// (`passed == false`) instead of returning `Err`. Use where the call site must always
    /// produce a decision and a failed validation must never be a silent pass.
    pub async fn evaluate_or_fail_closed(
        &self,
        rows: &[DatasetRow],
        profile: &HorizonProfile,
        dataset_name: &str,
        n_groups: u32,
        k_test_groups: u32,
        n_trials: u32,
    ) -> GateDecision {
        match self
            .evaluate(
                rows,
                profile,
                dataset_name,
                n_groups,
                k_test_groups,
                n_trials,
            )
            .await
        {
            Ok(outcome) => outcome.decision,
            Err(e) => GateDecision::fail_closed(format!("validation failed: {e}")),
        }
    }

    /// The underlying client (e.g. for a health probe before a run).
    pub fn client(&self) -> &MlClient {
        &self.client
    }
}
