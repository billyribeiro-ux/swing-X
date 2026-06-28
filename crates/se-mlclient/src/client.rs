//! The HTTP client for the Python `se_ml` sidecar.
//!
//! [`MlClient`] is the single chokepoint through which Rust talks to Python. It is
//! **fail-closed**: every transport error, non-2xx status, timeout, or decode failure
//! surfaces as [`MlError`] (never a defaulted "pass"). Validation callers MUST treat any
//! error as a non-promotion.

use std::time::Duration;

use crate::contract::{
    CalibrationResult, FitResult, HealthResponse, ImportanceResult, MlCalibrateRequest,
    MlFitRequest, MlImportanceRequest, MlValidateRequest, ValidationResult,
};
use crate::error::MlError;

/// Default base URL when `ML_WORKER_URL` is unset.
pub const DEFAULT_BASE_URL: &str = "http://localhost:8088";

/// Default per-request timeout. Validation can be heavy (CPCV over a trial grid), so this
/// is generous; a hang still fails closed rather than blocking a promotion forever.
const DEFAULT_TIMEOUT_SECS: u64 = 300;

/// Async client for the ML worker. Cheap to clone (the inner `reqwest::Client` is `Arc`-backed).
#[derive(Debug, Clone)]
pub struct MlClient {
    base_url: String,
    http: reqwest::Client,
}

impl MlClient {
    /// Construct a client for an explicit base URL with the default timeout.
    pub fn new(base_url: impl Into<String>) -> Result<Self, MlError> {
        Self::with_timeout(base_url, Duration::from_secs(DEFAULT_TIMEOUT_SECS))
    }

    /// Construct a client for an explicit base URL and request timeout.
    pub fn with_timeout(base_url: impl Into<String>, timeout: Duration) -> Result<Self, MlError> {
        let base_url = base_url.into();
        let base_url = base_url.trim_end_matches('/').to_string();
        if base_url.is_empty() {
            return Err(MlError::Config("empty ML worker base URL".into()));
        }
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| MlError::Config(format!("failed to build http client: {e}")))?;
        Ok(MlClient { base_url, http })
    }

    /// Build from the environment: `ML_WORKER_URL` or [`DEFAULT_BASE_URL`].
    pub fn from_env() -> Result<Self, MlError> {
        let base_url =
            std::env::var("ML_WORKER_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        Self::new(base_url)
    }

    /// The configured base URL (no trailing slash).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base_url, path.trim_start_matches('/'))
    }

    /// `GET /health`.
    pub async fn health(&self) -> Result<HealthResponse, MlError> {
        let endpoint = "/health";
        let resp = self
            .http
            .get(self.url(endpoint))
            .send()
            .await
            .map_err(|source| MlError::Transport {
                endpoint: endpoint.into(),
                source,
            })?;
        Self::decode(endpoint, resp).await
    }

    /// `POST /fit`.
    pub async fn fit(&self, req: MlFitRequest) -> Result<FitResult, MlError> {
        self.post_json("/fit", &req).await
    }

    /// `POST /validate`. The single source of OOS validation metrics for the gate.
    pub async fn validate(&self, req: MlValidateRequest) -> Result<ValidationResult, MlError> {
        self.post_json("/validate", &req).await
    }

    /// `POST /calibrate`.
    pub async fn calibrate(&self, req: MlCalibrateRequest) -> Result<CalibrationResult, MlError> {
        self.post_json("/calibrate", &req).await
    }

    /// `POST /importance`.
    pub async fn importance(&self, req: MlImportanceRequest) -> Result<ImportanceResult, MlError> {
        self.post_json("/importance", &req).await
    }

    /// POST a JSON body and decode the JSON response into `R`, mapping every failure mode
    /// to [`MlError`] (fail-closed).
    async fn post_json<B, R>(&self, endpoint: &str, body: &B) -> Result<R, MlError>
    where
        B: serde::Serialize,
        R: serde::de::DeserializeOwned,
    {
        let resp = self
            .http
            .post(self.url(endpoint))
            .json(body)
            .send()
            .await
            .map_err(|source| MlError::Transport {
                endpoint: endpoint.into(),
                source,
            })?;
        Self::decode(endpoint, resp).await
    }

    /// Turn a response into a decoded `R`, or an [`MlError`] for any non-2xx / decode issue.
    async fn decode<R>(endpoint: &str, resp: reqwest::Response) -> Result<R, MlError>
    where
        R: serde::de::DeserializeOwned,
    {
        let status = resp.status();
        if !status.is_success() {
            // Read the body for diagnostics, but never let a failed read mask the HTTP error.
            let body = resp.text().await.unwrap_or_default();
            let body = Self::truncate(&body, 512);
            return Err(MlError::Http {
                endpoint: endpoint.into(),
                status: status.as_u16(),
                body,
            });
        }
        resp.json::<R>().await.map_err(|source| MlError::Decode {
            endpoint: endpoint.into(),
            source,
        })
    }

    fn truncate(s: &str, max: usize) -> String {
        if s.len() <= max {
            s.to_string()
        } else {
            let mut end = max;
            while !s.is_char_boundary(end) {
                end -= 1;
            }
            format!("{}…", &s[..end])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_trailing_slash_and_builds_url() {
        let c = MlClient::new("http://example.com:8088/").unwrap();
        assert_eq!(c.base_url(), "http://example.com:8088");
        assert_eq!(c.url("/validate"), "http://example.com:8088/validate");
        assert_eq!(c.url("health"), "http://example.com:8088/health");
    }

    #[test]
    fn empty_base_url_is_config_error() {
        assert!(matches!(MlClient::new(""), Err(MlError::Config(_))));
    }

    #[test]
    fn from_env_default_when_unset() {
        // SAFETY: single-threaded test; we restore nothing because default is the unset case.
        std::env::remove_var("ML_WORKER_URL");
        let c = MlClient::from_env().unwrap();
        assert_eq!(c.base_url(), DEFAULT_BASE_URL);
    }
}
