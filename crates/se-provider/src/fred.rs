//! FRED adapter (free) — credit spreads + net-liquidity series FMP doesn't supply.
//!
//! Endpoint (verified shape, stable public API):
//! `GET https://api.stlouisfed.org/fred/series/observations
//!      ?series_id=&api_key=&file_type=json&observation_start=&observation_end=`
//! -> `{"observations":[{"date":"YYYY-MM-DD","value":"...", "realtime_start":..}]}`
//! where `value` is a string and missing observations are the literal `"."`.
//!
//! Series mapping:
//! * HY OAS           = `BAMLH0A0HYM2`  (daily,  lag 1d)
//! * IG OAS           = `BAMLC0A0CM`    (daily,  lag 1d)
//! * Fed balance sheet= `WALCL`         (weekly, lag 7d)
//! * TGA              = `WTREGEN`       (weekly, lag 7d)
//! * Reverse repo     = `RRPONTSYD`     (daily,  lag 1d)
//!
//! FRED publishes with a lag, so each point's `as_of = ts + lead_time`.
//! Without `FRED_API_KEY` the provider is still constructible but every method
//! returns [`Error::FeatureUnavailable`] (never errors the whole process).

use async_trait::async_trait;
use chrono::{Duration, NaiveDate};
use se_core::{Error, LeadTimeTag, Result};
use serde::Deserialize;
use tracing::warn;

use crate::http_util::{parse_ymd, session_close_ts};
use crate::provider::DataProvider;
use crate::types::{Capabilities, MacroPoint, MacroSeries, ProviderKind};

const FRED_BASE: &str = "https://api.stlouisfed.org/fred/series/observations";

#[derive(Debug, Clone, Default)]
pub struct FredProvider {
    pub(crate) client: reqwest::Client,
    pub(crate) api_key: Option<String>,
}

impl FredProvider {
    pub fn new(api_key: Option<String>) -> Self {
        FredProvider {
            client: reqwest::Client::new(),
            api_key,
        }
    }

    pub fn from_env() -> Self {
        FredProvider::new(std::env::var("FRED_API_KEY").ok().filter(|s| !s.is_empty()))
    }
}

/// FRED series id + publication-lag (calendar days) for each series FRED serves.
fn fred_spec(series: MacroSeries) -> Option<(&'static str, i64)> {
    match series {
        MacroSeries::HyOas => Some(("BAMLH0A0HYM2", 1)),
        MacroSeries::IgOas => Some(("BAMLC0A0CM", 1)),
        MacroSeries::FedBalanceSheet => Some(("WALCL", 7)),
        MacroSeries::Tga => Some(("WTREGEN", 7)),
        MacroSeries::ReverseRepo => Some(("RRPONTSYD", 1)),
        _ => None,
    }
}

#[derive(Debug, Deserialize)]
struct FredResponse {
    #[serde(default)]
    observations: Vec<FredObservation>,
}

#[derive(Debug, Deserialize)]
struct FredObservation {
    date: String,
    /// String value; missing observations come back as `"."`.
    value: String,
}

impl FredProvider {
    /// Map a series to its FRED points over the range. Missing values (`"."`) are
    /// skipped. Returns `FeatureUnavailable` if the series isn't FRED's or no key.
    async fn fetch_series(
        &self,
        series: MacroSeries,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<MacroPoint>> {
        let Some(api_key) = self.api_key.as_deref() else {
            return Err(Error::FeatureUnavailable(format!(
                "fred `{}` unavailable: FRED_API_KEY not set",
                series.as_str()
            )));
        };
        let Some((series_id, lag_days)) = fred_spec(series) else {
            return Err(Error::FeatureUnavailable(format!(
                "fred does not serve `{}`",
                series.as_str()
            )));
        };

        let url = format!(
            "{FRED_BASE}?series_id={series_id}&api_key={api_key}\
             &file_type=json&observation_start={start}&observation_end={end}"
        );
        let resp = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| Error::Provider(format!("fred GET {series_id} failed: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(Error::Provider(format!(
                "fred GET {series_id} returned HTTP {status}"
            )));
        }
        let parsed: FredResponse = resp
            .json()
            .await
            .map_err(|e| Error::Provider(format!("fred {series_id} decode failed: {e}")))?;

        let lead_time = LeadTimeTag::LaggedDays(lag_days);
        let mut out = Vec::with_capacity(parsed.observations.len());
        for obs in &parsed.observations {
            // FRED uses "." for missing; skip rather than fabricate.
            if obs.value.trim() == "." {
                continue;
            }
            let Some(date) = parse_ymd(&obs.date) else {
                continue;
            };
            let Ok(value) = obs.value.trim().parse::<f64>() else {
                continue;
            };
            let ts = session_close_ts(date);
            let as_of = ts + Duration::days(lag_days);
            out.push(MacroPoint {
                series,
                ts,
                value,
                as_of,
                lead_time,
                source: "fred".into(),
            });
        }
        out.sort_by_key(|p| p.ts);
        Ok(out)
    }
}

#[async_trait]
impl DataProvider for FredProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Fred
    }

    fn capabilities(&self) -> Capabilities {
        if self.api_key.is_some() {
            Capabilities {
                macro_series: true,
                ..Capabilities::NONE
            }
        } else {
            Capabilities::NONE
        }
    }

    async fn macro_series(
        &self,
        series: MacroSeries,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<MacroPoint>> {
        if self.api_key.is_none() {
            warn!(
                series = series.as_str(),
                "fred: FRED_API_KEY not set; macro_series unavailable"
            );
        }
        self.fetch_series(series, start, end).await
    }

    // daily_bars / quotes fall through to the trait defaults, which return
    // FeatureUnavailable — FRED serves neither.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_observations_skips_missing() {
        // Shape per FRED docs; "." marks a missing observation.
        let sample = r#"{
            "realtime_start":"2026-06-27","realtime_end":"2026-06-27",
            "observation_start":"2026-06-20","observation_end":"2026-06-26",
            "units":"lin","output_type":1,"file_type":"json",
            "order_by":"observation_date","sort_order":"asc","count":3,
            "observations":[
                {"realtime_start":"2026-06-27","realtime_end":"2026-06-27","date":"2026-06-22","value":"3.05"},
                {"realtime_start":"2026-06-27","realtime_end":"2026-06-27","date":"2026-06-23","value":"."},
                {"realtime_start":"2026-06-27","realtime_end":"2026-06-27","date":"2026-06-24","value":"3.11"}
            ]
        }"#;
        let parsed: FredResponse = serde_json::from_str(sample).unwrap();
        assert_eq!(parsed.observations.len(), 3);

        // Replicate the mapping logic the fetch path uses.
        let lag = 1i64;
        let mut pts: Vec<MacroPoint> = Vec::new();
        for obs in &parsed.observations {
            if obs.value.trim() == "." {
                continue;
            }
            let date = parse_ymd(&obs.date).unwrap();
            let value: f64 = obs.value.trim().parse().unwrap();
            let ts = session_close_ts(date);
            pts.push(MacroPoint {
                series: MacroSeries::HyOas,
                ts,
                value,
                as_of: ts + Duration::days(lag),
                lead_time: LeadTimeTag::LaggedDays(lag),
                source: "fred".into(),
            });
        }
        assert_eq!(pts.len(), 2, "missing '.' observation must be skipped");
        assert_eq!(pts[0].value, 3.05);
        assert_eq!(pts[1].value, 3.11);
        // as_of is one day after the reference ts (publication lag).
        assert_eq!(pts[0].as_of, pts[0].ts + Duration::days(1));
        assert_eq!(pts[0].lead_time, LeadTimeTag::LaggedDays(1));
        assert_eq!(pts[0].source, "fred");
    }

    #[test]
    fn specs_cover_all_fred_series_only() {
        for s in [
            MacroSeries::HyOas,
            MacroSeries::IgOas,
            MacroSeries::FedBalanceSheet,
            MacroSeries::Tga,
            MacroSeries::ReverseRepo,
        ] {
            assert!(fred_spec(s).is_some(), "{s:?} should be a FRED series");
        }
        for s in [MacroSeries::Vix, MacroSeries::Ust2y, MacroSeries::Gold] {
            assert!(fred_spec(s).is_none(), "{s:?} is not FRED's");
        }
        // Weekly series carry the 7-day lag, daily series the 1-day lag.
        assert_eq!(fred_spec(MacroSeries::FedBalanceSheet).unwrap().1, 7);
        assert_eq!(fred_spec(MacroSeries::Tga).unwrap().1, 7);
        assert_eq!(fred_spec(MacroSeries::HyOas).unwrap().1, 1);
        assert_eq!(fred_spec(MacroSeries::ReverseRepo).unwrap().1, 1);
    }

    #[tokio::test]
    async fn no_key_returns_unavailable_and_none_caps() {
        let p = FredProvider::new(None);
        assert_eq!(p.capabilities(), Capabilities::NONE);
        let start = NaiveDate::from_ymd_opt(2026, 6, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2026, 6, 26).unwrap();
        let err = p
            .macro_series(MacroSeries::HyOas, start, end)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::FeatureUnavailable(_)));
    }

    #[tokio::test]
    async fn with_key_caps_macro_only() {
        let p = FredProvider::new(Some("dummy".into()));
        let caps = p.capabilities();
        assert!(caps.macro_series);
        assert!(!caps.daily_bars);
        assert!(!caps.quotes);
        assert!(!caps.etf_profile);
        // Non-FRED series rejected even with a key (without any network call).
        let start = NaiveDate::from_ymd_opt(2026, 6, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2026, 6, 26).unwrap();
        let err = p
            .fetch_series(MacroSeries::Vix, start, end)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::FeatureUnavailable(_)));
    }
}
