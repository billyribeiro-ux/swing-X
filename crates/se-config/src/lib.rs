//! `se-config` — runtime configuration.
//!
//! Loads a `.env` file (if present) then reads environment variables into an
//! [`AppConfig`]. The active [`HorizonProfile`] is selected by `SE_HORIZON`, which
//! makes the entire pipeline re-runnable per horizon (the P8 generalization axis).

use std::path::Path;

use se_core::{Error, Horizon, HorizonProfile, Result, RiskModel, StopSpec, TargetSpec, Ticker};

/// Fully-resolved application configuration.
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub database_url: String,
    pub ml_worker_url: String,
    pub api_bind: String,
    pub horizon: HorizonProfile,
    pub universe: Vec<Ticker>,
    pub fmp_configured: bool,
    pub fred_configured: bool,
    /// Operator's ground-rule risk geometry (stop/target). From `SE_STOP`/`SE_TARGET1`/
    /// `SE_TARGET2`, falling back to the active horizon's profile geometry.
    pub risk: RiskModel,
    /// Whether the search locks risk to `risk` (ground rules fixed; conditions optimized) or
    /// explores it. From `SE_LOCK_RISK`.
    pub lock_risk: bool,
}

impl AppConfig {
    /// Load `.env` (best-effort) then read env vars with sane defaults.
    pub fn from_env() -> Result<Self> {
        load_dotenv(Path::new(".env"));

        let database_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://swing:swing@localhost:5433/swing".into());
        let ml_worker_url =
            std::env::var("ML_WORKER_URL").unwrap_or_else(|_| "http://localhost:8088".into());
        let api_bind = std::env::var("SE_API_BIND").unwrap_or_else(|_| "0.0.0.0:8080".into());

        let horizon = std::env::var("SE_HORIZON")
            .ok()
            .map(|h| h.parse::<Horizon>())
            .transpose()
            .map_err(|e| Error::Config(e.to_string()))?
            .unwrap_or(Horizon::Swing);

        let fmp_configured = std::env::var("FMP_API_KEY")
            .map(|k| !k.is_empty() && k != "__set_me__")
            .unwrap_or(false);
        let fred_configured = std::env::var("FRED_API_KEY")
            .map(|k| !k.is_empty())
            .unwrap_or(false);

        let profile = HorizonProfile::for_horizon(horizon);
        let risk = resolve_risk(&profile)?;
        let lock_risk = std::env::var("SE_LOCK_RISK")
            .ok()
            .map(|v| parse_bool(&v))
            .unwrap_or(false);

        Ok(AppConfig {
            database_url,
            ml_worker_url,
            api_bind,
            horizon: profile,
            universe: Ticker::ALL.to_vec(),
            fmp_configured,
            fred_configured,
            risk,
            lock_risk,
        })
    }
}

/// Resolve the operator's ground-rule [`RiskModel`] from `SE_STOP`/`SE_TARGET1`/`SE_TARGET2`,
/// each falling back to the horizon profile's default geometry when unset.
fn resolve_risk(profile: &HorizonProfile) -> Result<RiskModel> {
    let default = RiskModel::from_profile(profile);
    let stop: StopSpec = match std::env::var("SE_STOP") {
        Ok(s) if !s.trim().is_empty() => {
            s.parse().map_err(|e: Error| Error::Config(e.to_string()))?
        }
        _ => default.stop,
    };
    let target1: TargetSpec = match std::env::var("SE_TARGET1") {
        Ok(s) if !s.trim().is_empty() => {
            s.parse().map_err(|e: Error| Error::Config(e.to_string()))?
        }
        _ => default.target1,
    };
    // SE_TARGET2 unset => keep the profile default's second target. Empty/"none" => drop it.
    let target2: Option<TargetSpec> = match std::env::var("SE_TARGET2") {
        Ok(s) if s.trim().eq_ignore_ascii_case("none") => None,
        Ok(s) if !s.trim().is_empty() => {
            Some(s.parse().map_err(|e: Error| Error::Config(e.to_string()))?)
        }
        _ => default.target2,
    };
    Ok(RiskModel::new(stop, target1, target2))
}

fn parse_bool(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Minimal `.env` loader: `KEY=VALUE` lines, `#` comments, blanks ignored.
/// Does not overwrite variables already present in the environment.
pub fn load_dotenv(path: &Path) {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return;
    };
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, val)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let val = val.trim().trim_matches('"');
        if std::env::var_os(key).is_none() {
            // SAFETY: single-threaded config load at process startup.
            unsafe { std::env::set_var(key, val) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bool_truthy() {
        for t in ["1", "true", "TRUE", "yes", "on"] {
            assert!(parse_bool(t), "{t} should be true");
        }
        for f in ["0", "false", "no", "", "off"] {
            assert!(!parse_bool(f), "{f} should be false");
        }
    }
}
