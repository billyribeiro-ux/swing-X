//! Wire DTOs for the operator dashboard.
//!
//! These are the EXACT camelCase shapes the SvelteKit app validates with its zod
//! schemas (`apps/web/src/lib/api/schemas.ts`). They are deliberately decoupled
//! from the DB rows and the `se-core` domain types: the API maps DB rows -> DTO so
//! a schema change on either side is a localized edit here.
//!
//! Note on `actionTaken`: `se_core::MonitorAction` has ten variants, but the
//! dashboard's `MONITOR_ACTIONS` enum is the smaller operator-facing set
//! (`shrink|quarantine|refit|recalibrate|suppress|disable|alert`). [`monitor_action_dto`]
//! folds the richer engine actions onto that set.

use serde::Serialize;

/// One driver behind a signal / trade attribution entry.
#[derive(Debug, Clone, Serialize)]
pub struct DriverDto {
    pub layer: String,
    pub key: String,
    pub contribution: f64,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SignalDto {
    pub signal_id: String,
    pub strategy_id: String,
    pub ticker: String,
    pub side: String,
    /// ISO-8601.
    pub decision_ts: String,
    pub horizon: String,
    pub entry: f64,
    pub stop: f64,
    pub target1: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target2: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rr1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rr2: Option<f64>,
    pub conviction: f64,
    pub cohort_n: i64,
    pub regime_desc: String,
    pub why: Vec<DriverDto>,
    pub invalidation: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cohort_expectancy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cvar5: Option<f64>,
    /// Lead-time edge in minutes (numeric per the dashboard schema).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lead_time: Option<f64>,
    pub payload_json: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OosScoreDto {
    pub dsr: f64,
    pub pbo: f64,
    pub oos_expectancy_cost_aware: f64,
    pub profit_factor: f64,
    pub cvar5: f64,
    pub mar: f64,
    pub n_regimes_positive: i64,
    pub passed_gate: bool,
    /// ISO-8601.
    pub evaluated_at: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StrategyDto {
    pub strategy_id: String,
    pub horizon: String,
    pub status: String,
    pub generation: i64,
    pub genome_summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latest_score: Option<OosScoreDto>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MonitorEventDto {
    /// ISO-8601.
    pub ts: String,
    pub detector: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ticker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strategy_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metric_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    pub action_taken: String,
    /// Human-readable string (the dashboard schema is `z.string()`); the raw
    /// `detail` jsonb is rendered to a note.
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TradeDto {
    pub trade_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strategy_id: Option<String>,
    pub ticker: String,
    pub side: String,
    pub mode: String,
    /// ISO-8601.
    pub entry_ts: String,
    pub fill_px: f64,
    /// ISO-8601.
    pub fill_ts: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_ts: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_px: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pnl_r: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_frac: Option<f64>,
    pub attribution: Vec<DriverDto>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChangelogWeekDto {
    pub week: String,
    pub decayed: Vec<String>,
    pub retired: Vec<String>,
    pub adapted: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HealthDto {
    pub status: String,
}

/// Fold the engine's `monitor_events.action_taken` string onto the dashboard's
/// operator-facing `MONITOR_ACTIONS` set. Unknown values map to `alert` so the
/// dashboard never rejects a payload it cannot enumerate.
pub fn monitor_action_dto(action: &str) -> &'static str {
    match action {
        "shrink_size" => "shrink",
        "quarantine" => "quarantine",
        "force_refit" => "refit",
        "recalibrate" => "recalibrate",
        "suppress" => "suppress",
        "skip_degraded" => "suppress",
        "disable" | "retire" | "demote" => "disable",
        "alert" => "alert",
        _ => "alert",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_mapping_covers_engine_actions() {
        assert_eq!(monitor_action_dto("shrink_size"), "shrink");
        assert_eq!(monitor_action_dto("force_refit"), "refit");
        assert_eq!(monitor_action_dto("skip_degraded"), "suppress");
        assert_eq!(monitor_action_dto("retire"), "disable");
        assert_eq!(monitor_action_dto("demote"), "disable");
        assert_eq!(monitor_action_dto("alert"), "alert");
        // unknown -> alert (never rejected by the dashboard enum)
        assert_eq!(monitor_action_dto("nonsense"), "alert");
    }

    #[test]
    fn optional_fields_are_omitted_when_none() {
        let s = SignalDto {
            signal_id: "s".into(),
            strategy_id: "x".into(),
            ticker: "SPY".into(),
            side: "long".into(),
            decision_ts: "2026-06-28T00:00:00Z".into(),
            horizon: "swing".into(),
            entry: 1.0,
            stop: 0.9,
            target1: 1.2,
            target2: None,
            rr1: Some(2.0),
            rr2: None,
            conviction: 0.6,
            cohort_n: 10,
            regime_desc: "r".into(),
            why: vec![],
            invalidation: "x".into(),
            cohort_expectancy: None,
            cvar5: None,
            lead_time: None,
            payload_json: serde_json::json!({}),
        };
        let v = serde_json::to_value(&s).unwrap();
        assert!(v.get("target2").is_none());
        assert!(v.get("rr2").is_none());
        assert!(v.get("rr1").is_some());
        // camelCase rename check
        assert!(v.get("signalId").is_some());
        assert!(v.get("cohortN").is_some());
        assert!(v.get("payloadJson").is_some());
    }
}
