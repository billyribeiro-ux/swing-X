//! DB-row -> DTO mapping. Every query selects exactly the columns the DTO needs and
//! converts timestamps to RFC-3339 strings and jsonb to typed driver vectors. All
//! reads handle the "table empty" case by returning an empty vec.

use chrono::{DateTime, Utc};
use se_core::{Error, Result};
use se_store::sqlx;
use se_store::sqlx::Row;
use se_store::Store;
use uuid::Uuid;

use crate::dto::{
    monitor_action_dto, ChangelogWeekDto, DriverDto, MonitorEventDto, OosScoreDto, SignalDto,
    StrategyDto, TradeDto,
};

fn store_err(e: impl std::fmt::Display) -> Error {
    Error::Store(e.to_string())
}

fn iso(ts: DateTime<Utc>) -> String {
    ts.to_rfc3339()
}

/// Parse a jsonb `why`/`attribution` array into typed drivers. Tolerant: a missing
/// or malformed array yields an empty vec rather than an error (the dashboard treats
/// attribution as optional context, never a hard dependency).
fn drivers_from_json(v: &serde_json::Value) -> Vec<DriverDto> {
    v.as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|d| {
                    Some(DriverDto {
                        layer: d.get("layer")?.as_str()?.to_string(),
                        key: d.get("key")?.as_str()?.to_string(),
                        contribution: d
                            .get("contribution")
                            .and_then(|x| x.as_f64())
                            .unwrap_or(0.0),
                        detail: d
                            .get("detail")
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Render a jsonb `detail` blob to a human string for the monitor feed. Prefers a
/// `note` field, falls back to the compact JSON.
fn detail_to_string(v: &serde_json::Value) -> String {
    if let Some(s) = v.get("note").and_then(|x| x.as_str()) {
        return s.to_string();
    }
    if v.is_null() {
        return String::new();
    }
    v.to_string()
}

// ---------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------

pub async fn signals(store: &Store) -> Result<Vec<SignalDto>> {
    let rows = sqlx::query(
        "SELECT signal_id, strategy_id, ticker, side, decision_ts, horizon, entry, stop, \
                target1, target2, rr1, rr2, conviction, cohort_n, regime_desc, why, \
                invalidation, cohort_expectancy, cvar5, lead_time, payload_json \
         FROM signals ORDER BY decision_ts DESC LIMIT 500",
    )
    .fetch_all(store.pool())
    .await
    .map_err(store_err)?;
    Ok(rows.iter().map(signal_from_row).collect())
}

pub async fn signal_by_id(store: &Store, id: &str) -> Result<Option<SignalDto>> {
    let uuid = match Uuid::parse_str(id) {
        Ok(u) => u,
        Err(_) => return Ok(None),
    };
    let row = sqlx::query(
        "SELECT signal_id, strategy_id, ticker, side, decision_ts, horizon, entry, stop, \
                target1, target2, rr1, rr2, conviction, cohort_n, regime_desc, why, \
                invalidation, cohort_expectancy, cvar5, lead_time, payload_json \
         FROM signals WHERE signal_id = $1",
    )
    .bind(uuid)
    .fetch_optional(store.pool())
    .await
    .map_err(store_err)?;
    Ok(row.as_ref().map(signal_from_row))
}

fn signal_from_row(row: &sqlx::postgres::PgRow) -> SignalDto {
    let why: serde_json::Value = row.try_get("why").unwrap_or(serde_json::Value::Null);
    let payload_json: serde_json::Value = row
        .try_get("payload_json")
        .unwrap_or(serde_json::Value::Null);
    // lead_time is TEXT in the DB; the dashboard wants a number (minutes). Parse a
    // leading integer if present, else omit.
    let lead_time: Option<f64> = row
        .try_get::<Option<String>, _>("lead_time")
        .ok()
        .flatten()
        .and_then(|s| parse_minutes(&s));
    SignalDto {
        signal_id: row.get::<Uuid, _>("signal_id").to_string(),
        strategy_id: row.get::<Uuid, _>("strategy_id").to_string(),
        ticker: row.get("ticker"),
        side: row.get("side"),
        decision_ts: iso(row.get("decision_ts")),
        horizon: row.get("horizon"),
        entry: row.get("entry"),
        stop: row.get("stop"),
        target1: row.get("target1"),
        target2: row.try_get("target2").ok().flatten(),
        rr1: row.try_get("rr1").ok().flatten(),
        rr2: row.try_get("rr2").ok().flatten(),
        conviction: row.get("conviction"),
        cohort_n: row.get::<i32, _>("cohort_n") as i64,
        regime_desc: row.get("regime_desc"),
        why: drivers_from_json(&why),
        invalidation: row.get("invalidation"),
        cohort_expectancy: row.try_get("cohort_expectancy").ok().flatten(),
        cvar5: row.try_get("cvar5").ok().flatten(),
        lead_time,
        payload_json,
    }
}

/// Extract a leading numeric (minutes) from a lead-time string like "34" or "34m".
fn parse_minutes(s: &str) -> Option<f64> {
    let trimmed = s.trim();
    let digits: String = trimmed
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
        .collect();
    digits.parse::<f64>().ok()
}

// ---------------------------------------------------------------------------
// Population (strategies + latest oos_score)
// ---------------------------------------------------------------------------

pub async fn population(store: &Store) -> Result<Vec<StrategyDto>> {
    let rows = sqlx::query(
        "SELECT s.strategy_id, s.horizon, s.status, s.generation, s.genome, \
                o.dsr, o.pbo, o.oos_expectancy_cost_aware, o.profit_factor, o.cvar5, o.mar, \
                o.n_regimes_positive, o.passed_gate, o.evaluated_at \
         FROM strategies s \
         LEFT JOIN LATERAL ( \
             SELECT dsr, pbo, oos_expectancy_cost_aware, profit_factor, cvar5, mar, \
                    n_regimes_positive, passed_gate, evaluated_at \
             FROM oos_scores WHERE strategy_id = s.strategy_id \
             ORDER BY evaluated_at DESC LIMIT 1 \
         ) o ON TRUE \
         ORDER BY s.generation DESC, s.created_at DESC LIMIT 500",
    )
    .fetch_all(store.pool())
    .await
    .map_err(store_err)?;

    Ok(rows
        .iter()
        .map(|row| {
            let genome: serde_json::Value =
                row.try_get("genome").unwrap_or(serde_json::Value::Null);
            let evaluated_at: Option<DateTime<Utc>> = row.try_get("evaluated_at").ok().flatten();
            // A score row exists only if it was evaluated; gate on evaluated_at.
            let latest_score = evaluated_at.map(|ts| OosScoreDto {
                dsr: row.try_get("dsr").ok().flatten().unwrap_or(0.0),
                pbo: row.try_get("pbo").ok().flatten().unwrap_or(0.0),
                oos_expectancy_cost_aware: row
                    .try_get("oos_expectancy_cost_aware")
                    .ok()
                    .flatten()
                    .unwrap_or(0.0),
                profit_factor: row.try_get("profit_factor").ok().flatten().unwrap_or(0.0),
                cvar5: row.try_get("cvar5").ok().flatten().unwrap_or(0.0),
                mar: row.try_get("mar").ok().flatten().unwrap_or(0.0),
                n_regimes_positive: row.get::<i32, _>("n_regimes_positive") as i64,
                passed_gate: row.get("passed_gate"),
                evaluated_at: iso(ts),
            });
            StrategyDto {
                strategy_id: row.get::<Uuid, _>("strategy_id").to_string(),
                horizon: row.get("horizon"),
                status: row.get("status"),
                generation: row.get::<i32, _>("generation") as i64,
                genome_summary: genome_summary(&genome),
                latest_score,
            }
        })
        .collect())
}

/// Compact one-line summary of a genome jsonb (side · horizon · predicate count).
fn genome_summary(g: &serde_json::Value) -> String {
    let side = g.get("side").and_then(|s| s.as_str()).unwrap_or("?");
    let horizon = g.get("horizon").and_then(|s| s.as_str()).unwrap_or("?");
    let preds = g.get("predicates").and_then(|p| p.as_array());
    match preds {
        Some(arr) => {
            let conds: Vec<String> = arr
                .iter()
                .filter_map(|p| {
                    let key = p.get("feature_key")?.as_str()?;
                    let op = p.get("op")?.as_str()?;
                    let thr = p.get("threshold")?.as_f64()?;
                    Some(format!("{key} {op} {thr:.3}"))
                })
                .collect();
            if conds.is_empty() {
                format!("{side} {horizon}")
            } else {
                format!("{side} {horizon} :: {}", conds.join(" AND "))
            }
        }
        None => format!("{side} {horizon}"),
    }
}

// ---------------------------------------------------------------------------
// Monitor events
// ---------------------------------------------------------------------------

pub async fn monitor_events(store: &Store) -> Result<Vec<MonitorEventDto>> {
    let rows = sqlx::query(
        "SELECT ts, detector, ticker, strategy_id, metric_value, threshold, action_taken, detail \
         FROM monitor_events ORDER BY ts DESC LIMIT 500",
    )
    .fetch_all(store.pool())
    .await
    .map_err(store_err)?;
    Ok(rows.iter().map(monitor_event_from_row).collect())
}

fn monitor_event_from_row(row: &sqlx::postgres::PgRow) -> MonitorEventDto {
    let detail: serde_json::Value = row.try_get("detail").unwrap_or(serde_json::Value::Null);
    MonitorEventDto {
        ts: iso(row.get("ts")),
        detector: row.get("detector"),
        ticker: row.try_get("ticker").ok().flatten(),
        strategy_id: row
            .try_get::<Option<Uuid>, _>("strategy_id")
            .ok()
            .flatten()
            .map(|u| u.to_string()),
        metric_value: row.try_get("metric_value").ok().flatten(),
        threshold: row.try_get("threshold").ok().flatten(),
        action_taken: monitor_action_dto(&row.get::<String, _>("action_taken")).to_string(),
        detail: detail_to_string(&detail),
    }
}

// ---------------------------------------------------------------------------
// Journal
// ---------------------------------------------------------------------------

pub async fn journal(store: &Store) -> Result<Vec<TradeDto>> {
    let rows = sqlx::query(
        "SELECT trade_id, signal_id, strategy_id, ticker, side, mode, entry_ts, fill_px, fill_ts, \
                exit_ts, exit_px, pnl_r, cost_frac, attribution \
         FROM trades_journal ORDER BY entry_ts DESC LIMIT 500",
    )
    .fetch_all(store.pool())
    .await
    .map_err(store_err)?;
    Ok(rows.iter().map(trade_from_row).collect())
}

fn trade_from_row(row: &sqlx::postgres::PgRow) -> TradeDto {
    let attribution: serde_json::Value = row
        .try_get("attribution")
        .unwrap_or(serde_json::Value::Null);
    let exit_ts: Option<DateTime<Utc>> = row.try_get("exit_ts").ok().flatten();
    TradeDto {
        trade_id: row.get::<Uuid, _>("trade_id").to_string(),
        signal_id: row
            .try_get::<Option<Uuid>, _>("signal_id")
            .ok()
            .flatten()
            .map(|u| u.to_string()),
        strategy_id: row
            .try_get::<Option<Uuid>, _>("strategy_id")
            .ok()
            .flatten()
            .map(|u| u.to_string()),
        ticker: row.get("ticker"),
        side: row.get("side"),
        mode: row.get("mode"),
        entry_ts: iso(row.get("entry_ts")),
        fill_px: row.get("fill_px"),
        fill_ts: iso(row.get("fill_ts")),
        exit_ts: exit_ts.map(iso),
        exit_px: row.try_get("exit_px").ok().flatten(),
        pnl_r: row.try_get("pnl_r").ok().flatten(),
        cost_frac: row.try_get("cost_frac").ok().flatten(),
        attribution: drivers_from_json(&attribution),
    }
}

// ---------------------------------------------------------------------------
// Changelog
// ---------------------------------------------------------------------------

pub async fn changelog(store: &Store) -> Result<Vec<ChangelogWeekDto>> {
    let rows = sqlx::query(
        "SELECT week, decayed, retired, adapted FROM changelog ORDER BY week DESC LIMIT 200",
    )
    .fetch_all(store.pool())
    .await
    .map_err(store_err)?;
    Ok(rows
        .iter()
        .map(|row| {
            let week: chrono::NaiveDate = row.get("week");
            ChangelogWeekDto {
                week: iso_week_label(week),
                decayed: string_array(&row.try_get("decayed").unwrap_or(serde_json::Value::Null)),
                retired: string_array(&row.try_get("retired").unwrap_or(serde_json::Value::Null)),
                adapted: string_array(&row.try_get("adapted").unwrap_or(serde_json::Value::Null)),
            }
        })
        .collect())
}

/// Coerce a jsonb array to `Vec<String>`. Non-string members are stringified.
fn string_array(v: &serde_json::Value) -> Vec<String> {
    v.as_array()
        .map(|arr| {
            arr.iter()
                .map(|x| match x.as_str() {
                    Some(s) => s.to_string(),
                    None => x.to_string(),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Render a `DATE` (week Monday) as the dashboard's ISO-week label `YYYY-Www`.
fn iso_week_label(d: chrono::NaiveDate) -> String {
    use chrono::Datelike;
    let iso = d.iso_week();
    format!("{}-W{:02}", iso.year(), iso.week())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drivers_parse_tolerantly() {
        let v = serde_json::json!([
            { "layer": "regime", "key": "k", "contribution": 0.3, "detail": "d" },
            { "layer": "trigger", "key": "k2" }
        ]);
        let ds = drivers_from_json(&v);
        assert_eq!(ds.len(), 2);
        assert_eq!(ds[0].key, "k");
        assert_eq!(ds[1].contribution, 0.0);
        // null / non-array -> empty
        assert!(drivers_from_json(&serde_json::Value::Null).is_empty());
    }

    #[test]
    fn detail_prefers_note() {
        assert_eq!(
            detail_to_string(&serde_json::json!({ "note": "hi", "x": 1 })),
            "hi"
        );
        assert_eq!(detail_to_string(&serde_json::Value::Null), "");
        assert_eq!(
            detail_to_string(&serde_json::json!({ "x": 1 })),
            "{\"x\":1}"
        );
    }

    #[test]
    fn iso_week_label_format() {
        // 2026-06-22 is a Monday in ISO week 26.
        let d = chrono::NaiveDate::from_ymd_opt(2026, 6, 22).unwrap();
        assert_eq!(iso_week_label(d), "2026-W26");
    }

    #[test]
    fn parse_minutes_handles_suffixes() {
        assert_eq!(parse_minutes("34"), Some(34.0));
        assert_eq!(parse_minutes("34m"), Some(34.0));
        assert_eq!(parse_minutes("eod"), None);
    }

    #[test]
    fn genome_summary_compacts() {
        let g = serde_json::json!({
            "side": "Long", "horizon": "swing",
            "predicates": [{ "feature_key": "gex", "op": ">", "threshold": 0.5 }]
        });
        let s = genome_summary(&g);
        assert!(s.contains("gex"));
        assert!(s.contains("swing"));
    }
}
