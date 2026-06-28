//! `se-monitor` (P6) — the forward-adaptation watch/detect/act loop.
//!
//! Once a day [`run_daily`] sweeps the database and runs a set of detectors. Each
//! detector maps a **metric** over recent live behaviour to a **threshold** and,
//! when breached, a [`se_core::MonitorAction`]. Every firing:
//!   1. writes a `monitor_events` row (detector, metric_value, threshold,
//!      action_taken, detail jsonb), and
//!   2. drives the matching side effect so the engine never silently keeps trading
//!      something broken (e.g. Quarantine/Disable flip `strategies.status`).
//!
//! Detectors implemented (see [`detectors`] for the pure math + thresholds):
//!   * **backtest-vs-live divergence** — rolling realized expectancy vs OOS expectancy.
//!   * **drawdown breach** — per-strategy CVaR(5%) / max-drawdown beyond a floor.
//!   * **calibration break** — predicted conviction vs realized hit-rate gap.
//!   * **data outage / staleness** — freshness of latest bar/feature per source.
//!   * **regime OOD** — count of recent `out_of_distribution` regime labels.
//!
//! [`weekly_changelog`] summarizes what decayed / retired / adapted into `changelog`.
//!
//! Inputs that are absent (no trades, no scores yet) are skipped gracefully — the
//! monitor never fabricates a verdict from missing data.

pub mod detectors;

use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc};
use se_core::{MonitorAction, Result};
use se_store::sqlx;
use se_store::Store;
use serde_json::json;
use uuid::Uuid;

pub use detectors::{Decision, Thresholds};

/// Lookback (days) for the regime-OOD count.
const OOD_WINDOW_DAYS: i64 = 5;
/// Lookback (days) for "recent" realized trades per strategy.
const TRADE_WINDOW_DAYS: i64 = 60;

/// Summary of a daily run — how many of each detector fired. Returned so callers
/// (CLI / tests) can assert/log without re-querying.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct DailyReport {
    pub divergence: usize,
    pub drawdown: usize,
    pub calibration: usize,
    pub staleness: usize,
    pub regime_ood: usize,
}

impl DailyReport {
    pub fn total(&self) -> usize {
        self.divergence + self.drawdown + self.calibration + self.staleness + self.regime_ood
    }
}

/// Run every detector once over the current database state, emitting
/// `monitor_events` and applying side effects. Idempotency note: this is a poll —
/// running it twice in a day will emit duplicate events; the orchestrator is
/// expected to call it once per daily cycle.
pub async fn run_daily(store: &Store) -> Result<DailyReport> {
    run_daily_with(store, &Thresholds::default(), Utc::now()).await
}

/// `run_daily` with explicit thresholds and a fixed "now" (for deterministic tests).
pub async fn run_daily_with(
    store: &Store,
    t: &Thresholds,
    now: DateTime<Utc>,
) -> Result<DailyReport> {
    let pool = store.pool();
    let mut report = DailyReport::default();

    // ---- per-strategy detectors (divergence, drawdown, calibration) -------
    for strat in load_strategies(pool).await? {
        let trades = load_recent_trades(pool, strat.strategy_id).await?;
        let realized_r: Vec<f64> = trades.iter().filter_map(|tr| tr.pnl_r).collect();

        // backtest-vs-live divergence (needs an OOS expectancy to compare to)
        if let Some(oos) = strat.oos_expectancy {
            if let Some(d) = detectors::detect_divergence(&realized_r, oos, t) {
                apply_strategy_decision(pool, now, &strat, "backtest_vs_live_divergence", &d)
                    .await?;
                report.divergence += 1;
            }
        }

        // drawdown breach -> Disable + paired Alert
        if let Some(d) = detectors::detect_drawdown(&realized_r, t) {
            apply_strategy_decision(pool, now, &strat, "drawdown_breach", &d).await?;
            // paired alert so an operator is notified, not just the status flip.
            write_event(
                pool,
                now,
                Some(strat.strategy_id),
                None,
                "drawdown_breach",
                Some(d.metric_value),
                Some(d.threshold),
                MonitorAction::Alert,
                &json!({ "note": d.note, "paired_with": "disable" }),
            )
            .await?;
            report.drawdown += 1;
        }

        // calibration break -> Recalibrate
        let convictions: Vec<f64> = trades.iter().filter_map(|tr| tr.conviction).collect();
        let wins: Vec<bool> = trades
            .iter()
            .filter(|tr| tr.conviction.is_some())
            .filter_map(|tr| tr.pnl_r.map(|r| r > 0.0))
            .collect();
        if let Some(d) = detectors::detect_calibration(&convictions, &wins, t) {
            apply_strategy_decision(pool, now, &strat, "calibration_break", &d).await?;
            report.calibration += 1;
        }
    }

    // ---- data outage / staleness (per source) -----------------------------
    for (source, latest) in load_source_freshness(pool).await? {
        let age_hours = (now - latest).num_minutes() as f64 / 60.0;
        if let Some(d) = detectors::detect_staleness(age_hours, t) {
            write_event(
                pool,
                now,
                None,
                None,
                "data_staleness",
                Some(d.metric_value),
                Some(d.threshold),
                d.action,
                &json!({ "note": d.note, "source": source, "latest": latest.to_rfc3339() }),
            )
            .await?;
            write_event(
                pool,
                now,
                None,
                None,
                "data_staleness",
                Some(d.metric_value),
                Some(d.threshold),
                MonitorAction::Alert,
                &json!({ "note": d.note, "source": source, "paired_with": "skip_degraded" }),
            )
            .await?;
            report.staleness += 1;
        }
    }

    // ---- regime OOD (recent out_of_distribution labels) -------------------
    let ood = count_recent_ood(pool, now).await?;
    if let Some(d) = detectors::detect_regime_ood(ood, t) {
        write_event(
            pool,
            now,
            None,
            None,
            "regime_ood",
            Some(d.metric_value),
            Some(d.threshold),
            d.action,
            &json!({ "note": d.note, "window_days": OOD_WINDOW_DAYS }),
        )
        .await?;
        report.regime_ood += 1;
    }

    Ok(report)
}

// ---------------------------------------------------------------------------
// Side effects
// ---------------------------------------------------------------------------

/// Write the event row, then apply the lifecycle side effect implied by the action.
async fn apply_strategy_decision(
    pool: &sqlx::PgPool,
    now: DateTime<Utc>,
    strat: &StratRow,
    detector: &str,
    d: &Decision,
) -> Result<()> {
    write_event(
        pool,
        now,
        Some(strat.strategy_id),
        None,
        detector,
        Some(d.metric_value),
        Some(d.threshold),
        d.action,
        &json!({ "note": d.note }),
    )
    .await?;
    // Map the action to a strategy-status transition where one applies. We never
    // keep trading a strategy a detector just condemned.
    let new_status = match d.action {
        MonitorAction::Quarantine => Some("quarantined"),
        MonitorAction::Disable | MonitorAction::Retire => Some("retired"),
        MonitorAction::Demote => Some("demoted"),
        // ShrinkSize / Recalibrate / Suppress / etc. don't change lifecycle status.
        _ => None,
    };
    if let Some(status) = new_status {
        set_strategy_status(pool, strat.strategy_id, status).await?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn write_event(
    pool: &sqlx::PgPool,
    ts: DateTime<Utc>,
    strategy_id: Option<Uuid>,
    ticker: Option<&str>,
    detector: &str,
    metric_value: Option<f64>,
    threshold: Option<f64>,
    action: MonitorAction,
    detail: &serde_json::Value,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO monitor_events \
         (ts, strategy_id, ticker, detector, metric_value, threshold, action_taken, detail) \
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
    )
    .bind(ts)
    .bind(strategy_id)
    .bind(ticker)
    .bind(detector)
    .bind(metric_value)
    .bind(threshold)
    .bind(action.as_str())
    .bind(detail)
    .execute(pool)
    .await
    .map_err(|e| se_core::Error::Store(e.to_string()))?;
    Ok(())
}

async fn set_strategy_status(pool: &sqlx::PgPool, id: Uuid, status: &str) -> Result<()> {
    sqlx::query("UPDATE strategies SET status = $1, updated_at = now() WHERE strategy_id = $2")
        .bind(status)
        .bind(id)
        .execute(pool)
        .await
        .map_err(|e| se_core::Error::Store(e.to_string()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Reads (all via store.pool(); these are operational reads, not PIT decisioning)
// ---------------------------------------------------------------------------

struct StratRow {
    strategy_id: Uuid,
    oos_expectancy: Option<f64>,
}

async fn load_strategies(pool: &sqlx::PgPool) -> Result<Vec<StratRow>> {
    // Join each strategy to its latest OOS expectancy (if any). LEFT JOIN keeps
    // strategies without a score, whose divergence detector is simply skipped.
    let rows: Vec<(Uuid, Option<f64>)> = sqlx::query_as(
        "SELECT s.strategy_id, o.oos_expectancy_cost_aware \
         FROM strategies s \
         LEFT JOIN LATERAL ( \
             SELECT oos_expectancy_cost_aware FROM oos_scores \
             WHERE strategy_id = s.strategy_id \
             ORDER BY evaluated_at DESC LIMIT 1 \
         ) o ON TRUE \
         WHERE s.status <> 'retired'",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| se_core::Error::Store(e.to_string()))?;
    Ok(rows
        .into_iter()
        .map(|(strategy_id, oos_expectancy)| StratRow {
            strategy_id,
            oos_expectancy,
        })
        .collect())
}

struct TradeRow {
    pnl_r: Option<f64>,
    conviction: Option<f64>,
}

async fn load_recent_trades(pool: &sqlx::PgPool, strategy_id: Uuid) -> Result<Vec<TradeRow>> {
    // Realized R from trades_journal, with the originating signal's conviction for
    // the calibration detector. Only closed trades (pnl_r present) carry a hit/loss.
    let rows: Vec<(Option<f64>, Option<f64>)> = sqlx::query_as(
        "SELECT t.pnl_r, sg.conviction \
         FROM trades_journal t \
         LEFT JOIN signals sg ON sg.signal_id = t.signal_id \
         WHERE t.strategy_id = $1 \
           AND t.entry_ts >= now() - ($2 || ' days')::interval \
         ORDER BY t.entry_ts DESC",
    )
    .bind(strategy_id)
    .bind(TRADE_WINDOW_DAYS.to_string())
    .fetch_all(pool)
    .await
    .map_err(|e| se_core::Error::Store(e.to_string()))?;
    Ok(rows
        .into_iter()
        .map(|(pnl_r, conviction)| TradeRow { pnl_r, conviction })
        .collect())
}

async fn load_source_freshness(pool: &sqlx::PgPool) -> Result<Vec<(String, DateTime<Utc>)>> {
    // Latest knowable timestamp per source across bars and features.
    let rows: Vec<(String, DateTime<Utc>)> = sqlx::query_as(
        "SELECT source, MAX(ts) AS latest FROM bars GROUP BY source \
         UNION ALL \
         SELECT source, MAX(decision_ts) AS latest FROM features_pit GROUP BY source",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| se_core::Error::Store(e.to_string()))?;
    Ok(rows)
}

async fn count_recent_ood(pool: &sqlx::PgPool, now: DateTime<Utc>) -> Result<i64> {
    let since = now - Duration::days(OOD_WINDOW_DAYS);
    let row: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM regimes \
         WHERE regime_label = 'out_of_distribution' AND decision_ts >= $1",
    )
    .bind(since)
    .fetch_one(pool)
    .await
    .map_err(|e| se_core::Error::Store(e.to_string()))?;
    Ok(row.0)
}

// ---------------------------------------------------------------------------
// Weekly changelog
// ---------------------------------------------------------------------------

/// Summarize the week's adaptation into the `changelog` table:
///   * **decayed** — strategies/features trending down (status moved to
///     demoted/quarantined, or this week's decay detectors fired).
///   * **retired** — strategies retired this week.
///   * **adapted** — corrective actions the monitor took (recalibrate / shrink /
///     suppress / skip-degraded / force-refit) this week.
///
/// "Importance trending down" is approximated from status transitions and the decay
/// detectors' firings — we do not keep a separate importance series, so we surface
/// the observable proxies rather than fabricate a number.
pub async fn weekly_changelog(store: &Store) -> Result<NaiveDate> {
    weekly_changelog_at(store, Utc::now()).await
}

/// `weekly_changelog` anchored at an explicit instant (for tests). Returns the ISO
/// week's Monday `DATE` used as the `changelog` primary key.
pub async fn weekly_changelog_at(store: &Store, now: DateTime<Utc>) -> Result<NaiveDate> {
    let pool = store.pool();
    let week_start = monday_of(now.date_naive());
    let week_start_ts = week_start
        .and_hms_opt(0, 0, 0)
        .expect("00:00:00 valid")
        .and_utc();

    // decayed: strategies currently demoted/quarantined + this week's decay events.
    let decayed_strats: Vec<(String, String)> = sqlx::query_as(
        "SELECT strategy_id::text, status FROM strategies \
         WHERE status IN ('demoted','quarantined') AND updated_at >= $1",
    )
    .bind(week_start_ts)
    .fetch_all(pool)
    .await
    .map_err(|e| se_core::Error::Store(e.to_string()))?;

    let decay_events: Vec<(String, Option<f64>)> = sqlx::query_as(
        "SELECT detector, metric_value FROM monitor_events \
         WHERE ts >= $1 \
           AND detector IN ('backtest_vs_live_divergence','calibration_break','drawdown_breach') \
         ORDER BY ts DESC",
    )
    .bind(week_start_ts)
    .fetch_all(pool)
    .await
    .map_err(|e| se_core::Error::Store(e.to_string()))?;

    let mut decayed: Vec<String> = decayed_strats
        .iter()
        .map(|(id, status)| format!("strategy {id} -> {status}"))
        .collect();
    decayed.extend(decay_events.iter().map(|(det, mv)| match mv {
        Some(v) => format!("{det} fired (metric {v:.3})"),
        None => format!("{det} fired"),
    }));

    // retired: strategies moved to retired this week.
    let retired: Vec<String> = sqlx::query_as::<_, (String,)>(
        "SELECT strategy_id::text FROM strategies \
         WHERE status = 'retired' AND updated_at >= $1",
    )
    .bind(week_start_ts)
    .fetch_all(pool)
    .await
    .map_err(|e| se_core::Error::Store(e.to_string()))?
    .into_iter()
    .map(|(id,)| format!("strategy {id} retired"))
    .collect();

    // adapted: corrective actions taken this week.
    let adapted: Vec<String> = sqlx::query_as::<_, (String, String)>(
        "SELECT detector, action_taken FROM monitor_events \
         WHERE ts >= $1 \
           AND action_taken IN ('recalibrate','shrink_size','suppress','skip_degraded','force_refit') \
         ORDER BY ts DESC",
    )
    .bind(week_start_ts)
    .fetch_all(pool)
    .await
    .map_err(|e| se_core::Error::Store(e.to_string()))?
    .into_iter()
    .map(|(det, action)| format!("{det} -> {action}"))
    .collect();

    sqlx::query(
        "INSERT INTO changelog (week, decayed, retired, adapted) VALUES ($1,$2,$3,$4) \
         ON CONFLICT (week) DO UPDATE SET \
           decayed = EXCLUDED.decayed, retired = EXCLUDED.retired, adapted = EXCLUDED.adapted",
    )
    .bind(week_start)
    .bind(json!(decayed))
    .bind(json!(retired))
    .bind(json!(adapted))
    .execute(pool)
    .await
    .map_err(|e| se_core::Error::Store(e.to_string()))?;

    Ok(week_start)
}

/// Monday of the ISO week containing `d`.
fn monday_of(d: NaiveDate) -> NaiveDate {
    let dow = d.weekday().num_days_from_monday() as i64;
    d - Duration::days(dow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn monday_of_normalizes_to_week_start() {
        // 2026-06-28 is a Sunday -> Monday is 2026-06-22.
        let sun = NaiveDate::from_ymd_opt(2026, 6, 28).unwrap();
        assert_eq!(
            monday_of(sun),
            NaiveDate::from_ymd_opt(2026, 6, 22).unwrap()
        );
        let mon = NaiveDate::from_ymd_opt(2026, 6, 22).unwrap();
        assert_eq!(monday_of(mon), mon);
    }

    #[test]
    fn daily_report_totals() {
        let r = DailyReport {
            divergence: 1,
            drawdown: 2,
            calibration: 0,
            staleness: 3,
            regime_ood: 1,
        };
        assert_eq!(r.total(), 7);
    }

    #[test]
    fn ood_window_anchoring() {
        // sanity: the window subtraction is well-formed.
        let now = Utc.with_ymd_and_hms(2026, 6, 28, 12, 0, 0).unwrap();
        let since = now - Duration::days(OOD_WINDOW_DAYS);
        assert_eq!(since, Utc.with_ymd_and_hms(2026, 6, 23, 12, 0, 0).unwrap());
    }
}
