//! Direct sqlx persistence for the population, against the existing `strategies` and
//! `oos_scores` tables (se-store is off-limits, so we use its public `pool()` + re-exported
//! `sqlx`). No schema changes; only inserts/updates of columns defined in `0001_init.sql`.

use se_core::{Genome, Result, Scanner, Strategy, StrategyId, StrategyStatus};
use se_store::Store;

use crate::score::OosScore;

fn store_err(e: impl std::fmt::Display) -> se_core::Error {
    se_core::Error::Store(e.to_string())
}

/// Insert (or update) a strategy row: genome jsonb, status, generation, parent, scanner.
pub async fn upsert_strategy(store: &Store, strategy: &Strategy, scanner: Scanner) -> Result<()> {
    let genome_json = serde_json::to_value(&strategy.genome).map_err(store_err)?;
    se_store::sqlx::query(
        "INSERT INTO strategies \
            (strategy_id, horizon, genome, parent_id, status, generation, scanner) \
         VALUES ($1, $2, $3, $4, $5, $6, $7) \
         ON CONFLICT (strategy_id) DO UPDATE SET \
             genome = EXCLUDED.genome, status = EXCLUDED.status, \
             generation = EXCLUDED.generation, parent_id = EXCLUDED.parent_id, \
             updated_at = now()",
    )
    .bind(strategy.id.inner())
    .bind(strategy.genome.horizon.as_str())
    .bind(genome_json)
    .bind(strategy.parent.map(|p| p.inner()))
    .bind(strategy.status.as_str())
    .bind(strategy.generation as i32)
    .bind(scanner.as_str())
    .execute(store.pool())
    .await
    .map_err(store_err)?;
    Ok(())
}

/// Update only a strategy's lifecycle status (e.g. promote / kill).
pub async fn update_status(store: &Store, id: StrategyId, status: StrategyStatus) -> Result<()> {
    se_store::sqlx::query(
        "UPDATE strategies SET status = $2, updated_at = now() WHERE strategy_id = $1",
    )
    .bind(id.inner())
    .bind(status.as_str())
    .execute(store.pool())
    .await
    .map_err(store_err)?;
    Ok(())
}

/// Insert an OOS score row for a strategy. `fold_spec` is stored as JSON for provenance.
pub async fn insert_oos_score(
    store: &Store,
    score: &OosScore,
    fold_spec: &serde_json::Value,
) -> Result<()> {
    let regime_contrib = serde_json::to_value(&score.regime_contrib).map_err(store_err)?;
    se_store::sqlx::query(
        "INSERT INTO oos_scores \
            (strategy_id, fold_spec, dsr, pbo, oos_expectancy_cost_aware, profit_factor, \
             cvar5, mar, regime_contrib, n_regimes_positive, passed_gate) \
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)",
    )
    .bind(score.strategy_id.inner())
    .bind(fold_spec)
    .bind(score.dsr)
    .bind(score.pbo)
    .bind(score.oos_expectancy_cost_aware)
    .bind(score.profit_factor)
    .bind(score.cvar5)
    .bind(score.mar)
    .bind(regime_contrib)
    .bind(score.n_regimes_positive)
    .bind(score.passed_gate)
    .execute(store.pool())
    .await
    .map_err(store_err)?;
    Ok(())
}

/// Read back a strategy (genome + status + generation + parent) by id.
pub async fn load_strategy(store: &Store, id: StrategyId) -> Result<Option<Strategy>> {
    let row: Option<(serde_json::Value, String, i32, Option<uuid::Uuid>)> =
        se_store::sqlx::query_as(
            "SELECT genome, status, generation, parent_id FROM strategies WHERE strategy_id = $1",
        )
        .bind(id.inner())
        .fetch_optional(store.pool())
        .await
        .map_err(store_err)?;

    let Some((genome_json, status, generation, parent)) = row else {
        return Ok(None);
    };
    let genome: Genome = serde_json::from_value(genome_json).map_err(store_err)?;
    Ok(Some(Strategy {
        id,
        genome,
        status: parse_status(&status),
        generation: generation.max(0) as u32,
        parent: parent.map(StrategyId::from_uuid),
    }))
}

/// All promoted strategies for a (horizon, scanner), newest first — used for signal generation.
pub async fn load_promoted(
    store: &Store,
    horizon: &str,
    scanner: Scanner,
) -> Result<Vec<Strategy>> {
    let rows: Vec<(uuid::Uuid, serde_json::Value, i32, Option<uuid::Uuid>)> =
        se_store::sqlx::query_as(
            "SELECT strategy_id, genome, generation, parent_id FROM strategies \
         WHERE status = 'promoted' AND horizon = $1 AND scanner = $2 ORDER BY updated_at DESC",
        )
        .bind(horizon)
        .bind(scanner.as_str())
        .fetch_all(store.pool())
        .await
        .map_err(store_err)?;

    let mut out = Vec::new();
    for (id, genome_json, generation, parent) in rows {
        let genome: Genome = serde_json::from_value(genome_json).map_err(store_err)?;
        out.push(Strategy {
            id: StrategyId::from_uuid(id),
            genome,
            status: StrategyStatus::Promoted,
            generation: generation.max(0) as u32,
            parent: parent.map(StrategyId::from_uuid),
        });
    }
    Ok(out)
}

/// The latest OOS score for a strategy (for signal cohort stats), if any.
pub async fn latest_oos_score(store: &Store, id: StrategyId) -> Result<Option<StoredOosScore>> {
    type Row = (
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        serde_json::Value,
        i32,
        bool,
        serde_json::Value,
    );
    let row: Option<Row> = se_store::sqlx::query_as(
        "SELECT dsr, pbo, oos_expectancy_cost_aware, profit_factor, cvar5, mar, \
                regime_contrib, n_regimes_positive, passed_gate, fold_spec \
         FROM oos_scores WHERE strategy_id = $1 ORDER BY evaluated_at DESC LIMIT 1",
    )
    .bind(id.inner())
    .fetch_optional(store.pool())
    .await
    .map_err(store_err)?;

    Ok(row.map(|r| {
        let n_entries =
            r.9.get("n_entries")
                .and_then(|v| v.as_u64())
                .map(|n| n as u32)
                .unwrap_or(0);
        StoredOosScore {
            dsr: r.0,
            pbo: r.1,
            oos_expectancy_cost_aware: r.2,
            profit_factor: r.3,
            cvar5: r.4,
            mar: r.5,
            regime_contrib: r.6,
            n_regimes_positive: r.7,
            passed_gate: r.8,
            n_entries,
        }
    }))
}

/// A row read back from `oos_scores` (the persisted OOS metrics).
#[derive(Debug, Clone)]
pub struct StoredOosScore {
    pub dsr: Option<f64>,
    pub pbo: Option<f64>,
    pub oos_expectancy_cost_aware: Option<f64>,
    pub profit_factor: Option<f64>,
    pub cvar5: Option<f64>,
    pub mar: Option<f64>,
    pub regime_contrib: serde_json::Value,
    pub n_regimes_positive: i32,
    pub passed_gate: bool,
    /// Cohort size (entry count) recovered from the stored `fold_spec` JSON.
    pub n_entries: u32,
}

fn parse_status(s: &str) -> StrategyStatus {
    match s {
        "promoted" => StrategyStatus::Promoted,
        "quarantined" => StrategyStatus::Quarantined,
        "demoted" => StrategyStatus::Demoted,
        "retired" => StrategyStatus::Retired,
        _ => StrategyStatus::Candidate,
    }
}
