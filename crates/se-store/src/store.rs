//! The database handle: connection pool, migrations, and write paths.
//! Reads for decision-making go exclusively through [`PitContext`].

use chrono::{DateTime, Utc};
use se_core::{Bar, DecisionTs, Error, RegimeLabel, Result, Ticker};
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::QueryBuilder;
use uuid::Uuid;

use crate::models::FeatureWrite;
use crate::pit::PitContext;

fn store_err(e: impl std::fmt::Display) -> Error {
    Error::Store(e.to_string())
}

/// Owns the Postgres pool. Cheap to clone (the pool is `Arc`-backed).
#[derive(Clone)]
pub struct Store {
    pool: PgPool,
}

impl Store {
    /// Connect (eagerly) to `database_url` and build a pool.
    pub async fn connect(database_url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await
            .map_err(store_err)?;
        Ok(Store { pool })
    }

    /// Build a pool without opening a connection up front (connects on first use).
    pub fn connect_lazy(database_url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect_lazy(database_url)
            .map_err(store_err)?;
        Ok(Store { pool })
    }

    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Run the embedded migrations. Idempotent.
    pub async fn migrate(&self) -> Result<()> {
        sqlx::migrate!("../../migrations")
            .run(&self.pool)
            .await
            .map_err(store_err)?;
        Ok(())
    }

    pub async fn ping(&self) -> Result<()> {
        sqlx::query("SELECT 1")
            .execute(&self.pool)
            .await
            .map_err(store_err)?;
        Ok(())
    }

    /// A leakage-safe read handle bound to `(ticker, decision_ts)`.
    pub fn pit(&self, ticker: Ticker, decision_ts: DecisionTs) -> PitContext<'_> {
        PitContext::new(&self.pool, ticker, decision_ts)
    }

    // ---- writes -----------------------------------------------------------

    /// Upsert a batch of bars (idempotent on `(ticker, cadence, ts)`).
    ///
    /// Each bar's `as_of` is its own close timestamp — a bar is knowable at its
    /// close, which is exactly what the PIT read predicate relies on.
    pub async fn upsert_bars(&self, bars: &[Bar], cadence: &str, source: &str) -> Result<u64> {
        if bars.is_empty() {
            return Ok(0);
        }
        let mut total = 0u64;
        // Chunk to stay well under the Postgres parameter limit (~65535 / 10 cols).
        for chunk in bars.chunks(2000) {
            let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
                "INSERT INTO bars (ticker, cadence, ts, open, high, low, close, volume, source, as_of) ",
            );
            qb.push_values(chunk, |mut b, bar| {
                b.push_bind(bar.ticker.as_str())
                    .push_bind(cadence)
                    .push_bind(bar.ts)
                    .push_bind(bar.open)
                    .push_bind(bar.high)
                    .push_bind(bar.low)
                    .push_bind(bar.close)
                    .push_bind(bar.volume)
                    .push_bind(source)
                    .push_bind(bar.ts);
            });
            qb.push(
                " ON CONFLICT (ticker, cadence, ts) DO UPDATE SET \
                 open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, \
                 close = EXCLUDED.close, volume = EXCLUDED.volume, \
                 source = EXCLUDED.source, as_of = EXCLUDED.as_of",
            );
            let res = qb.build().execute(&self.pool).await.map_err(store_err)?;
            total += res.rows_affected();
        }
        Ok(total)
    }

    /// Insert/upsert PIT feature values (idempotent on the bitemporal key).
    pub async fn insert_features(&self, feats: &[FeatureWrite]) -> Result<u64> {
        if feats.is_empty() {
            return Ok(0);
        }
        let mut total = 0u64;
        for chunk in feats.chunks(2000) {
            let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
                "INSERT INTO features_pit \
                 (ticker, feature_key, layer, decision_ts, as_of, value, lead_time, source) ",
            );
            qb.push_values(chunk, |mut b, f| {
                b.push_bind(f.ticker.as_str())
                    .push_bind(f.feature_key.as_str())
                    .push_bind(f.layer.as_str())
                    .push_bind(f.decision_ts.inner())
                    .push_bind(f.as_of.inner())
                    .push_bind(f.value)
                    .push_bind(f.lead_time.to_tag_string())
                    .push_bind(f.source.as_str());
            });
            qb.push(
                " ON CONFLICT (ticker, feature_key, decision_ts, as_of) DO UPDATE SET \
                 value = EXCLUDED.value, layer = EXCLUDED.layer, \
                 lead_time = EXCLUDED.lead_time, source = EXCLUDED.source",
            );
            let res = qb.build().execute(&self.pool).await.map_err(store_err)?;
            total += res.rows_affected();
        }
        Ok(total)
    }

    /// Persist a regime assignment.
    pub async fn insert_regime(
        &self,
        ticker: Ticker,
        decision_ts: DecisionTs,
        as_of: DateTime<Utc>,
        label: RegimeLabel,
        prob_map: &serde_json::Value,
        model_id: Option<Uuid>,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO regimes (ticker, decision_ts, as_of, regime_label, prob_map, model_id) \
             VALUES ($1,$2,$3,$4,$5,$6) \
             ON CONFLICT (ticker, decision_ts, as_of) \
             DO UPDATE SET regime_label = EXCLUDED.regime_label, prob_map = EXCLUDED.prob_map, model_id = EXCLUDED.model_id",
        )
        .bind(ticker.as_str())
        .bind(decision_ts.inner())
        .bind(as_of)
        .bind(label.as_str())
        .bind(prob_map)
        .bind(model_id)
        .execute(&self.pool)
        .await
        .map_err(store_err)?;
        Ok(())
    }

    /// Count distinct feature keys stored for a ticker (diagnostics / tests).
    pub async fn feature_key_count(&self, ticker: Ticker) -> Result<i64> {
        let row: (i64,) = sqlx::query_as(
            "SELECT COUNT(DISTINCT feature_key) FROM features_pit WHERE ticker = $1",
        )
        .bind(ticker.as_str())
        .fetch_one(&self.pool)
        .await
        .map_err(store_err)?;
        Ok(row.0)
    }

    /// Delete feature rows by provenance source (used to clean up test data).
    pub async fn delete_features_by_source(&self, source: &str) -> Result<u64> {
        let res = sqlx::query("DELETE FROM features_pit WHERE source = $1")
            .bind(source)
            .execute(&self.pool)
            .await
            .map_err(store_err)?;
        Ok(res.rows_affected())
    }
}
