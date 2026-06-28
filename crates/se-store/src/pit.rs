//! [`PitContext`] — the leakage-safe read handle.
//!
//! Bound to a single `(ticker, decision_ts)`, every query it issues hard-codes the
//! predicate `as_of <= decision_ts`. There is deliberately NO method that accepts a
//! caller-supplied knowledge-time filter, so no feature can be read "from the future".
//! This is the data-layer half of the system's leakage prevention (the other half is
//! purge + embargo in cross-validation, on the Python side).

use chrono::{DateTime, Utc};
use se_core::{Bar, DecisionTs, Error, Feature, Layer, Result, Ticker};
use sqlx::postgres::PgPool;

use crate::models::{BarRow, FeatureRow, MacroHistoryRow};

fn store_err(e: impl std::fmt::Display) -> Error {
    Error::Store(e.to_string())
}

pub struct PitContext<'a> {
    pool: &'a PgPool,
    ticker: Ticker,
    decision_ts: DecisionTs,
}

impl<'a> PitContext<'a> {
    pub(crate) fn new(pool: &'a PgPool, ticker: Ticker, decision_ts: DecisionTs) -> Self {
        PitContext {
            pool,
            ticker,
            decision_ts,
        }
    }

    pub fn ticker(&self) -> Ticker {
        self.ticker
    }

    pub fn decision_ts(&self) -> DecisionTs {
        self.decision_ts
    }

    fn cutoff(&self) -> DateTime<Utc> {
        self.decision_ts.inner()
    }

    /// All features knowable at the decision bar — the latest version of each
    /// `feature_key` whose `decision_ts <= cutoff` AND `as_of <= cutoff`.
    /// Optionally restrict to a single [`Layer`].
    pub async fn features(&self, layer: Option<Layer>) -> Result<Vec<Feature>> {
        let layer_filter: Option<&str> = layer.map(|l| l.as_str());
        let rows: Vec<FeatureRow> = sqlx::query_as::<_, FeatureRow>(
            "SELECT DISTINCT ON (feature_key) \
                feature_key, value, layer, as_of, lead_time, source \
             FROM features_pit \
             WHERE ticker = $1 \
               AND decision_ts <= $2 \
               AND as_of <= $2 \
               AND ($3::text IS NULL OR layer = $3) \
             ORDER BY feature_key, decision_ts DESC, as_of DESC",
        )
        .bind(self.ticker.as_str())
        .bind(self.cutoff())
        .bind(layer_filter)
        .fetch_all(self.pool)
        .await
        .map_err(store_err)?;

        Ok(rows.iter().filter_map(|r| r.to_feature()).collect())
    }

    /// A single feature by key, as known at the decision bar.
    pub async fn feature(&self, key: &str) -> Result<Option<Feature>> {
        let row: Option<FeatureRow> = sqlx::query_as::<_, FeatureRow>(
            "SELECT feature_key, value, layer, as_of, lead_time, source \
             FROM features_pit \
             WHERE ticker = $1 AND feature_key = $2 \
               AND decision_ts <= $3 AND as_of <= $3 \
             ORDER BY decision_ts DESC, as_of DESC \
             LIMIT 1",
        )
        .bind(self.ticker.as_str())
        .bind(key)
        .bind(self.cutoff())
        .fetch_optional(self.pool)
        .await
        .map_err(store_err)?;

        Ok(row.and_then(|r| r.to_feature()))
    }

    /// Convenience: just the numeric value of a feature, if present.
    pub async fn feature_value(&self, key: &str) -> Result<Option<f64>> {
        Ok(self.feature(key).await?.map(|f| f.value))
    }

    /// The most recent `lookback` bars with `ts <= cutoff` AND `as_of <= cutoff`,
    /// returned in chronological (oldest-first) order.
    pub async fn bars(&self, cadence: &str, lookback: i64) -> Result<Vec<Bar>> {
        let rows: Vec<BarRow> = sqlx::query_as::<_, BarRow>(
            "SELECT ticker, ts, open, high, low, close, volume \
             FROM bars \
             WHERE ticker = $1 AND cadence = $2 \
               AND ts <= $3 AND as_of <= $3 \
             ORDER BY ts DESC \
             LIMIT $4",
        )
        .bind(self.ticker.as_str())
        .bind(cadence)
        .bind(self.cutoff())
        .bind(lookback.max(0))
        .fetch_all(self.pool)
        .await
        .map_err(store_err)?;

        // DB gave newest-first; reverse to chronological for feature math.
        Ok(rows.iter().rev().filter_map(|r| r.to_bar()).collect())
    }

    /// The single most recent observable bar.
    pub async fn latest_bar(&self, cadence: &str) -> Result<Option<Bar>> {
        Ok(self.bars(cadence, 1).await?.into_iter().next_back())
    }

    /// Same as [`bars`](Self::bars) but for a DIFFERENT `ticker` than the one this
    /// context is bound to — needed for cross-ticker features (relative strength
    /// vs SPY, universe breadth). The leakage predicate is identical: only bars
    /// with `ts <= cutoff AND as_of <= cutoff` are returned, where `cutoff` is
    /// this context's `decision_ts`. Chronological (oldest-first) order.
    pub async fn bars_for(&self, other: Ticker, cadence: &str, lookback: i64) -> Result<Vec<Bar>> {
        let rows: Vec<BarRow> = sqlx::query_as::<_, BarRow>(
            "SELECT ticker, ts, open, high, low, close, volume \
             FROM bars \
             WHERE ticker = $1 AND cadence = $2 \
               AND ts <= $3 AND as_of <= $3 \
             ORDER BY ts DESC \
             LIMIT $4",
        )
        .bind(other.as_str())
        .bind(cadence)
        .bind(self.cutoff())
        .bind(lookback.max(0))
        .fetch_all(self.pool)
        .await
        .map_err(store_err)?;

        Ok(rows.iter().rev().filter_map(|r| r.to_bar()).collect())
    }

    // ---- macro (market-wide) PIT reads -----------------------------------
    //
    // The macro store is NOT per-ticker: these methods ignore `self.ticker` and
    // use only `self.decision_ts` as the cutoff. The leakage predicate is the
    // same as for features/bars — a value is visible only when its reference date
    // AND its knowledge time are both on or before the decision bar:
    //   `ts <= cutoff AND as_of <= cutoff`.

    /// Latest value of a macro `series` knowable at the decision bar — the most
    /// recent observation with `ts <= cutoff AND as_of <= cutoff`, picking the
    /// newest vintage (`as_of`) when several exist for the same `ts`.
    pub async fn macro_value_as_of(&self, series: &str) -> Result<Option<f64>> {
        let row: Option<(f64,)> = sqlx::query_as(
            "SELECT value \
             FROM macro_series_pit \
             WHERE series = $1 AND ts <= $2 AND as_of <= $2 \
             ORDER BY ts DESC, as_of DESC \
             LIMIT 1",
        )
        .bind(series)
        .bind(self.cutoff())
        .fetch_optional(self.pool)
        .await
        .map_err(store_err)?;
        Ok(row.map(|r| r.0))
    }

    /// Chronological (oldest-first) history of a macro `series`: the latest
    /// knowable value per reference date `ts`, for the most recent `lookback`
    /// dates with `ts <= cutoff AND as_of <= cutoff`.
    pub async fn macro_history(
        &self,
        series: &str,
        lookback: i64,
    ) -> Result<Vec<(DateTime<Utc>, f64)>> {
        let rows: Vec<MacroHistoryRow> = sqlx::query_as::<_, MacroHistoryRow>(
            "SELECT ts, value FROM ( \
                 SELECT DISTINCT ON (ts) ts, value \
                 FROM macro_series_pit \
                 WHERE series = $1 AND ts <= $2 AND as_of <= $2 \
                 ORDER BY ts DESC, as_of DESC \
                 LIMIT $3 \
             ) sub \
             ORDER BY ts ASC",
        )
        .bind(series)
        .bind(self.cutoff())
        .bind(lookback.max(0))
        .fetch_all(self.pool)
        .await
        .map_err(store_err)?;
        Ok(rows.into_iter().map(|r| (r.ts, r.value)).collect())
    }
}
