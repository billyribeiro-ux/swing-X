//! Per-bar feature matrix construction.
//!
//! The search needs, for one `(ticker, window)`, a series of decision bars each carrying a
//! PIT-safe feature map (all layers), plus the bar series and the per-bar ATR that sizes the
//! triple-barrier labels. We compute features ON THE FLY by running the layer modules through
//! the leakage-safe [`FeatureContext`] at each bar — exactly the path `se-regime`/the CLI use —
//! rather than relying on whatever happens to be persisted in `features_pit`. That makes the
//! backtest reproducible and correct even when only the latest bar was scanned.
//!
//! Every feature read filters `as_of <= decision_ts`, so no value can leak from the future.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use futures::StreamExt;
use se_core::{Bar, DecisionTs, HorizonProfile, Result, Ticker};
use se_features::indicators::atr;
use se_features::{
    EventOverlay, FeatureContext, FeatureModule, LocationModule, RegimeModule, TradeabilityModule,
    TriggerModule,
};
use se_provider::NullProprietary;
use se_store::Store;

/// One decision bar with everything the backtest needs at that point in time.
#[derive(Debug, Clone)]
pub struct BarPoint {
    /// The decision bar.
    pub bar: Bar,
    /// Position of `bar` within [`FeatureWindow::bars`] (so the labeler can walk forward).
    pub idx: usize,
    /// PIT-safe feature map (dotted keys, e.g. `regime.adx`), as known at this bar.
    pub features: BTreeMap<String, f64>,
    /// Wilder ATR over the profile's lookback at this bar (`None` if insufficient history).
    pub atr: Option<f64>,
}

/// The materialized window for a ticker: the full bar series plus per-bar points.
#[derive(Debug, Clone)]
pub struct FeatureWindow {
    pub ticker: Ticker,
    /// All bars in `[from, to]`, chronological (oldest first). The labeler walks these.
    pub bars: Vec<Bar>,
    /// One [`BarPoint`] per decision bar we could compute features for, chronological.
    pub points: Vec<BarPoint>,
}

impl FeatureWindow {
    /// All distinct feature keys observed across the window (dotted form).
    pub fn feature_keys(&self) -> Vec<String> {
        let mut set = std::collections::BTreeSet::new();
        for p in &self.points {
            for k in p.features.keys() {
                set.insert(k.clone());
            }
        }
        set.into_iter().collect()
    }
}

/// Build a [`FeatureWindow`] for `ticker` over `[from, to]` under `profile`.
///
/// `warmup_bars` decision bars at the start are still labeled but produce thinner feature
/// maps (some indicators need history); we keep them and let `genome.fires` decide. Feature
/// computation runs every layer module at each bar through the PIT-safe context.
pub async fn build_window(
    store: &Store,
    ticker: Ticker,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    profile: HorizonProfile,
) -> Result<FeatureWindow> {
    let cadence = "daily";

    // The decision bars we want to evaluate are the stored bars within [from, to].
    let bar_ts = stored_bar_ts(store, ticker, from, to, cadence).await?;
    if bar_ts.is_empty() {
        return Ok(FeatureWindow {
            ticker,
            bars: Vec::new(),
            points: Vec::new(),
        });
    }

    // Pull the full bar series up to the LAST decision bar (PIT-safe at that cutoff). This is
    // the series the labeler walks; entries index into it.
    let last_ts = *bar_ts.last().unwrap();
    let pit_last = store.pit(ticker, DecisionTs::new(last_ts));
    // Generous lookback: everything from `from` (minus warmup) to `to`. We over-fetch and the
    // PIT filter trims to `ts <= last_ts`.
    let bars = pit_last.bars(cadence, 4000).await?;
    if bars.is_empty() {
        return Ok(FeatureWindow {
            ticker,
            bars: Vec::new(),
            points: Vec::new(),
        });
    }

    // Index bars by timestamp so each decision bar maps to a position in `bars`.
    let pos_of: BTreeMap<DateTime<Utc>, usize> =
        bars.iter().enumerate().map(|(i, b)| (b.ts, i)).collect();

    let atr_lookback = profile.atr_lookback as usize;

    // Per-bar feature computation is independent and I/O-bound (each bar issues its own PIT
    // queries against the pool). Run with bounded concurrency so window materialization is not
    // serialized on DB round-trips. The pool caps real parallelism; `CONCURRENCY` just keeps it
    // saturated without exhausting connections.
    const CONCURRENCY: usize = 8;
    let idxs: Vec<(DateTime<Utc>, usize)> = bar_ts
        .into_iter()
        .filter_map(|ts| pos_of.get(&ts).map(|&idx| (ts, idx)))
        .collect();

    let bars_ref = &bars;
    let mut points: Vec<BarPoint> = futures::stream::iter(idxs)
        .map(|(ts, idx)| async move {
            let prop = NullProprietary;
            let modules: [&dyn FeatureModule; 5] = [
                &TradeabilityModule::default(),
                &RegimeModule,
                &LocationModule::new(),
                &TriggerModule::new(),
                &EventOverlay::new(),
            ];
            let decision = DecisionTs::new(ts);
            let pit = store.pit(ticker, decision);
            let ctx = FeatureContext::new(&pit, &prop, profile);

            let mut features: BTreeMap<String, f64> = BTreeMap::new();
            for m in modules {
                let feats = m.compute(&ctx).await?;
                for f in feats {
                    if f.value.is_finite() {
                        features.insert(f.key, f.value);
                    }
                }
            }
            let atr_val =
                atr(&bars_ref[..=idx], atr_lookback).filter(|a| *a > 0.0 && a.is_finite());
            Ok::<BarPoint, se_core::Error>(BarPoint {
                bar: bars_ref[idx],
                idx,
                features,
                atr: atr_val,
            })
        })
        .buffer_unordered(CONCURRENCY)
        .collect::<Vec<Result<BarPoint>>>()
        .await
        .into_iter()
        .collect::<Result<Vec<BarPoint>>>()?;

    // `buffer_unordered` does not preserve order; restore chronological order by bar index.
    points.sort_by_key(|p| p.idx);

    Ok(FeatureWindow {
        ticker,
        bars,
        points,
    })
}

/// Stored decision-bar timestamps for `ticker` in `[from, to]`, ascending. Uses the raw bars
/// table (the window endpoints ARE the decision bars); each is read PIT-safely when evaluated.
async fn stored_bar_ts(
    store: &Store,
    ticker: Ticker,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    cadence: &str,
) -> Result<Vec<DateTime<Utc>>> {
    let rows: Vec<(DateTime<Utc>,)> = se_store::sqlx::query_as(
        "SELECT ts FROM bars \
         WHERE ticker = $1 AND cadence = $2 AND ts >= $3 AND ts <= $4 \
         ORDER BY ts ASC",
    )
    .bind(ticker.as_str())
    .bind(cadence)
    .bind(from)
    .bind(to)
    .fetch_all(store.pool())
    .await
    .map_err(|e| se_core::Error::Store(e.to_string()))?;
    Ok(rows.into_iter().map(|r| r.0).collect())
}

/// Convert a dotted PIT feature key (`regime.adx`) into the `layer__feature` column name the
/// ML worker's Parquet schema expects (`regime__adx`). Keys without a `.` are passed through.
pub fn dotted_to_column(key: &str) -> String {
    match key.split_once('.') {
        Some((layer, rest)) => format!("{layer}__{}", rest.replace('.', "_")),
        None => key.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_name_conversion() {
        assert_eq!(dotted_to_column("regime.adx"), "regime__adx");
        assert_eq!(
            dotted_to_column("location.anchored_vwap_dist"),
            "location__anchored_vwap_dist"
        );
        assert_eq!(dotted_to_column("plain"), "plain");
    }
}
