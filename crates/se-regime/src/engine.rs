//! [`RegimeEngine`] — walks the stored bars of a ticker over a window, computes
//! the Layer-1 regime features through the PIT-safe [`RegimeModule`], persists
//! them, runs the [`RegimeClassifier`], and stores the resulting regime label.
//!
//! It is provider-INDEPENDENT: it reads bars + macro exclusively from the store
//! via [`PitContext`], so labeling is reproducible and leakage-safe (every read
//! filters `as_of <= decision_ts`). Ingestion (pulling from FMP/FRED) is the
//! CLI's job; the engine only consumes what is already persisted.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use se_core::{DecisionTs, Result, Ticker};
use se_features::{FeatureContext, FeatureModule, RegimeModule};
use se_provider::NullProprietary;
use se_store::{FeatureWrite, Store};

use crate::classifier::{RegimeAssessment, RegimeClassifier};

/// Drives regime feature assembly + classification + storage over a window.
#[derive(Debug, Clone, Default)]
pub struct RegimeEngine {
    pub classifier: RegimeClassifier,
}

impl RegimeEngine {
    pub fn new(classifier: RegimeClassifier) -> Self {
        RegimeEngine { classifier }
    }

    /// Compute + classify the regime at a single decision bar, persisting the
    /// regime features. Returns the assessment (or `None` if there is no bar /
    /// not enough data to produce any feature at all).
    pub async fn assess_at(
        &self,
        store: &Store,
        ticker: Ticker,
        decision_ts: DecisionTs,
    ) -> Result<Option<RegimeAssessment>> {
        let prop = NullProprietary;
        let module = RegimeModule;
        let profile = se_core::HorizonProfile::swing();

        let pit = store.pit(ticker, decision_ts);
        let ctx = FeatureContext::new(&pit, &prop, profile);
        let feats = module.compute(&ctx).await?;
        if feats.is_empty() {
            return Ok(None);
        }

        // Persist the regime features (exercises the full PIT pipeline).
        let writes: Vec<FeatureWrite> = feats
            .iter()
            .map(|f| FeatureWrite::from_feature(ticker, decision_ts, f))
            .collect();
        store.insert_features(&writes).await?;

        // Build the classifier input from the just-computed features.
        let map: BTreeMap<String, f64> = feats.iter().map(|f| (f.key.clone(), f.value)).collect();
        let assessment = self.classifier.classify(&map);

        store
            .insert_regime(
                ticker,
                decision_ts,
                decision_ts.inner(),
                assessment.label,
                &assessment.prob_map_json(),
                None,
            )
            .await?;

        Ok(Some(assessment))
    }

    /// Label every stored daily bar of `ticker` whose `ts` is within `[from, to]`.
    /// Returns the per-bar assessments in chronological order.
    pub async fn label_window(
        &self,
        store: &Store,
        ticker: Ticker,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<(DateTime<Utc>, RegimeAssessment)>> {
        let bars = stored_bar_ts(store, ticker, from, to).await?;
        let mut out = Vec::with_capacity(bars.len());
        for ts in bars {
            let decision = DecisionTs::new(ts);
            if let Some(a) = self.assess_at(store, ticker, decision).await? {
                out.push((ts, a));
            }
        }
        Ok(out)
    }

    /// Read back a previously stored regime label at `decision_ts` (if any).
    /// Does not recompute; returns the persisted label + prob map.
    pub async fn regime_at(
        &self,
        store: &Store,
        ticker: Ticker,
        decision_ts: DecisionTs,
    ) -> Result<Option<StoredRegime>> {
        stored_regime(store, ticker, decision_ts).await
    }
}

/// A regime row read back from storage.
#[derive(Debug, Clone)]
pub struct StoredRegime {
    pub label: se_core::RegimeLabel,
    pub prob_map: serde_json::Value,
}

/// All stored daily-bar timestamps for `ticker` within `[from, to]`, ascending.
/// Uses the raw bars table (not PIT-filtered) because the window endpoints ARE the
/// decision bars we want to label — each is then read PIT-safely when assessed.
async fn stored_bar_ts(
    store: &Store,
    ticker: Ticker,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> Result<Vec<DateTime<Utc>>> {
    let rows: Vec<(DateTime<Utc>,)> = se_store::sqlx::query_as(
        "SELECT ts FROM bars \
         WHERE ticker = $1 AND cadence = 'daily' AND ts >= $2 AND ts <= $3 \
         ORDER BY ts ASC",
    )
    .bind(ticker.as_str())
    .bind(from)
    .bind(to)
    .fetch_all(store.pool())
    .await
    .map_err(|e| se_core::Error::Store(e.to_string()))?;
    Ok(rows.into_iter().map(|r| r.0).collect())
}

async fn stored_regime(
    store: &Store,
    ticker: Ticker,
    decision_ts: DecisionTs,
) -> Result<Option<StoredRegime>> {
    let row: Option<(String, serde_json::Value)> = se_store::sqlx::query_as(
        "SELECT regime_label, prob_map FROM regimes \
         WHERE ticker = $1 AND decision_ts = $2 \
         ORDER BY as_of DESC LIMIT 1",
    )
    .bind(ticker.as_str())
    .bind(decision_ts.inner())
    .fetch_optional(store.pool())
    .await
    .map_err(|e| se_core::Error::Store(e.to_string()))?;

    Ok(row.and_then(|(label, prob_map)| {
        label
            .parse()
            .ok()
            .map(|label| StoredRegime { label, prob_map })
    }))
}
