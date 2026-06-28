//! Row mappings between the database and `se-core` domain types.

use chrono::{DateTime, Utc};
use se_core::{AsOf, Bar, DecisionTs, Feature, Layer, LeadTimeTag, Ticker};

/// A market-wide macro/cross-asset/vol observation to persist into the PIT macro
/// store, with full provenance. Not per-ticker — keyed by `(series, ts, as_of)`.
#[derive(Debug, Clone)]
pub struct MacroWrite {
    /// Stable series key, e.g. `vix`, `ust10y`, `hy_oas` (matches `MacroSeries::as_str`).
    pub series: String,
    /// Reference date of the observation (event time).
    pub ts: DateTime<Utc>,
    /// When the value became knowable (knowledge time). For lagged series, later than `ts`.
    pub as_of: DateTime<Utc>,
    pub value: f64,
    pub lead_time: LeadTimeTag,
    pub source: String,
}

/// A bar row as stored. `cadence` distinguishes daily vs intraday series.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct BarRow {
    pub ticker: String,
    pub ts: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl BarRow {
    /// Convert to a domain [`Bar`]. Returns `None` if the ticker is outside the universe.
    pub fn to_bar(&self) -> Option<Bar> {
        let ticker: Ticker = self.ticker.parse().ok()?;
        Some(Bar {
            ticker,
            ts: self.ts,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
        })
    }
}

/// A feature value to persist into the PIT store, with full provenance.
#[derive(Debug, Clone)]
pub struct FeatureWrite {
    pub ticker: Ticker,
    pub feature_key: String,
    pub layer: Layer,
    /// The bar the value is "for".
    pub decision_ts: DecisionTs,
    /// When the value became knowable.
    pub as_of: AsOf,
    pub value: f64,
    pub lead_time: LeadTimeTag,
    pub source: String,
}

impl FeatureWrite {
    /// Build from a computed [`Feature`] plus the decision bar it belongs to.
    pub fn from_feature(ticker: Ticker, decision_ts: DecisionTs, f: &Feature) -> Self {
        FeatureWrite {
            ticker,
            feature_key: f.key.clone(),
            layer: f.layer,
            decision_ts,
            as_of: f.as_of,
            value: f.value,
            lead_time: f.lead_time,
            source: f.source.clone(),
        }
    }
}

/// Internal read row for the PIT feature query.
#[derive(Debug, Clone, sqlx::FromRow)]
pub(crate) struct FeatureRow {
    pub feature_key: String,
    pub value: f64,
    pub layer: String,
    pub as_of: DateTime<Utc>,
    pub lead_time: String,
    pub source: String,
}

impl FeatureRow {
    pub(crate) fn to_feature(&self) -> Option<Feature> {
        let layer: Layer = self.layer.parse().ok()?;
        Some(Feature {
            key: self.feature_key.clone(),
            value: self.value,
            layer,
            as_of: AsOf::new(self.as_of),
            lead_time: LeadTimeTag::parse_tag(&self.lead_time),
            source: self.source.clone(),
        })
    }
}

/// Internal read row for the PIT macro history query.
#[derive(Debug, Clone, sqlx::FromRow)]
pub(crate) struct MacroHistoryRow {
    pub ts: DateTime<Utc>,
    pub value: f64,
}
