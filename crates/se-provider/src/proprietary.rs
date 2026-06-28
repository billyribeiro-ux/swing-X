//! `PROPRIETARY_FEATURE` data hooks. The operator injects private feeds here
//! (dealer GEX/charm/vanna/walls, tick order-flow). Until configured, every hook
//! returns [`Error::FeatureUnavailable`] — the system NEVER fabricates these.
//!
//! Whatever a hook returns must pass the SAME PIT + validation gates as native
//! features; there is no privileged path. In v1 the default is [`NullProprietary`].

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use se_core::{DecisionTs, Error, Result, Ticker};
use serde::{Deserialize, Serialize};

/// A dealer-positioning snapshot a proprietary feed could supply.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GexSnapshot {
    pub ticker: Ticker,
    pub decision_ts: DateTime<Utc>,
    pub net_gex: f64,
    pub gamma_flip: Option<f64>,
    pub call_wall: Option<f64>,
    pub put_wall: Option<f64>,
    pub abs_gex: f64,
    pub as_of: DateTime<Utc>,
}

#[async_trait]
pub trait ProprietaryProvider: Send + Sync {
    fn name(&self) -> &str;

    /// Dealer gamma exposure snapshot. Default: unavailable (stubbed in v1).
    async fn gex(&self, _ticker: Ticker, _at: DecisionTs) -> Result<Option<GexSnapshot>> {
        Err(Error::FeatureUnavailable("proprietary:gex".into()))
    }

    /// Aggregated signed order-flow imbalance over a recent window in [-1, 1].
    async fn order_flow_imbalance(&self, _ticker: Ticker, _at: DecisionTs) -> Result<Option<f64>> {
        Err(Error::FeatureUnavailable("proprietary:order_flow".into()))
    }

    /// Dark-pool index (DIX). Default: unavailable.
    async fn dix(&self, _ticker: Ticker, _at: DecisionTs) -> Result<Option<f64>> {
        Err(Error::FeatureUnavailable("proprietary:dix".into()))
    }
}

/// The v1 default: every proprietary signal is unavailable.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullProprietary;

#[async_trait]
impl ProprietaryProvider for NullProprietary {
    fn name(&self) -> &str {
        "null-proprietary"
    }
}
