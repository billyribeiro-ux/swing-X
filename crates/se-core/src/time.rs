//! Point-in-time timestamps and the price-bar primitive.
//!
//! [`AsOf`] (when a value became *knowable*) and [`DecisionTs`] (the bar at which
//! a decision is made) are distinct newtypes so the leakage rule —
//! *a value at decision bar T may only use data with `as_of <= T`* — is checkable
//! at the type level via [`AsOf::is_observable_at`].

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::Ticker;

/// Timestamp at which a value/feature first became observable to the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AsOf(pub DateTime<Utc>);

/// The decision bar's timestamp. A model deciding at `DecisionTs` may read only
/// values whose [`AsOf`] is `<=` this instant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DecisionTs(pub DateTime<Utc>);

impl AsOf {
    pub fn new(ts: DateTime<Utc>) -> Self {
        AsOf(ts)
    }
    pub fn inner(self) -> DateTime<Utc> {
        self.0
    }
    /// The PIT predicate, at the type level: is this value knowable at `decision`?
    pub fn is_observable_at(self, decision: DecisionTs) -> bool {
        self.0 <= decision.0
    }
}

impl DecisionTs {
    pub fn new(ts: DateTime<Utc>) -> Self {
        DecisionTs(ts)
    }
    pub fn inner(self) -> DateTime<Utc> {
        self.0
    }
}

impl From<DateTime<Utc>> for AsOf {
    fn from(d: DateTime<Utc>) -> Self {
        AsOf(d)
    }
}
impl From<DateTime<Utc>> for DecisionTs {
    fn from(d: DateTime<Utc>) -> Self {
        DecisionTs(d)
    }
}

/// How far behind real time a value's reference period sits — provenance metadata
/// used to derive the correct [`AsOf`] for delayed series (e.g. macro releases).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeadTimeTag {
    /// Streaming/real-time during the session.
    Realtime,
    /// Known at the official close of the session.
    EndOfDay,
    /// Not actionable until the next session's open.
    NextOpen,
    /// Published with a fixed lag in calendar days (e.g. weekly Fed data).
    LaggedDays(i64),
}

impl LeadTimeTag {
    /// Compact string form for DB storage / wire (`realtime`, `eod`, `next_open`, `lagged:7`).
    pub fn to_tag_string(self) -> String {
        match self {
            LeadTimeTag::Realtime => "realtime".to_string(),
            LeadTimeTag::EndOfDay => "eod".to_string(),
            LeadTimeTag::NextOpen => "next_open".to_string(),
            LeadTimeTag::LaggedDays(n) => format!("lagged:{n}"),
        }
    }

    /// Parse the compact string form; unknown inputs fall back to [`LeadTimeTag::EndOfDay`].
    pub fn parse_tag(s: &str) -> Self {
        match s.trim() {
            "realtime" => LeadTimeTag::Realtime,
            "eod" => LeadTimeTag::EndOfDay,
            "next_open" => LeadTimeTag::NextOpen,
            other => other
                .strip_prefix("lagged:")
                .and_then(|n| n.parse::<i64>().ok())
                .map(LeadTimeTag::LaggedDays)
                .unwrap_or(LeadTimeTag::EndOfDay),
        }
    }
}

/// Trade direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Long,
    Short,
}

impl Side {
    /// +1 for long, -1 for short — for signed PnL/return math.
    pub const fn sign(self) -> f64 {
        match self {
            Side::Long => 1.0,
            Side::Short => -1.0,
        }
    }
}

/// A single OHLCV price bar. Provenance (`source`, `as_of`) for bars is tracked in
/// the store; this struct is the numeric record used by the feature engine.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bar {
    pub ticker: Ticker,
    pub ts: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Bar {
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Classic typical price (H+L+C)/3 — used for VWAP and volume-profile binning.
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Wilder's true range against the previous bar's close.
    pub fn true_range(&self, prev_close: f64) -> f64 {
        let hl = self.high - self.low;
        let hc = (self.high - prev_close).abs();
        let lc = (self.low - prev_close).abs();
        hl.max(hc).max(lc)
    }

    pub fn is_up(&self) -> bool {
        self.close >= self.open
    }

    /// Simple (arithmetic) return over the bar.
    pub fn intrabar_return(&self) -> f64 {
        if self.open == 0.0 {
            0.0
        } else {
            self.close / self.open - 1.0
        }
    }
}
