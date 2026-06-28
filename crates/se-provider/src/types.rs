//! Shared ingestion DTOs and the macro-series catalog.

use chrono::{DateTime, Utc};
use se_core::{LeadTimeTag, Ticker};
use serde::{Deserialize, Serialize};

/// A latest quote for an ETF or an index symbol (e.g. `^VIX`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Quote {
    pub symbol: String,
    pub price: f64,
    pub ts: DateTime<Utc>,
    pub day_change_pct: Option<f64>,
}

/// One observation of a macro/economic/index time series.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MacroPoint {
    pub series: MacroSeries,
    /// Reference date of the observation (event time).
    pub ts: DateTime<Utc>,
    pub value: f64,
    /// When it became knowable (knowledge time). For lagged series this is later than `ts`.
    pub as_of: DateTime<Utc>,
    pub lead_time: LeadTimeTag,
    pub source: String,
}

/// The macro/cross-asset/vol series the regime layer consumes. Each maps to a
/// concrete provider endpoint/symbol in the adapters (FMP or FRED).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MacroSeries {
    // --- volatility complex (FMP index symbols) ---
    Vix,
    Vix9d,
    Vix3m,
    Vvix,
    Skew,
    // --- rates / cross-asset (FMP) ---
    Ust2y,
    Ust10y,
    Dxy,
    Gold,
    Oil,
    Copper,
    // --- credit + liquidity (FRED, free) ---
    HyOas,
    IgOas,
    FedBalanceSheet,
    Tga,
    ReverseRepo,
}

impl MacroSeries {
    pub const ALL: [MacroSeries; 16] = [
        MacroSeries::Vix,
        MacroSeries::Vix9d,
        MacroSeries::Vix3m,
        MacroSeries::Vvix,
        MacroSeries::Skew,
        MacroSeries::Ust2y,
        MacroSeries::Ust10y,
        MacroSeries::Dxy,
        MacroSeries::Gold,
        MacroSeries::Oil,
        MacroSeries::Copper,
        MacroSeries::HyOas,
        MacroSeries::IgOas,
        MacroSeries::FedBalanceSheet,
        MacroSeries::Tga,
        MacroSeries::ReverseRepo,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            MacroSeries::Vix => "vix",
            MacroSeries::Vix9d => "vix9d",
            MacroSeries::Vix3m => "vix3m",
            MacroSeries::Vvix => "vvix",
            MacroSeries::Skew => "skew",
            MacroSeries::Ust2y => "ust2y",
            MacroSeries::Ust10y => "ust10y",
            MacroSeries::Dxy => "dxy",
            MacroSeries::Gold => "gold",
            MacroSeries::Oil => "oil",
            MacroSeries::Copper => "copper",
            MacroSeries::HyOas => "hy_oas",
            MacroSeries::IgOas => "ig_oas",
            MacroSeries::FedBalanceSheet => "fed_balance_sheet",
            MacroSeries::Tga => "tga",
            MacroSeries::ReverseRepo => "reverse_repo",
        }
    }

    /// Which provider naturally serves this series in v1.
    pub const fn preferred_source(self) -> ProviderKind {
        match self {
            MacroSeries::HyOas
            | MacroSeries::IgOas
            | MacroSeries::FedBalanceSheet
            | MacroSeries::Tga
            | MacroSeries::ReverseRepo => ProviderKind::Fred,
            _ => ProviderKind::Fmp,
        }
    }
}

/// ETF profile/holdings context used by the tradeability gate (flow capacity).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EtfProfile {
    pub ticker: Ticker,
    pub aum: Option<f64>,
    pub expense_ratio: Option<f64>,
    pub avg_volume: Option<f64>,
    pub holdings_count: Option<i64>,
    pub top_sector_weight: Option<f64>,
}

/// Identifies a provider implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Mock,
    Fmp,
    Fred,
    Proprietary,
}

/// What a provider can supply. The tradeability/regime layers query this to know
/// which features are real vs. must be stubbed (`PROPRIETARY_FEATURE`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    pub daily_bars: bool,
    pub intraday_bars: bool,
    pub quotes: bool,
    pub macro_series: bool,
    pub etf_profile: bool,
    /// Options-derived data (GEX/charm/vanna/walls). False for FMP/FRED/Mock in v1.
    pub options: bool,
    /// Tick-level signed order flow. False in v1.
    pub order_flow: bool,
}

impl Capabilities {
    pub const NONE: Capabilities = Capabilities {
        daily_bars: false,
        intraday_bars: false,
        quotes: false,
        macro_series: false,
        etf_profile: false,
        options: false,
        order_flow: false,
    };
}
