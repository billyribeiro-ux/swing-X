//! The `DataProvider` abstraction: one trait, swappable adapters. Default method
//! bodies return [`Error::FeatureUnavailable`] so an adapter only implements the
//! categories it actually serves, and callers can fall back to another provider.

use async_trait::async_trait;
use chrono::NaiveDate;
use se_core::{Bar, Error, Result, Ticker};

use crate::types::{
    Capabilities, EarningsEvent, EtfProfile, MacroPoint, MacroSeries, ProviderKind, Quote,
};

#[async_trait]
pub trait DataProvider: Send + Sync {
    fn kind(&self) -> ProviderKind;
    fn capabilities(&self) -> Capabilities;

    /// Daily OHLCV bars for `ticker` over the inclusive date range.
    async fn daily_bars(
        &self,
        _ticker: Ticker,
        _start: NaiveDate,
        _end: NaiveDate,
    ) -> Result<Vec<Bar>> {
        Err(Error::FeatureUnavailable("daily_bars".into()))
    }

    /// Latest quotes for arbitrary symbols (ETFs or index symbols such as `^VIX`).
    async fn quotes(&self, _symbols: &[String]) -> Result<Vec<Quote>> {
        Err(Error::FeatureUnavailable("quotes".into()))
    }

    /// A macro/cross-asset/vol series over the inclusive date range.
    async fn macro_series(
        &self,
        _series: MacroSeries,
        _start: NaiveDate,
        _end: NaiveDate,
    ) -> Result<Vec<MacroPoint>> {
        Err(Error::FeatureUnavailable("macro_series".into()))
    }

    /// ETF profile/holdings context (flow capacity for the tradeability gate).
    async fn etf_profile(&self, _ticker: Ticker) -> Result<Option<EtfProfile>> {
        Ok(None)
    }

    /// The equity scanner's universe: up to `max` liquid US common-stock symbols.
    /// Default is empty so adapters that serve no equity universe (FRED,
    /// proprietary) need not override — callers fall back to another provider.
    async fn equity_universe(&self, _max: usize) -> Result<Vec<Ticker>> {
        Ok(vec![])
    }

    /// Scheduled earnings announcements over the inclusive date range, for the
    /// downstream earnings-blackout guard. Default is empty (no announcements).
    async fn earnings_calendar(
        &self,
        _from: NaiveDate,
        _to: NaiveDate,
    ) -> Result<Vec<EarningsEvent>> {
        Ok(vec![])
    }
}
