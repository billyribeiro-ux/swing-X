//! Financial Modeling Prep adapter (`/stable` REST). Primary v1 provider.
//!
//! Serves daily OHLCV bars, batch quotes, the rates/vol/cross-asset macro
//! complex, and ETF profile/holdings. Credit-spread and net-liquidity series are
//! FRED's job, so this adapter returns [`Error::FeatureUnavailable`] for those.
//!
//! Verified live (2026-06) against `https://financialmodelingprep.com/stable`:
//!
//! * Daily bars: `GET /historical-price-eod/full?symbol=&from=&to=` returns
//!   `[{symbol,date,open,high,low,close,volume,change,changePercent,vwap}]`.
//! * Quotes: `GET /quote?symbol=SPY,QQQ,^VIX` returns
//!   `[{symbol,price,changePercentage,timestamp,...}]`.
//! * Treasury: `GET /treasury-rates?from=&to=` returns
//!   `[{date,month1..,year2,year10,year30,...}]`.
//! * Vol complex: historical-price-eod for `^VIX`,`^VIX9D`,`^VIX3M`,`^VVIX`
//!   (`^SKEW` returns no data on this tier -> FeatureUnavailable).
//! * Cross-asset: historical-price-eod for `DXUSD`,`GCUSD`,`CLUSD`,`HGUSD`.
//! * ETF: `GET /etf/info?symbol=`, `GET /etf/sector-weightings?symbol=`.

use async_trait::async_trait;
use chrono::NaiveDate;
use se_core::{Bar, Error, LeadTimeTag, Result, Ticker};
use serde::Deserialize;
use tracing::warn;

use crate::http_util::{parse_ymd, session_close_ts};
use crate::provider::DataProvider;
use crate::types::{Capabilities, EtfProfile, MacroPoint, MacroSeries, ProviderKind, Quote};

#[derive(Debug, Clone)]
pub struct FmpProvider {
    pub(crate) client: reqwest::Client,
    pub(crate) api_key: String,
    pub(crate) base_url: String,
}

impl FmpProvider {
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        FmpProvider {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }

    /// Build from `FMP_API_KEY` / `FMP_BASE_URL` env vars.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("FMP_API_KEY")
            .map_err(|_| Error::Config("FMP_API_KEY not set".into()))?;
        let base_url = std::env::var("FMP_BASE_URL")
            .unwrap_or_else(|_| "https://financialmodelingprep.com/stable".into());
        Ok(FmpProvider::new(api_key, base_url))
    }

    /// `{base}/{path}` with `?{query}&apikey=...` appended. `query` is the joined
    /// query string WITHOUT the leading `?` or the apikey.
    fn url(&self, path: &str, query: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        if query.is_empty() {
            format!("{base}/{path}?apikey={}", self.api_key)
        } else {
            format!("{base}/{path}?{query}&apikey={}", self.api_key)
        }
    }

    /// GET + JSON-decode into `T`, mapping transport/parse failures to
    /// [`Error::Provider`]. The api key is never included in any error string.
    async fn get_json<T: serde::de::DeserializeOwned>(&self, path: &str, query: &str) -> Result<T> {
        let resp = self
            .client
            .get(self.url(path, query))
            .send()
            .await
            .map_err(|e| Error::Provider(format!("fmp GET /{path} failed: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(Error::Provider(format!(
                "fmp GET /{path} returned HTTP {status}"
            )));
        }
        resp.json::<T>()
            .await
            .map_err(|e| Error::Provider(format!("fmp /{path} decode failed: {e}")))
    }

    /// Fetch the raw historical-price-eod array for any symbol over a date range.
    async fn historical_eod(
        &self,
        symbol: &str,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<EodRow>> {
        let query = format!("symbol={symbol}&from={start}&to={end}");
        self.get_json::<Vec<EodRow>>("historical-price-eod/full", &query)
            .await
    }
}

// --- wire DTOs (private to this adapter) -----------------------------------

/// One row of `/historical-price-eod/full`.
#[derive(Debug, Clone, Deserialize)]
struct EodRow {
    date: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    #[serde(default)]
    volume: f64,
}

/// One row of `/quote`. Many more fields exist; we take only what `Quote` needs.
#[derive(Debug, Clone, Deserialize)]
struct QuoteRow {
    symbol: String,
    price: f64,
    #[serde(default)]
    #[serde(rename = "changePercentage")]
    change_percentage: Option<f64>,
    /// Unix epoch seconds.
    #[serde(default)]
    timestamp: Option<i64>,
}

/// One row of `/treasury-rates`.
#[derive(Debug, Clone, Deserialize)]
struct TreasuryRow {
    date: String,
    #[serde(default)]
    year2: Option<f64>,
    #[serde(default)]
    year10: Option<f64>,
}

/// `/etf/info` row (single-element array for one symbol).
#[derive(Debug, Clone, Deserialize)]
struct EtfInfoRow {
    #[serde(default, rename = "assetsUnderManagement")]
    assets_under_management: Option<f64>,
    /// Reported as a percent (e.g. 0.09 == 0.09%).
    #[serde(default, rename = "expenseRatio")]
    expense_ratio: Option<f64>,
    #[serde(default, rename = "avgVolume")]
    avg_volume: Option<f64>,
    #[serde(default, rename = "holdingsCount")]
    holdings_count: Option<i64>,
}

/// `/etf/sector-weightings` row. `weightPercentage` is a percent (e.g. 39.05).
#[derive(Debug, Clone, Deserialize)]
struct SectorWeightRow {
    #[serde(default, rename = "weightPercentage")]
    weight_percentage: Option<f64>,
}

// --- mapping helpers --------------------------------------------------------

impl EodRow {
    /// To an `se_core::Bar`. Returns `None` if the date is unparseable.
    fn to_bar(&self, ticker: Ticker) -> Option<Bar> {
        let date = parse_ymd(&self.date)?;
        Some(Bar {
            ticker,
            ts: session_close_ts(date),
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
        })
    }

    /// To a `MacroPoint` for an index/cross-asset series (EOD lead time).
    fn to_macro_point(&self, series: MacroSeries) -> Option<MacroPoint> {
        let date = parse_ymd(&self.date)?;
        let ts = session_close_ts(date);
        Some(MacroPoint {
            series,
            ts,
            value: self.close,
            as_of: ts,
            lead_time: LeadTimeTag::EndOfDay,
            source: "fmp".into(),
        })
    }
}

/// FMP symbol for a series this adapter serves; `None` means FRED's job.
/// `Some(None)` is impossible — we return the symbol or signal unavailable below.
fn macro_symbol(series: MacroSeries) -> Option<&'static str> {
    match series {
        MacroSeries::Vix => Some("^VIX"),
        MacroSeries::Vix9d => Some("^VIX9D"),
        MacroSeries::Vix3m => Some("^VIX3M"),
        MacroSeries::Vvix => Some("^VVIX"),
        // Cross-asset (verified working symbols on this tier).
        MacroSeries::Dxy => Some("DXUSD"),
        MacroSeries::Gold => Some("GCUSD"),
        MacroSeries::Oil => Some("CLUSD"),
        MacroSeries::Copper => Some("HGUSD"),
        // Served by treasury-rates, not historical-eod.
        MacroSeries::Ust2y | MacroSeries::Ust10y => None,
        // `^SKEW` returns no data on this key's tier.
        MacroSeries::Skew => None,
        // FRED-preferred — not this adapter's responsibility.
        MacroSeries::HyOas
        | MacroSeries::IgOas
        | MacroSeries::FedBalanceSheet
        | MacroSeries::Tga
        | MacroSeries::ReverseRepo => None,
    }
}

#[async_trait]
impl DataProvider for FmpProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Fmp
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            daily_bars: true,
            intraday_bars: false,
            quotes: true,
            macro_series: true,
            etf_profile: true,
            options: false,
            order_flow: false,
        }
    }

    async fn daily_bars(
        &self,
        ticker: Ticker,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<Bar>> {
        let rows = self.historical_eod(ticker.as_str(), start, end).await?;
        let mut bars: Vec<Bar> = rows.iter().filter_map(|r| r.to_bar(ticker)).collect();
        // FMP returns newest-first; the engine expects ascending time order.
        bars.sort_by_key(|b| b.ts);
        Ok(bars)
    }

    async fn quotes(&self, symbols: &[String]) -> Result<Vec<Quote>> {
        if symbols.is_empty() {
            return Ok(Vec::new());
        }
        let joined = symbols.join(",");
        let query = format!("symbol={joined}");
        let rows = self.get_json::<Vec<QuoteRow>>("quote", &query).await?;
        let out = rows
            .into_iter()
            .map(|r| {
                let ts = r
                    .timestamp
                    .and_then(|s| chrono::DateTime::from_timestamp(s, 0))
                    .unwrap_or_else(chrono::Utc::now);
                Quote {
                    symbol: r.symbol,
                    price: r.price,
                    ts,
                    day_change_pct: r.change_percentage,
                }
            })
            .collect();
        Ok(out)
    }

    async fn macro_series(
        &self,
        series: MacroSeries,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<MacroPoint>> {
        match series {
            // Rates come from the treasury-rates endpoint.
            MacroSeries::Ust2y | MacroSeries::Ust10y => {
                let query = format!("from={start}&to={end}");
                let rows = self
                    .get_json::<Vec<TreasuryRow>>("treasury-rates", &query)
                    .await?;
                let mut out = Vec::new();
                for r in &rows {
                    let Some(date) = parse_ymd(&r.date) else {
                        continue;
                    };
                    let val = match series {
                        MacroSeries::Ust2y => r.year2,
                        MacroSeries::Ust10y => r.year10,
                        _ => unreachable!(),
                    };
                    if let Some(value) = val {
                        let ts = session_close_ts(date);
                        out.push(MacroPoint {
                            series,
                            ts,
                            value,
                            as_of: ts,
                            lead_time: LeadTimeTag::EndOfDay,
                            source: "fmp".into(),
                        });
                    }
                }
                out.sort_by_key(|p| p.ts);
                Ok(out)
            }
            // FRED-preferred series: not this adapter's job.
            MacroSeries::HyOas
            | MacroSeries::IgOas
            | MacroSeries::FedBalanceSheet
            | MacroSeries::Tga
            | MacroSeries::ReverseRepo => Err(Error::FeatureUnavailable(format!(
                "fmp does not serve `{}` (use FRED)",
                series.as_str()
            ))),
            // SKEW has no data on this key's tier.
            MacroSeries::Skew => {
                warn!(
                    series = series.as_str(),
                    "fmp: ^SKEW returns no data on this API tier; series unavailable"
                );
                Err(Error::FeatureUnavailable(format!(
                    "fmp `{}` (^SKEW) unavailable on this tier",
                    series.as_str()
                )))
            }
            // Vol complex + cross-asset via historical-eod.
            _ => {
                let Some(symbol) = macro_symbol(series) else {
                    return Err(Error::FeatureUnavailable(format!(
                        "fmp `{}` has no mapped symbol",
                        series.as_str()
                    )));
                };
                let rows = self.historical_eod(symbol, start, end).await?;
                if rows.is_empty() {
                    warn!(
                        series = series.as_str(),
                        symbol, "fmp: historical-eod returned no rows for symbol"
                    );
                }
                let mut out: Vec<MacroPoint> = rows
                    .iter()
                    .filter_map(|r| r.to_macro_point(series))
                    .collect();
                out.sort_by_key(|p| p.ts);
                Ok(out)
            }
        }
    }

    async fn etf_profile(&self, ticker: Ticker) -> Result<Option<EtfProfile>> {
        let query = format!("symbol={}", ticker.as_str());

        let info = self
            .get_json::<Vec<EtfInfoRow>>("etf/info", &query)
            .await?
            .into_iter()
            .next();

        // Top sector weight is independent; tolerate it being empty.
        let top_sector_weight = match self
            .get_json::<Vec<SectorWeightRow>>("etf/sector-weightings", &query)
            .await
        {
            Ok(rows) => rows
                .iter()
                .filter_map(|r| r.weight_percentage)
                .fold(None::<f64>, |acc, w| Some(acc.map_or(w, |m| m.max(w))))
                // express as a fraction to match the rest of the system (0.39, not 39.05)
                .map(|pct| pct / 100.0),
            Err(e) => {
                warn!(ticker = ticker.as_str(), error = %e, "fmp: sector-weightings fetch failed");
                None
            }
        };

        let Some(info) = info else {
            // No info row -> nothing useful to return, but sector weight alone is
            // not a "profile". Signal absence rather than a half-empty record.
            if top_sector_weight.is_none() {
                return Ok(None);
            }
            return Ok(Some(EtfProfile {
                ticker,
                aum: None,
                expense_ratio: None,
                avg_volume: None,
                holdings_count: None,
                top_sector_weight,
            }));
        };

        Ok(Some(EtfProfile {
            ticker,
            aum: info.assets_under_management,
            // API reports expense ratio as a percent; store as a fraction.
            expense_ratio: info.expense_ratio.map(|p| p / 100.0),
            avg_volume: info.avg_volume,
            holdings_count: info.holdings_count,
            top_sector_weight,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Samples copied verbatim from observed live responses (2026-06).

    #[test]
    fn parse_eod_rows_to_bars() {
        let sample = r#"[
            {"symbol":"SPY","date":"2026-06-26","open":728.95,"high":736.53,"low":716.58,"close":728.99,"volume":71033969,"change":0.045,"changePercent":0.00548734,"vwap":727.7625},
            {"symbol":"SPY","date":"2026-06-25","open":730.10,"high":732.00,"low":725.10,"close":728.94,"volume":60000000,"change":-1.0,"changePercent":-0.1,"vwap":729.0}
        ]"#;
        let rows: Vec<EodRow> = serde_json::from_str(sample).unwrap();
        assert_eq!(rows.len(), 2);
        let mut bars: Vec<Bar> = rows.iter().filter_map(|r| r.to_bar(Ticker::SPY)).collect();
        bars.sort_by_key(|b| b.ts);
        assert_eq!(bars.len(), 2);
        // ascending after sort: 06-25 then 06-26
        assert_eq!(
            bars[0].ts,
            session_close_ts(parse_ymd("2026-06-25").unwrap())
        );
        let b = &bars[1];
        assert_eq!(b.ticker, Ticker::SPY);
        assert_eq!(b.close, 728.99);
        assert!(b.high >= b.low);
        assert!(b.high >= b.open && b.high >= b.close);
        assert!(b.low <= b.open && b.low <= b.close);
        // session-close convention: 21:00:00 UTC
        assert_eq!(b.ts.format("%H:%M:%S").to_string(), "21:00:00");
    }

    #[test]
    fn parse_quote_rows() {
        let sample = r#"[
            {"symbol":"SPY","name":"State Street SPDR S&P 500 ETF","price":728.99,"changePercentage":-0.72314,"change":-5.31,"volume":69241946,"exchange":"AMEX","open":728.945,"previousClose":734.3,"timestamp":1782504000}
        ]"#;
        let rows: Vec<QuoteRow> = serde_json::from_str(sample).unwrap();
        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.symbol, "SPY");
        assert_eq!(r.price, 728.99);
        assert_eq!(r.change_percentage, Some(-0.72314));
        let ts = chrono::DateTime::from_timestamp(r.timestamp.unwrap(), 0).unwrap();
        assert_eq!(ts.format("%Y-%m-%d").to_string(), "2026-06-26");
    }

    #[test]
    fn parse_treasury_rows() {
        let sample = r#"[
            {"date":"2026-06-26","month1":3.7,"month2":3.75,"month3":3.83,"month6":3.94,"year1":3.94,"year2":4.07,"year3":4.09,"year5":4.12,"year7":4.23,"year10":4.38,"year20":4.87,"year30":4.87}
        ]"#;
        let rows: Vec<TreasuryRow> = serde_json::from_str(sample).unwrap();
        assert_eq!(rows[0].year2, Some(4.07));
        assert_eq!(rows[0].year10, Some(4.38));
    }

    #[test]
    fn parse_vix_eod_to_macro_point() {
        let sample = r#"[
            {"symbol":"^VIX","date":"2026-06-26","open":19.7,"high":20.72,"low":18.2,"close":18.41,"volume":0,"change":-1.29,"changePercent":-6.55,"vwap":19.2575}
        ]"#;
        let rows: Vec<EodRow> = serde_json::from_str(sample).unwrap();
        let p = rows[0].to_macro_point(MacroSeries::Vix).unwrap();
        assert_eq!(p.series, MacroSeries::Vix);
        assert_eq!(p.value, 18.41);
        assert_eq!(p.source, "fmp");
        assert_eq!(p.lead_time, LeadTimeTag::EndOfDay);
        assert_eq!(p.as_of, p.ts);
    }

    #[test]
    fn parse_etf_info_and_sectors() {
        let info_sample = r#"[
            {"symbol":"SPY","name":"SPDR","assetsUnderManagement":772079160000,"expenseRatio":0.09,"avgVolume":57978022,"holdingsCount":504,"nav":732.92}
        ]"#;
        let info: Vec<EtfInfoRow> = serde_json::from_str(info_sample).unwrap();
        let i = &info[0];
        assert_eq!(i.assets_under_management, Some(772079160000.0));
        assert_eq!(i.expense_ratio, Some(0.09));
        assert_eq!(i.avg_volume, Some(57978022.0));
        assert_eq!(i.holdings_count, Some(504));

        let sec_sample = r#"[
            {"symbol":"SPY","sector":"Technology","weightPercentage":39.05},
            {"symbol":"SPY","sector":"Financial Services","weightPercentage":11.07}
        ]"#;
        let secs: Vec<SectorWeightRow> = serde_json::from_str(sec_sample).unwrap();
        let top = secs
            .iter()
            .filter_map(|r| r.weight_percentage)
            .fold(None::<f64>, |acc, w| Some(acc.map_or(w, |m| m.max(w))))
            .map(|p| p / 100.0);
        assert!((top.unwrap() - 0.3905).abs() < 1e-9);
    }

    #[tokio::test]
    async fn fred_preferred_series_unavailable() {
        let p = FmpProvider::new("k", "https://example.invalid");
        let start = NaiveDate::from_ymd_opt(2026, 6, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2026, 6, 26).unwrap();
        for s in [
            MacroSeries::HyOas,
            MacroSeries::IgOas,
            MacroSeries::FedBalanceSheet,
            MacroSeries::Tga,
            MacroSeries::ReverseRepo,
            MacroSeries::Skew,
        ] {
            let err = p.macro_series(s, start, end).await.unwrap_err();
            assert!(matches!(err, Error::FeatureUnavailable(_)), "{s:?}");
        }
    }

    #[test]
    fn url_builds_with_and_without_query() {
        let p = FmpProvider::new("SECRET", "https://financialmodelingprep.com/stable/");
        let u = p.url("quote", "symbol=SPY");
        assert_eq!(
            u,
            "https://financialmodelingprep.com/stable/quote?symbol=SPY&apikey=SECRET"
        );
        let u2 = p.url("treasury-rates", "");
        assert_eq!(
            u2,
            "https://financialmodelingprep.com/stable/treasury-rates?apikey=SECRET"
        );
    }
}
