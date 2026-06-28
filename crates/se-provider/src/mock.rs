//! Deterministic synthetic provider for tests and offline development.
//!
//! Output is a pure function of `(ticker/series, date)` — no RNG state, no clock —
//! so every run is byte-for-byte reproducible. Paths are smooth and mean-reverting
//! enough to exercise the feature math without pretending to be real market data.

use async_trait::async_trait;
use chrono::{Datelike, NaiveDate, TimeZone, Utc, Weekday};
use se_core::{Bar, LeadTimeTag, Result, Ticker};

use crate::provider::DataProvider;
use crate::types::{
    Capabilities, EarningsEvent, EtfProfile, MacroPoint, MacroSeries, ProviderKind, Quote,
};

/// The deterministic mock equity universe: liquid US large-cap common stocks, in a
/// fixed canonical order. `equity_universe(max)` returns a prefix of this list, and
/// `earnings_calendar` synthesizes one event per name — so the equity scanner and
/// its earnings-blackout guard are fully testable offline.
const MOCK_EQUITY_UNIVERSE: [&str; 20] = [
    "TSLA", "AAPL", "META", "NVDA", "AMZN", "GOOGL", "MSFT", "AMD", "NFLX", "CRM", "AVGO", "COST",
    "PEP", "ADBE", "INTC", "CSCO", "QCOM", "TXN", "AMAT", "MU",
];

/// Hash a few integers into a stable u64 (splitmix64 finalizer over a fnv mix).
fn hash_u64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Stable pseudo-random f64 in [0, 1) from a string + integers.
fn hash01(tag: &str, a: i64, b: u64) -> f64 {
    let mut h = 1469598103934665603u64; // fnv offset
    for byte in tag.bytes() {
        h = (h ^ byte as u64).wrapping_mul(1099511628211);
    }
    h ^= a as u64;
    h = hash_u64(h ^ b);
    (h >> 11) as f64 / (1u64 << 53) as f64
}

fn ticker_base_price(t: Ticker) -> f64 {
    const TABLE: &[(Ticker, f64)] = &[
        (Ticker::SPY, 530.0),
        (Ticker::QQQ, 460.0),
        (Ticker::IWM, 205.0),
        (Ticker::DIA, 400.0),
        (Ticker::XLF, 42.0),
        (Ticker::XLK, 225.0),
        (Ticker::XLE, 92.0),
        (Ticker::SMH, 235.0),
        (Ticker::XLV, 150.0),
        (Ticker::XLU, 70.0),
    ];
    if let Some(&(_, px)) = TABLE.iter().find(|(tk, _)| *tk == t) {
        return px;
    }
    // Arbitrary equity (mock mode): a deterministic base price in [20, 470) from the symbol.
    20.0 + (symbol_seed(t) % 450) as f64
}

fn ticker_base_volume(t: Ticker) -> f64 {
    // Rough daily share volume; broad indices trade far more than sectors.
    const TABLE: &[(Ticker, f64)] = &[
        (Ticker::SPY, 70_000_000.0),
        (Ticker::QQQ, 45_000_000.0),
        (Ticker::IWM, 30_000_000.0),
        (Ticker::DIA, 4_000_000.0),
        (Ticker::XLF, 40_000_000.0),
        (Ticker::XLK, 8_000_000.0),
        (Ticker::XLE, 18_000_000.0),
        (Ticker::SMH, 9_000_000.0),
        (Ticker::XLV, 9_000_000.0),
        (Ticker::XLU, 12_000_000.0),
    ];
    if let Some(&(_, vol)) = TABLE.iter().find(|(tk, _)| *tk == t) {
        return vol;
    }
    // Arbitrary equity (mock mode): a deterministic daily volume in [1M, 26M).
    1_000_000.0 + (symbol_seed(t) % 25) as f64 * 1_000_000.0
}

/// A small deterministic seed derived from a symbol's bytes (mock determinism for equities).
fn symbol_seed(t: Ticker) -> u64 {
    t.as_str().bytes().fold(1469598103934665603u64, |h, b| {
        (h ^ b as u64).wrapping_mul(1099511628211)
    })
}

fn synth_bar(ticker: Ticker, date: NaiveDate) -> Bar {
    let ord = date.num_days_from_ce() as i64;
    let sym = ticker.as_str();
    let base = ticker_base_price(ticker);
    // Smooth multi-scale trend + small idiosyncratic noise.
    let trend = (ord as f64 / 180.0).sin() * 0.08 + (ord as f64 / 40.0).sin() * 0.03;
    let noise = (hash01(sym, ord, 1) - 0.5) * 0.012;
    let close = base * (trend + noise).exp();
    let open = close * ((hash01(sym, ord, 2) - 0.5) * 0.006).exp();
    let hi_wick = hash01(sym, ord, 3) * 0.004;
    let lo_wick = hash01(sym, ord, 4) * 0.004;
    let high = open.max(close) * (1.0 + hi_wick);
    let low = open.min(close) * (1.0 - lo_wick);
    let vol = ticker_base_volume(ticker) * (1.0 + 0.3 * (ord as f64 / 20.0).sin() + noise);
    let ts = Utc
        .with_ymd_and_hms(date.year(), date.month(), date.day(), 21, 0, 0)
        .single()
        .expect("valid ts");
    Bar {
        ticker,
        ts,
        open,
        high,
        low,
        close,
        volume: vol.max(0.0),
    }
}

fn macro_baseline(series: MacroSeries) -> f64 {
    match series {
        MacroSeries::Vix => 16.0,
        MacroSeries::Vix9d => 14.5,
        MacroSeries::Vix3m => 18.0,
        MacroSeries::Vvix => 95.0,
        MacroSeries::Skew => 140.0,
        MacroSeries::Ust2y => 4.4,
        MacroSeries::Ust10y => 4.25,
        MacroSeries::Dxy => 104.0,
        MacroSeries::Gold => 2350.0,
        MacroSeries::Oil => 78.0,
        MacroSeries::Copper => 4.1,
        MacroSeries::HyOas => 3.2,
        MacroSeries::IgOas => 0.9,
        MacroSeries::FedBalanceSheet => 7_200_000.0,
        MacroSeries::Tga => 700_000.0,
        MacroSeries::ReverseRepo => 500_000.0,
    }
}

fn is_weekday(d: NaiveDate) -> bool {
    !matches!(d.weekday(), Weekday::Sat | Weekday::Sun)
}

/// Deterministic synthetic data provider.
#[derive(Debug, Default, Clone, Copy)]
pub struct MockProvider;

#[async_trait]
impl DataProvider for MockProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Mock
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
        let mut out = Vec::new();
        let mut d = start;
        while d <= end {
            if is_weekday(d) {
                out.push(synth_bar(ticker, d));
            }
            d = d.succ_opt().expect("date in range");
        }
        Ok(out)
    }

    async fn quotes(&self, symbols: &[String]) -> Result<Vec<Quote>> {
        let today = Utc::now().date_naive();
        let mut out = Vec::new();
        for s in symbols {
            if let Ok(t) = s.parse::<Ticker>() {
                let bar = synth_bar(t, today);
                out.push(Quote {
                    symbol: s.clone(),
                    price: bar.close,
                    ts: bar.ts,
                    day_change_pct: Some((bar.close / bar.open - 1.0) * 100.0),
                });
            }
        }
        Ok(out)
    }

    async fn macro_series(
        &self,
        series: MacroSeries,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<MacroPoint>> {
        let base = macro_baseline(series);
        let mut out = Vec::new();
        let mut d = start;
        while d <= end {
            if is_weekday(d) {
                let ord = d.num_days_from_ce() as i64;
                let wiggle = (ord as f64 / 30.0).sin() * 0.05
                    + (hash01(series.as_str(), ord, 7) - 0.5) * 0.04;
                let value = base * (1.0 + wiggle);
                let ts = Utc
                    .with_ymd_and_hms(d.year(), d.month(), d.day(), 21, 0, 0)
                    .single()
                    .expect("valid ts");
                out.push(MacroPoint {
                    series,
                    ts,
                    value,
                    as_of: ts,
                    lead_time: LeadTimeTag::EndOfDay,
                    source: "mock".into(),
                });
            }
            d = d.succ_opt().expect("date in range");
        }
        Ok(out)
    }

    async fn etf_profile(&self, ticker: Ticker) -> Result<Option<EtfProfile>> {
        let avg_vol = ticker_base_volume(ticker);
        let price = ticker_base_price(ticker);
        // crude AUM proxy: dollar volume * a multiple by broad vs sector
        let aum = avg_vol * price * if ticker.is_broad_index() { 8.0 } else { 3.0 };
        Ok(Some(EtfProfile {
            ticker,
            aum: Some(aum),
            expense_ratio: Some(if ticker.is_broad_index() {
                0.0009
            } else {
                0.0010
            }),
            avg_volume: Some(avg_vol),
            holdings_count: Some(if ticker.is_broad_index() { 500 } else { 70 }),
            top_sector_weight: Some(if ticker.is_broad_index() { 0.30 } else { 0.95 }),
        }))
    }

    async fn equity_universe(&self, max: usize) -> Result<Vec<Ticker>> {
        // Deterministic fixed list, truncated to `max`. Every entry is a valid
        // symbol, so `Ticker::new` never fails here.
        Ok(MOCK_EQUITY_UNIVERSE
            .iter()
            .filter_map(|s| Ticker::new(s).ok())
            .take(max)
            .collect())
    }

    async fn earnings_calendar(
        &self,
        from: NaiveDate,
        to: NaiveDate,
    ) -> Result<Vec<EarningsEvent>> {
        if from > to {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        // For each name, emit a stable, roughly-quarterly event: a per-symbol
        // day-offset into each calendar quarter, kept inside [from, to]. The
        // offset is derived from the symbol hash so the schedule is reproducible.
        for sym in MOCK_EQUITY_UNIVERSE {
            let Ok(ticker) = Ticker::new(sym) else {
                continue;
            };
            // 0..=80 days into the quarter (every quarter spans >= 90 days).
            let day_offset = (symbol_seed(ticker) % 81) as i64;
            for q_start in quarter_starts(from, to) {
                let event = q_start + chrono::Duration::days(day_offset);
                if event >= from && event <= to {
                    out.push(EarningsEvent {
                        ticker,
                        date: event,
                    });
                }
            }
        }
        out.sort_by(|a, b| (a.date, a.ticker).cmp(&(b.date, b.ticker)));
        Ok(out)
    }
}

/// First day of every calendar quarter (Jan/Apr/Jul/Oct 1) whose quarter overlaps
/// `[from, to]`. Starts one quarter before `from` so an event landing late in the
/// prior quarter can still fall within the range.
fn quarter_starts(from: NaiveDate, to: NaiveDate) -> Vec<NaiveDate> {
    let mut out = Vec::new();
    // Begin at the quarter start at or before `from`.
    let first_q_month = ((from.month0() / 3) * 3) + 1; // 1, 4, 7, or 10
    let mut cur =
        NaiveDate::from_ymd_opt(from.year(), first_q_month, 1).expect("valid quarter start");
    while cur <= to {
        out.push(cur);
        // Advance three months.
        let mut y = cur.year();
        let mut m = cur.month() + 3;
        if m > 12 {
            m -= 12;
            y += 1;
        }
        cur = NaiveDate::from_ymd_opt(y, m, 1).expect("valid quarter start");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn deterministic_and_weekday_only() {
        let p = MockProvider;
        let start = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2025, 1, 31).unwrap();
        let a = p.daily_bars(Ticker::SPY, start, end).await.unwrap();
        let b = p.daily_bars(Ticker::SPY, start, end).await.unwrap();
        assert_eq!(a, b, "mock output must be deterministic");
        assert!(!a.is_empty());
        for bar in &a {
            assert!(bar.high >= bar.low);
            assert!(bar.high >= bar.open && bar.high >= bar.close);
            assert!(bar.low <= bar.open && bar.low <= bar.close);
            assert!(is_weekday(bar.ts.date_naive()));
        }
    }

    #[tokio::test]
    async fn equity_universe_is_deterministic_and_truncates() {
        let p = MockProvider;
        let full = p.equity_universe(100).await.unwrap();
        // The whole fixed list is returned when `max` exceeds it.
        assert_eq!(full.len(), MOCK_EQUITY_UNIVERSE.len());
        assert_eq!(full[0].as_str(), "TSLA");
        assert_eq!(full[1].as_str(), "AAPL");
        assert_eq!(full.last().unwrap().as_str(), "MU");

        // Deterministic across calls.
        let again = p.equity_universe(100).await.unwrap();
        assert_eq!(full, again);

        // Truncates to `max`, preserving order.
        let five = p.equity_universe(5).await.unwrap();
        assert_eq!(five, full[..5].to_vec());

        // Zero is empty; none of them is a known v1 ETF.
        assert!(p.equity_universe(0).await.unwrap().is_empty());
        assert!(full.iter().all(|t| !t.is_etf()));
    }

    #[tokio::test]
    async fn earnings_calendar_falls_within_range_and_is_deterministic() {
        let p = MockProvider;
        let from = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let to = NaiveDate::from_ymd_opt(2026, 12, 31).unwrap();
        let events = p.earnings_calendar(from, to).await.unwrap();

        assert!(
            !events.is_empty(),
            "should synthesize earnings in a full year"
        );
        // Every event is inside the range and references a universe symbol.
        let universe: Vec<Ticker> = p.equity_universe(usize::MAX).await.unwrap();
        for e in &events {
            assert!(e.date >= from && e.date <= to, "{:?} out of range", e.date);
            assert!(
                universe.contains(&e.ticker),
                "{:?} not in universe",
                e.ticker
            );
        }
        // Sorted ascending by (date, ticker).
        let mut sorted = events.clone();
        sorted.sort_by(|a, b| (a.date, a.ticker).cmp(&(b.date, b.ticker)));
        assert_eq!(events, sorted);

        // Roughly quarterly: across a full year each name reports ~4 times.
        let aapl = Ticker::new("AAPL").unwrap();
        let aapl_events = events.iter().filter(|e| e.ticker == aapl).count();
        assert!(
            (3..=5).contains(&aapl_events),
            "expected ~quarterly cadence, got {aapl_events}"
        );

        // Deterministic across calls.
        let again = p.earnings_calendar(from, to).await.unwrap();
        assert_eq!(events, again);
    }

    #[tokio::test]
    async fn earnings_calendar_respects_a_narrow_window() {
        let p = MockProvider;
        // A single day: at most one event per symbol, all on that exact day.
        let day = NaiveDate::from_ymd_opt(2026, 5, 15).unwrap();
        let events = p.earnings_calendar(day, day).await.unwrap();
        for e in &events {
            assert_eq!(e.date, day);
        }
        // Inverted range yields nothing.
        let to = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let from = NaiveDate::from_ymd_opt(2026, 12, 31).unwrap();
        assert!(p.earnings_calendar(from, to).await.unwrap().is_empty());
    }
}
