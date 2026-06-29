//! Live FMP integration test. Gated on `FMP_API_KEY`: if unset it prints a skip
//! notice and returns Ok, so CI without secrets stays green. When the key is
//! present it makes a couple of small calls (respecting rate limits): ~2 weeks of
//! daily SPY bars + a `^VIX` quote, asserting non-empty + OHLC/value sanity.
//!
//! Run live with:
//!   set -a; source ../../.env; set +a
//!   cargo test -p se-provider --test fmp_live -- --nocapture

use chrono::{Duration, Utc};
use se_core::Ticker;
use se_provider::{DataProvider, FmpProvider};

#[tokio::test]
async fn fmp_live_spy_bars_and_vix_quote() {
    if std::env::var("FMP_API_KEY")
        .ok()
        .filter(|s| !s.is_empty())
        .is_none()
    {
        eprintln!("SKIP fmp_live: FMP_API_KEY not set");
        return;
    }

    let provider = FmpProvider::from_env().expect("FmpProvider::from_env with key set");

    // ~2-week window ending today.
    let end = Utc::now().date_naive();
    let start = end - Duration::days(14);

    let bars = provider
        .daily_bars(Ticker::SPY, start, end)
        .await
        .expect("SPY daily_bars live fetch");

    assert!(
        !bars.is_empty(),
        "expected at least some SPY bars in a 2-week window"
    );
    // A 2-week window has ~10 trading days; allow slack for holidays/weekends.
    assert!(
        bars.len() >= 3,
        "expected several trading days, got {}",
        bars.len()
    );
    // Bars must be ascending in time and OHLC-sane with a plausible SPY price.
    let mut prev_ts = None;
    for b in &bars {
        if let Some(p) = prev_ts {
            assert!(b.ts > p, "bars must be strictly ascending in time");
        }
        prev_ts = Some(b.ts);
        assert!(b.high >= b.low, "high >= low");
        assert!(b.high >= b.open && b.high >= b.close, "high bounds o/c");
        assert!(b.low <= b.open && b.low <= b.close, "low bounds o/c");
        assert!(
            b.close > 50.0 && b.close < 5000.0,
            "SPY close plausible: {}",
            b.close
        );
        assert_eq!(
            b.ts.format("%H:%M:%S").to_string(),
            "21:00:00",
            "session-close convention"
        );
    }

    let last = bars.last().unwrap();
    eprintln!(
        "LIVE FMP: SPY bars={} | last {} O={} H={} L={} C={} V={}",
        bars.len(),
        last.ts.format("%Y-%m-%d"),
        last.open,
        last.high,
        last.low,
        last.close,
        last.volume
    );

    // ^VIX quote — assert a plausible level.
    let quotes = provider
        .quotes(&["^VIX".to_string()])
        .await
        .expect("^VIX quote live fetch");
    assert!(!quotes.is_empty(), "expected a ^VIX quote");
    let vix = &quotes[0];
    assert_eq!(vix.symbol, "^VIX");
    assert!(
        vix.price > 5.0 && vix.price < 150.0,
        "VIX value plausible: {}",
        vix.price
    );
    eprintln!(
        "LIVE FMP: ^VIX price={} ts={} day_change_pct={:?}",
        vix.price,
        vix.ts.format("%Y-%m-%d %H:%M:%SZ"),
        vix.day_change_pct
    );
}
