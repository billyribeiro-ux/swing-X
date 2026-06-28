//! `regime-sanity-check` — the P2 checkpoint.
//!
//! Labels SPY (and QQQ) over several KNOWN event windows and cross-references the
//! result against what the regime SHOULD look like in each. Because the default
//! 24-month window may not contain classic crises, this command pulls each window
//! directly from FMP (which has deep history), ingests bars + macro, labels every
//! trading day, and prints a per-window regime distribution with a PASS/FAIL on
//! the expectation.
//!
//! Windows + expectations:
//!   * 2020-02-20..2020-03-23  (COVID crash)  -> RiskOff / VolExpansion majority.
//!   * 2022-01-01..2022-06-30  (bear / vol)   -> frequent RiskOff / VolExpansion.
//!   * 2024-05-01..2024-07-31  (calm uptrend) -> RiskOn / VolCompression majority.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use chrono::{Duration, NaiveDate};
use se_core::{RegimeLabel, Ticker};
use se_provider::{FmpProvider, FredProvider};
use se_regime::{RegimeAssessment, RegimeEngine};
use se_store::Store;

use crate::ingest::{ingest_bars, ingest_macro};

/// A named event window plus the regimes we expect to dominate within it.
struct EventWindow {
    name: &'static str,
    from: NaiveDate,
    to: NaiveDate,
    /// Labels whose COMBINED share must clear `min_share` for a PASS.
    expect_any: &'static [RegimeLabel],
    min_share: f64,
}

fn ymd(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).expect("valid date")
}

fn windows() -> Vec<EventWindow> {
    vec![
        EventWindow {
            name: "COVID crash (2020-02-20..03-23)",
            from: ymd(2020, 2, 20),
            to: ymd(2020, 3, 23),
            expect_any: &[RegimeLabel::RiskOff, RegimeLabel::VolExpansion],
            min_share: 0.50,
        },
        EventWindow {
            name: "2022 bear/vol (2022-01..06)",
            from: ymd(2022, 1, 1),
            to: ymd(2022, 6, 30),
            expect_any: &[RegimeLabel::RiskOff, RegimeLabel::VolExpansion],
            min_share: 0.30,
        },
        EventWindow {
            name: "calm uptrend (2024-05..07)",
            from: ymd(2024, 5, 1),
            to: ymd(2024, 7, 31),
            expect_any: &[RegimeLabel::RiskOn, RegimeLabel::VolCompression],
            min_share: 0.50,
        },
    ]
}

/// Per-label share within a labeled window.
fn distribution(
    assessments: &[(chrono::DateTime<chrono::Utc>, RegimeAssessment)],
) -> (BTreeMap<RegimeLabel, f64>, usize) {
    let n = assessments.len();
    let mut counts: BTreeMap<RegimeLabel, usize> = BTreeMap::new();
    for (_, a) in assessments {
        *counts.entry(a.label).or_insert(0) += 1;
    }
    let dist = counts
        .into_iter()
        .map(|(k, c)| (k, if n > 0 { c as f64 / n as f64 } else { 0.0 }))
        .collect();
    (dist, n)
}

/// Run the full sanity check against live FMP. `tickers` is usually `[SPY, QQQ]`.
pub async fn run(store: &Store, tickers: &[Ticker]) -> Result<()> {
    let fmp = FmpProvider::from_env().context("init FMP provider (FMP_API_KEY required)")?;
    // FRED is optional (key may be empty) — credit/liquidity then stay unavailable.
    let fred = FredProvider::from_env();

    let engine = RegimeEngine::default();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        " swing-X regime-sanity-check │ provider=fmp │ tickers={}",
        tickers
            .iter()
            .map(|t| t.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut all_pass = true;

    for w in windows() {
        // Pad the ingest start so rv-percentile / trend windows have history that
        // is itself within the leakage-safe lookback when labeling the window.
        let ingest_from = w.from - Duration::days(420);

        // ---- ingest bars (universe needed? only the tickers we label) ------
        let bars_n = ingest_bars(store, &fmp, "fmp", tickers, ingest_from, w.to)
            .await
            .with_context(|| format!("ingest bars for {}", w.name))?;

        // ---- ingest macro (market-wide, once per window) -------------------
        let macro_report = ingest_macro(store, Some(&fmp), Some(&fred), ingest_from, w.to)
            .await
            .with_context(|| format!("ingest macro for {}", w.name))?;

        println!("\n▌ {}", w.name);
        println!(
            "  ingested {bars_n} bar-rows │ {}",
            macro_report.summary_line()
        );

        for &ticker in tickers {
            let from_ts = crate::session_close(w.from);
            let to_ts = crate::session_close(w.to);
            let labeled = engine
                .label_window(store, ticker, from_ts, to_ts)
                .await
                .with_context(|| format!("label {ticker} over {}", w.name))?;

            let (dist, n) = distribution(&labeled);
            let share: f64 = w
                .expect_any
                .iter()
                .map(|l| dist.get(l).copied().unwrap_or(0.0))
                .sum();
            let pass = n > 0 && share >= w.min_share;
            all_pass &= pass;

            let expect_str = w
                .expect_any
                .iter()
                .map(|l| l.as_str())
                .collect::<Vec<_>>()
                .join("/");
            println!(
                "  {:<4} n={:<3} expect[{}]>={:.0}% got={:.0}%  {}",
                ticker.as_str(),
                n,
                expect_str,
                w.min_share * 100.0,
                share * 100.0,
                if pass { "PASS" } else { "FAIL" }
            );
            // Per-regime breakdown line.
            let mut parts: Vec<(RegimeLabel, f64)> = dist.into_iter().collect();
            parts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let breakdown = parts
                .iter()
                .map(|(l, p)| format!("{}={:.0}%", l.as_str(), p * 100.0))
                .collect::<Vec<_>>()
                .join("  ");
            println!("       {breakdown}");
        }
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        " regime-sanity-check: {}",
        if all_pass {
            "ALL PASS"
        } else {
            "SOME FAIL (see above)"
        }
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    Ok(())
}
