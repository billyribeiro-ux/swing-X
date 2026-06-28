//! Ingestion helpers shared by `scan` and `regime-sanity-check`.
//!
//! Two responsibilities:
//!   * `ingest_bars` — pull daily OHLCV for the universe and upsert them.
//!   * `ingest_macro` — once per run (market-wide), pull every `MacroSeries` from
//!     the right provider (`series.preferred_source()` -> FMP or FRED), map
//!     `MacroPoint` -> `MacroWrite`, and `upsert_macro`. Unavailable series are
//!     logged once (`tracing::warn`) and skipped — they NEVER fail the run.

use anyhow::{Context, Result};
use chrono::NaiveDate;
use se_core::Ticker;
use se_provider::{DataProvider, FmpProvider, FredProvider, MacroPoint, MacroSeries, ProviderKind};
use se_store::{MacroWrite, Store};

/// Outcome of a market-wide macro ingest: which series stored rows, and which
/// came back unavailable (so the caller can print a one-line summary).
#[derive(Debug, Default)]
pub struct MacroIngestReport {
    /// (series key, rows stored) for series that produced data.
    pub stored: Vec<(String, u64)>,
    /// Series that were unavailable on this setup.
    pub unavailable: Vec<String>,
}

impl MacroIngestReport {
    pub fn summary_line(&self) -> String {
        let stored: Vec<String> = self.stored.iter().map(|(s, _)| s.clone()).collect();
        format!(
            "macro stored [{}] │ unavailable [{}]",
            stored.join(", "),
            self.unavailable.join(", ")
        )
    }
}

/// Pull the earnings calendar over `[from, to]` for the universe and upsert it into the
/// `earnings` table (the equity scanner's blackout source). Filters to the active universe so the
/// table stays scoped to what we scan. Returns the number of rows stored. Best-effort: a provider
/// that can't supply earnings (mock returns synthetic; FRED/proprietary return empty) just stores
/// nothing rather than failing the run.
pub async fn ingest_earnings(
    store: &Store,
    provider: &dyn DataProvider,
    universe: &[Ticker],
    from: NaiveDate,
    to: NaiveDate,
) -> Result<u64> {
    let events = provider
        .earnings_calendar(from, to)
        .await
        .context("fetch earnings calendar")?;
    let in_universe: std::collections::HashSet<&str> =
        universe.iter().map(|t| t.as_str()).collect();
    let source = match provider.kind() {
        ProviderKind::Fmp => "fmp",
        ProviderKind::Mock => "mock",
        ProviderKind::Fred => "fred",
        ProviderKind::Proprietary => "proprietary",
    };
    let mut stored = 0u64;
    for ev in &events {
        if !in_universe.contains(ev.ticker.as_str()) {
            continue;
        }
        se_store::sqlx::query(
            "INSERT INTO earnings (ticker, date, source) VALUES ($1, $2, $3) \
             ON CONFLICT (ticker, date) DO NOTHING",
        )
        .bind(ev.ticker.as_str())
        .bind(ev.date)
        .bind(source)
        .execute(store.pool())
        .await
        .map_err(|e| anyhow::anyhow!("insert earnings: {e}"))?;
        stored += 1;
    }
    Ok(stored)
}

fn map_point(p: &MacroPoint) -> MacroWrite {
    MacroWrite {
        series: p.series.as_str().to_string(),
        ts: p.ts,
        as_of: p.as_of,
        value: p.value,
        lead_time: p.lead_time,
        source: p.source.clone(),
    }
}

/// Pull every `MacroSeries` over `[from, to]` from the correct provider and store
/// it. `fmp` serves the vol/rates/cross-asset complex; `fred` (if a key is set)
/// serves credit + liquidity. Both are optional — a missing/keyless provider just
/// marks its series unavailable.
pub async fn ingest_macro(
    store: &Store,
    fmp: Option<&FmpProvider>,
    fred: Option<&FredProvider>,
    from: NaiveDate,
    to: NaiveDate,
) -> Result<MacroIngestReport> {
    let mut report = MacroIngestReport::default();

    for series in MacroSeries::ALL {
        let provider: Option<&dyn DataProvider> = match series.preferred_source() {
            ProviderKind::Fmp => fmp.map(|p| p as &dyn DataProvider),
            ProviderKind::Fred => fred.map(|p| p as &dyn DataProvider),
            _ => None,
        };
        let Some(provider) = provider else {
            tracing::warn!(
                series = series.as_str(),
                "macro: no provider configured for series; skipping"
            );
            report.unavailable.push(series.as_str().to_string());
            continue;
        };

        match provider.macro_series(series, from, to).await {
            Ok(points) if !points.is_empty() => {
                let writes: Vec<MacroWrite> = points.iter().map(map_point).collect();
                let n = store
                    .upsert_macro(&writes)
                    .await
                    .with_context(|| format!("upsert macro {}", series.as_str()))?;
                report.stored.push((series.as_str().to_string(), n));
            }
            Ok(_) => {
                tracing::warn!(
                    series = series.as_str(),
                    "macro: provider returned no rows; series unavailable"
                );
                report.unavailable.push(series.as_str().to_string());
            }
            Err(e) => {
                tracing::warn!(
                    series = series.as_str(),
                    error = %e,
                    "macro: series unavailable; skipping (run continues)"
                );
                report.unavailable.push(series.as_str().to_string());
            }
        }
    }

    Ok(report)
}

/// Ingest every `MacroSeries` through a single arbitrary `DataProvider` (used by
/// the mock provider for offline runs). Series the provider cannot serve are
/// warned + marked unavailable; the run continues.
pub async fn ingest_macro_via(
    store: &Store,
    provider: &dyn DataProvider,
    from: NaiveDate,
    to: NaiveDate,
) -> Result<MacroIngestReport> {
    let mut report = MacroIngestReport::default();
    for series in MacroSeries::ALL {
        match provider.macro_series(series, from, to).await {
            Ok(points) if !points.is_empty() => {
                let writes: Vec<MacroWrite> = points.iter().map(map_point).collect();
                let n = store
                    .upsert_macro(&writes)
                    .await
                    .with_context(|| format!("upsert macro {}", series.as_str()))?;
                report.stored.push((series.as_str().to_string(), n));
            }
            _ => {
                tracing::warn!(
                    series = series.as_str(),
                    "macro: series unavailable from provider; skipping"
                );
                report.unavailable.push(series.as_str().to_string());
            }
        }
    }
    Ok(report)
}

/// Pull daily bars for each `ticker` over `[from, to]` and upsert them. Returns
/// the total bar-rows written. A ticker with no bars is warned and skipped.
pub async fn ingest_bars(
    store: &Store,
    provider: &dyn DataProvider,
    source: &str,
    universe: &[Ticker],
    from: NaiveDate,
    to: NaiveDate,
) -> Result<u64> {
    let mut total = 0u64;
    for &t in universe {
        let bars = provider
            .daily_bars(t, from, to)
            .await
            .with_context(|| format!("fetch bars for {t}"))?;
        if bars.is_empty() {
            tracing::warn!(ticker = %t, "no bars returned; skipping");
            continue;
        }
        total += store.upsert_bars(&bars, "daily", source).await?;
    }
    Ok(total)
}
