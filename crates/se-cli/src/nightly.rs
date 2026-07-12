//! `se nightly` — execute the nightly walk-forward loop (§0) once.
//!
//! Runs the canonical [`se_orchestrator::NIGHTLY`] job graph against the live store and
//! ML worker (ingest, search, signals, journal, monitor, changelog). Idempotent (upserts),
//! resumable (each step is independent), and logged. In production a cron/apalis schedule
//! fires this after the close; here it runs on demand.
//!
//! Note: the search step materializes per-bar features across the universe and validates
//! candidates on the ML worker, so a full multi-month history is a genuine batch job.

use anyhow::{bail, Context, Result};
use chrono::{Duration, Utc};

use se_config::AppConfig;
use se_orchestrator::{Step, NIGHTLY};
use se_provider::{FmpProvider, FredProvider, MockProvider};
use se_store::Store;

use crate::{ingest, search_cmd, signals_cmd};

#[derive(clap::Args)]
pub struct NightlyArgs {
    /// `fmp` or `mock`. Defaults to `fmp` when FMP_API_KEY is configured, else `mock`.
    #[arg(long)]
    pub provider: Option<String>,
    /// Generations to evolve in the search step.
    #[arg(long, default_value_t = 2)]
    pub generations: u32,
    /// Genomes evaluated per generation.
    #[arg(long, default_value_t = 12)]
    pub per_gen: usize,
    /// Days of history to ingest + search over (multi-regime history needs ~500+).
    #[arg(long, default_value_t = 540)]
    pub history_days: i64,
    /// Horizon override (P8 axis). Defaults to SE_HORIZON / config.
    #[arg(long)]
    pub horizon: Option<String>,
    /// Ground-rule stop geometry: `atr:1.0` | `fixed:5.35` | `pct:2.5`. Default: horizon/config.
    #[arg(long)]
    pub stop: Option<String>,
    /// Ground-rule primary target: `r:2.0` | `atr:2.0` | `fixed:10` | `pct:3`. Default: config.
    #[arg(long)]
    pub target1: Option<String>,
    /// Ground-rule second target (same forms, or `none` to drop it). Default: config.
    #[arg(long)]
    pub target2: Option<String>,
    /// Lock risk to the operator's ground rules (optimize only the conditions). Default: explore.
    #[arg(long, default_value_t = false)]
    pub lock_risk: bool,
}

/// Pick the data provider for the nightly run: an explicit `--provider` (lowercased) wins;
/// otherwise default to `fmp` when a key is configured and `mock` when it isn't — so a keyless
/// environment never tries to hit the live API.
fn choose_provider(explicit: Option<&str>, fmp_configured: bool) -> String {
    explicit.map(|s| s.to_ascii_lowercase()).unwrap_or_else(|| {
        if fmp_configured {
            "fmp".into()
        } else {
            "mock".into()
        }
    })
}

pub async fn run(cfg: &AppConfig, args: NightlyArgs) -> Result<()> {
    let store = Store::connect(&cfg.database_url)
        .await
        .context("connect db")?;
    store.migrate().await.context("migrate")?;

    let chosen = choose_provider(args.provider.as_deref(), cfg.fmp_configured);

    let to = Utc::now().date_naive();
    let from = to - Duration::days(args.history_days.max(1));

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        " se nightly │ provider={} │ horizon={} │ history {} → {}",
        chosen,
        cfg.horizon.horizon.as_str(),
        from,
        to
    );
    println!(" loop: {}", NIGHTLY.map(|s| s.label()).join(" → "));
    // The nightly search step delegates to `search_cmd::run_search`, which reads `cfg.test_era()`
    // and hands the reserved window to the dataset FIREWALL — so any reserved out-of-time era is
    // purged from nightly training too (report-only; never feeds ranking/gate/promotion).
    if let Some((f, t)) = cfg.test_era() {
        println!(" out-of-time TEST ERA: {f} → {t} FIREWALLED from nightly search (report-only)");
    }
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    for step in NIGHTLY {
        println!("\n▌ {} — {}", step.label(), step.detail());
        match step {
            Step::Ingest => match chosen.as_str() {
                "fmp" => {
                    let fmp = FmpProvider::from_env().context("init FMP")?;
                    let fred = FredProvider::from_env();
                    let n =
                        ingest::ingest_bars(&store, &fmp, "fmp", &cfg.universe, from, to).await?;
                    let rep =
                        ingest::ingest_macro(&store, Some(&fmp), Some(&fred), from, to).await?;
                    let e = ingest::ingest_earnings(&store, &fmp, &cfg.universe, from, to).await?;
                    println!(
                        "  ingested {n} bar-rows │ {e} earnings │ {}",
                        rep.summary_line()
                    );
                }
                "mock" => {
                    let mock = MockProvider;
                    let n =
                        ingest::ingest_bars(&store, &mock, "mock", &cfg.universe, from, to).await?;
                    let rep = ingest::ingest_macro_via(&store, &mock, from, to).await?;
                    let e = ingest::ingest_earnings(&store, &mock, &cfg.universe, from, to).await?;
                    println!(
                        "  ingested {n} bar-rows │ {e} earnings │ {}",
                        rep.summary_line()
                    );
                }
                other => bail!("unknown provider '{other}'"),
            },
            Step::Search => {
                let risk = search_cmd::RiskArgs {
                    stop: args.stop.clone(),
                    target1: args.target1.clone(),
                    target2: args.target2.clone(),
                    lock_risk: args.lock_risk,
                };
                search_cmd::run_search(
                    cfg,
                    args.generations,
                    args.per_gen,
                    args.horizon.clone(),
                    Some(from.to_string()),
                    Some(to.to_string()),
                    false,
                    risk,
                )
                .await?;
            }
            Step::Signals => {
                signals_cmd::run_signals(cfg, args.horizon.clone(), false).await?;
            }
            Step::Journal => {
                // Journaling is performed inline by `signals --journal`; in the nightly
                // loop we resolve open paper trades against the freshly-ingested bars.
                signals_cmd::run_signals(cfg, args.horizon.clone(), true).await?;
            }
            Step::Monitor => {
                let report = se_monitor::run_daily(&store).await?;
                println!(
                    "  monitor fired {} event(s) [divergence={} drawdown={} calibration={} staleness={} ood={}]",
                    report.total(),
                    report.divergence,
                    report.drawdown,
                    report.calibration,
                    report.staleness,
                    report.regime_ood,
                );
            }
            Step::Changelog => {
                let week = se_monitor::weekly_changelog(&store).await?;
                println!("  weekly changelog written for week of {week}");
            }
        }
    }

    println!("\n✓ nightly loop complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::choose_provider;

    #[test]
    fn defaults_to_mock_without_fmp_key() {
        // The critical branch: no key => never pick fmp (which would crash on a live call).
        assert_eq!(choose_provider(None, false), "mock");
    }

    #[test]
    fn defaults_to_fmp_when_configured() {
        assert_eq!(choose_provider(None, true), "fmp");
    }

    #[test]
    fn explicit_provider_overrides_and_lowercases() {
        assert_eq!(choose_provider(Some("MOCK"), true), "mock");
        assert_eq!(choose_provider(Some("Fmp"), false), "fmp");
    }
}
