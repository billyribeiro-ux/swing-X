//! `se` — the swing-X operator CLI.
//!
//! P1 commands: `migrate` and `scan` (ingest a window + macro + run the Layer-0
//! tradeability gate and Layer-1 regime labeling). P2 adds `regime-sanity-check`
//! (cross-check regime labels against known historical events). Later phases add
//! `inject-leak-test` (P4) and `promote --dry-run` (P5).

mod ingest;
mod leak_test;
mod sanity;
mod search_cmd;
mod signals_cmd;

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Datelike, Duration, NaiveDate, TimeZone, Utc, Weekday};
use clap::{Args, Parser, Subcommand};

use se_config::AppConfig;
use se_core::{AsOf, DecisionTs, Layer, LeadTimeTag, Ticker};
use se_features::indicators::{dollar_adv, realized_vol};
use se_features::{
    EventOverlay, FeatureContext, FeatureModule, LocationModule, TradeabilityGate,
    TradeabilityInput, TradeabilityModule, TriggerModule,
};
use se_provider::{
    DataProvider, FmpProvider, FredProvider, MockProvider, NullProprietary, ProviderKind,
};
use se_regime::RegimeEngine;
use se_store::{FeatureWrite, Store};

/// Session-close UTC timestamp convention shared across the repo (21:00 UTC).
pub(crate) fn session_close(date: NaiveDate) -> DateTime<Utc> {
    Utc.from_utc_datetime(&date.and_hms_opt(21, 0, 0).expect("21:00:00 valid"))
}

#[derive(Parser)]
#[command(
    name = "se",
    about = "swing-X self-learning swing scanner — operator CLI"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Apply database migrations.
    Migrate,
    /// Ingest a window of data and run the Layer-0 tradeability gate.
    Scan(ScanArgs),
    /// (P4) Run the deliberate-leakage checkpoint.
    InjectLeakTest,
    /// (P2) Cross-check regime labels against known historical events.
    RegimeSanityCheck,
    /// (P5) Run the genome search/mutation loop on the OOS scoreboard.
    Search(SearchArgs),
    /// (P5) Print the full promotion-gate evaluation for top candidates (runs a short search).
    Promote(PromoteArgs),
    /// (P7) Generate + print current executable signals from promoted strategies.
    Signals(SignalsArgs),
}

#[derive(Args)]
struct SearchArgs {
    /// Number of generations to evolve.
    #[arg(long, default_value_t = 2)]
    generations: u32,
    /// Genomes evaluated per generation.
    #[arg(long, default_value_t = 12)]
    per_gen: usize,
    /// Provider hint (accepted for parity with `scan`; search reads from the store).
    #[arg(long)]
    provider: Option<String>,
    /// Horizon override (e.g. `swing`, `day`). Defaults to SE_HORIZON / config (P8 axis).
    #[arg(long)]
    horizon: Option<String>,
    /// Inclusive window start (YYYY-MM-DD). Default: 730 days before `to`.
    #[arg(long)]
    from: Option<String>,
    /// Inclusive window end (YYYY-MM-DD). Default: today.
    #[arg(long)]
    to: Option<String>,
}

#[derive(Args)]
struct PromoteArgs {
    /// Print the gate without persisting promotions beyond the search itself.
    #[arg(long, default_value_t = true)]
    dry_run: bool,
    #[arg(long, default_value_t = 2)]
    generations: u32,
    #[arg(long, default_value_t = 12)]
    per_gen: usize,
    #[arg(long)]
    horizon: Option<String>,
    #[arg(long)]
    from: Option<String>,
    #[arg(long)]
    to: Option<String>,
}

#[derive(Args)]
struct SignalsArgs {
    /// Provider hint (accepted for parity; signals read from the store).
    #[arg(long)]
    provider: Option<String>,
    /// Horizon override (P8 axis). Defaults to SE_HORIZON / config.
    #[arg(long)]
    horizon: Option<String>,
    /// Also open + resolve a paper trade per signal and print the realized fill.
    #[arg(long, default_value_t = false)]
    journal: bool,
}

#[derive(Args)]
struct ScanArgs {
    /// `fmp` or `mock`. Defaults to `fmp` when FMP_API_KEY is configured, else `mock`.
    #[arg(long)]
    provider: Option<String>,
    /// Inclusive start date (YYYY-MM-DD). Default: 120 days before `to`.
    #[arg(long)]
    from: Option<String>,
    /// Inclusive end date (YYYY-MM-DD). Default: most recent weekday.
    #[arg(long)]
    to: Option<String>,
    /// Lookback bars for liquidity / volatility windows.
    #[arg(long, default_value_t = 20)]
    lookback: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let cli = Cli::parse();
    let cfg = AppConfig::from_env().context("load config")?;

    match cli.cmd {
        Cmd::Migrate => {
            let store = Store::connect(&cfg.database_url)
                .await
                .context("connect db")?;
            store.migrate().await.context("migrate")?;
            println!("✓ migrations applied");
        }
        Cmd::Scan(args) => scan(&cfg, args).await?,
        Cmd::InjectLeakTest => leak_test::run().await?,
        Cmd::RegimeSanityCheck => {
            let store = Store::connect(&cfg.database_url)
                .await
                .context("connect db")?;
            store.migrate().await.context("migrate")?;
            sanity::run(&store, &[Ticker::Spy, Ticker::Qqq]).await?;
        }
        Cmd::Search(args) => {
            search_cmd::run_search(
                &cfg,
                args.generations,
                args.per_gen,
                args.horizon,
                args.from,
                args.to,
                false,
            )
            .await?;
        }
        Cmd::Promote(args) => {
            search_cmd::run_search(
                &cfg,
                args.generations,
                args.per_gen,
                args.horizon,
                args.from,
                args.to,
                args.dry_run,
            )
            .await?;
        }
        Cmd::Signals(args) => {
            signals_cmd::run_signals(&cfg, args.horizon, args.journal).await?;
        }
    }
    Ok(())
}

fn provider_source(kind: ProviderKind) -> &'static str {
    match kind {
        ProviderKind::Mock => "mock",
        ProviderKind::Fmp => "fmp",
        ProviderKind::Fred => "fred",
        ProviderKind::Proprietary => "proprietary",
    }
}

fn build_provider(name: Option<&str>, cfg: &AppConfig) -> Result<Box<dyn DataProvider>> {
    let chosen = name.map(|s| s.to_ascii_lowercase()).unwrap_or_else(|| {
        if cfg.fmp_configured {
            "fmp".into()
        } else {
            "mock".into()
        }
    });
    match chosen.as_str() {
        "mock" => Ok(Box::new(MockProvider)),
        "fmp" => Ok(Box::new(
            FmpProvider::from_env().context("init FMP provider")?,
        )),
        other => bail!("unknown provider '{other}' (expected 'fmp' or 'mock')"),
    }
}

fn last_weekday(mut d: NaiveDate) -> NaiveDate {
    while matches!(d.weekday(), Weekday::Sat | Weekday::Sun) {
        d = d.pred_opt().unwrap();
    }
    d
}

fn parse_date(s: &str) -> Result<NaiveDate> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .with_context(|| format!("bad date '{s}' (want YYYY-MM-DD)"))
}

async fn scan(cfg: &AppConfig, args: ScanArgs) -> Result<()> {
    let provider = build_provider(args.provider.as_deref(), cfg)?;
    let source = provider_source(provider.kind());

    let to = match args.to {
        Some(s) => parse_date(&s)?,
        None => last_weekday(Utc::now().date_naive()),
    };
    // Default to ~24 months so there's enough history to compute regime
    // percentiles/trends and to label a meaningful window.
    let from = match args.from {
        Some(s) => parse_date(&s)?,
        None => to - Duration::days(730),
    };

    let store = Store::connect(&cfg.database_url)
        .await
        .context("connect db")?;
    store.migrate().await.context("migrate")?;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        " swing-X scan │ provider={} │ horizon={} │ {} → {}",
        source,
        cfg.horizon.horizon.as_str(),
        from,
        to
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // ---- ingest ----------------------------------------------------------
    let mut latest: HashMap<Ticker, chrono::DateTime<Utc>> = HashMap::new();
    let mut total_bars = 0u64;
    for &t in &cfg.universe {
        let bars = provider
            .daily_bars(t, from, to)
            .await
            .with_context(|| format!("fetch bars for {t}"))?;
        if bars.is_empty() {
            tracing::warn!(ticker = %t, "no bars returned; skipping");
            continue;
        }
        let last_ts = bars.last().unwrap().ts;
        latest.insert(t, last_ts);
        total_bars += store.upsert_bars(&bars, "daily", source).await?;

        // ETF profile -> raw tradeability features at the latest decision bar.
        if let Some(profile) = provider.etf_profile(t).await.unwrap_or(None) {
            let decision = DecisionTs::new(last_ts);
            let as_of = AsOf::new(last_ts);
            let mut fws = Vec::new();
            let mut push = |key: &str, val: f64| {
                fws.push(FeatureWrite {
                    ticker: t,
                    feature_key: key.to_string(),
                    layer: Layer::Tradeability,
                    decision_ts: decision,
                    as_of,
                    value: val,
                    lead_time: LeadTimeTag::EndOfDay,
                    source: source.to_string(),
                });
            };
            if let Some(aum) = profile.aum {
                push("tradeability.aum_raw", aum);
            }
            if let Some(v) = profile.avg_volume {
                push("tradeability.avg_volume_raw", v);
            }
            if let Some(h) = profile.holdings_count {
                push("tradeability.holdings_raw", h as f64);
            }
            store.insert_features(&fws).await?;
        }
    }
    println!(
        "ingested {total_bars} bar-rows across {} names",
        latest.len()
    );

    // ---- macro ingest (market-wide, once per run) ------------------------
    // Pull every MacroSeries from its preferred provider (FMP or FRED). Only the
    // live HTTP providers serve real macro; the mock provider serves synthetic
    // macro so offline runs still populate the regime store.
    let macro_report = match provider.kind() {
        ProviderKind::Fmp => {
            let fmp = FmpProvider::from_env().context("init FMP provider for macro")?;
            let fred = FredProvider::from_env();
            ingest::ingest_macro(&store, Some(&fmp), Some(&fred), from, to).await?
        }
        _ => {
            // Mock (or other) provider: ingest macro through the same DataProvider
            // so the regime layer has inputs offline.
            ingest::ingest_macro_via(&store, provider.as_ref(), from, to).await?
        }
    };
    println!("{}\n", macro_report.summary_line());

    // ---- regime labeling at the latest bar (Layer-1) ---------------------
    let regime_engine = RegimeEngine::default();
    let mut regime_line = String::new();
    for &t in &cfg.universe {
        let Some(&last_ts) = latest.get(&t) else {
            continue;
        };
        if let Some(a) = regime_engine
            .assess_at(&store, t, DecisionTs::new(last_ts))
            .await?
        {
            regime_line.push_str(&format!(
                "{}={}({:.0}%)  ",
                t.as_str(),
                a.label.as_str(),
                a.confidence * 100.0
            ));
        }
    }
    if !regime_line.is_empty() {
        println!("regime @ latest bar: {}\n", regime_line.trim_end());
    }

    // ---- tradeability gate (real pipeline: PIT -> feature module -> store) --
    let gate = TradeabilityGate::default();
    let module = TradeabilityModule::default();
    let prop = NullProprietary;

    println!(
        "{:<6} {:>12} {:>10} {:>7} {:>7}  NOTES",
        "NAME", "ADV($M)", "AUM($B)", "SCORE", "GATE"
    );
    println!("{}", "-".repeat(70));

    let mut passed = 0usize;
    for &t in &cfg.universe {
        let Some(&last_ts) = latest.get(&t) else {
            continue;
        };
        let decision = DecisionTs::new(last_ts);
        let pit = store.pit(t, decision);

        // Persist the module's derived features (exercises the PIT pipeline).
        let ctx = FeatureContext::new(&pit, &prop, cfg.horizon);
        let feats = module.compute(&ctx).await?;
        let writes: Vec<FeatureWrite> = feats
            .iter()
            .map(|f| FeatureWrite::from_feature(t, decision, f))
            .collect();
        store.insert_features(&writes).await?;

        // Reconstruct the input for display.
        let bars = pit.bars("daily", args.lookback as i64 + 2).await?;
        let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
        let aum = pit.feature_value("tradeability.aum_raw").await?;
        let holdings = pit
            .feature_value("tradeability.holdings_raw")
            .await?
            .map(|v| v as i64);
        let input = TradeabilityInput {
            symbol: t.to_string(),
            dollar_adv: dollar_adv(&bars, args.lookback),
            aum,
            holdings_count: holdings,
            abs_gex: None,
            options_oi: None,
            realized_vol: realized_vol(&closes, args.lookback),
        };
        let score = gate.evaluate(&input);
        if score.passed {
            passed += 1;
        }
        let note = if score.components.large_hand_observed {
            String::new()
        } else {
            "GEX stubbed (proxy)".to_string()
        };
        println!(
            "{:<6} {:>12.1} {:>10.2} {:>7.3} {:>7}  {}",
            input.symbol,
            input.dollar_adv / 1e6,
            input.aum.unwrap_or(0.0) / 1e9,
            score.score,
            if score.passed { "PASS" } else { "REJECT" },
            note
        );
    }
    println!(
        "\n{passed}/{} universe names pass the tradeability gate.\n",
        latest.len()
    );

    // ---- P3 feature layers: location / trigger / event --------------------
    // Mirror the layer0/layer1 pipeline: for each ticker's latest decision bar
    // run the three modules through the PIT-safe FeatureContext and persist their
    // features. EventOverlay is calendar-only (no leakage); Location/Trigger read
    // PIT-safe bars (Trigger also reads SPY + the universe via bars_for).
    let location = LocationModule::new();
    let trigger = TriggerModule::new();
    let events = EventOverlay::new();
    let mut loc_count = 0usize;
    let mut trig_count = 0usize;
    let mut evt_count = 0usize;
    for &t in &cfg.universe {
        let Some(&last_ts) = latest.get(&t) else {
            continue;
        };
        let decision = DecisionTs::new(last_ts);
        let pit = store.pit(t, decision);
        let ctx = FeatureContext::new(&pit, &prop, cfg.horizon);

        for (module, counter) in [
            (&location as &dyn FeatureModule, &mut loc_count),
            (&trigger as &dyn FeatureModule, &mut trig_count),
            (&events as &dyn FeatureModule, &mut evt_count),
        ] {
            let feats = module.compute(&ctx).await?;
            let writes: Vec<FeatureWrite> = feats
                .iter()
                .map(|f| FeatureWrite::from_feature(t, decision, f))
                .collect();
            store.insert_features(&writes).await?;
            *counter += writes.len();
        }
    }
    println!(
        "P3 feature layers @ latest bar: location={loc_count} │ trigger={trig_count} │ event={evt_count} (across {} names)\n",
        latest.len()
    );

    // ---- illustrative rejects (off-universe candidates) -------------------
    println!("Rejected candidates (illustrative — outside the curated universe):");
    println!("{}", "-".repeat(70));
    for cand in illustrative_rejects() {
        let s = gate.evaluate(&cand);
        println!("  {:<6} score={:.3} → REJECT", cand.symbol, s.score);
        for r in &s.reasons {
            println!("      • {r}");
        }
    }

    Ok(())
}

/// Five deliberately-marginal off-universe names, to demonstrate the gate's
/// rejection reasons (the curated 10 ETFs all pass).
fn illustrative_rejects() -> Vec<TradeabilityInput> {
    let mk = |sym: &str, adv: f64, aum: Option<f64>, holdings: Option<i64>| TradeabilityInput {
        symbol: sym.into(),
        dollar_adv: adv,
        aum,
        holdings_count: holdings,
        abs_gex: None,
        options_oi: None,
        realized_vol: 0.35,
    };
    vec![
        mk("THNL", 3_000_000.0, Some(40_000_000.0), Some(25)),
        mk("LEV3", 8_000_000.0, Some(120_000_000.0), Some(5)),
        mk("MICR", 1_500_000.0, Some(30_000_000.0), Some(60)),
        mk("NOAU", 20_000_000.0, None, Some(80)),
        mk("EDGE", 45_000_000.0, Some(300_000_000.0), Some(80)),
    ]
}
