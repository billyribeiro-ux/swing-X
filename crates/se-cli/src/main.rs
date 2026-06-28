//! `se` — the swing-X operator CLI.
//!
//! P1 commands: `migrate` and `scan` (ingest a window + run the Layer-0
//! tradeability gate, showing scores and rejects-with-reasons). Later phases add
//! `inject-leak-test` (P4), `regime-sanity-check` (P2), `promote --dry-run` (P5).

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use chrono::{Datelike, Duration, NaiveDate, Utc, Weekday};
use clap::{Args, Parser, Subcommand};

use se_config::AppConfig;
use se_core::{AsOf, DecisionTs, Layer, LeadTimeTag, Ticker};
use se_features::indicators::{dollar_adv, realized_vol};
use se_features::{
    FeatureContext, FeatureModule, TradeabilityGate, TradeabilityInput, TradeabilityModule,
};
use se_provider::{DataProvider, FmpProvider, MockProvider, NullProprietary, ProviderKind};
use se_store::{FeatureWrite, Store};

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
        Cmd::InjectLeakTest => {
            println!("inject-leak-test is implemented in phase P4 (validation harness).");
        }
        Cmd::RegimeSanityCheck => {
            println!("regime-sanity-check is implemented in phase P2 (regime layer).");
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
    let from = match args.from {
        Some(s) => parse_date(&s)?,
        None => to - Duration::days(120),
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
        "ingested {total_bars} bar-rows across {} names\n",
        latest.len()
    );

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
