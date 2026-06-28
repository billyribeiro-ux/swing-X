//! `se search` — run the P5 genome search/mutation loop on the OOS scoreboard, persist the
//! population + scores, and print a leaderboard (ranked ONLY on out-of-sample metrics).
//!
//! `se promote --dry-run` — print the full promotion-gate evaluation (DSR>0, PBO<0.5, cost-aware
//! OOS expectancy>0, >=2 positive regimes) for the top candidates, pass/fail per condition.

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Duration, NaiveDate, TimeZone, Utc};

use se_config::AppConfig;
use se_core::{Horizon, HorizonProfile, RiskModel};
use se_search::{PopulationManager, ScoreConfig, SearchConfig};
use se_store::Store;
use se_validation::ValidationHarness;

/// Operator overrides for the ground-rule risk geometry, parsed from the CLI flags. `None`
/// fields fall back to the config / horizon default.
#[derive(Debug, Clone, Default)]
pub struct RiskArgs {
    pub stop: Option<String>,
    pub target1: Option<String>,
    pub target2: Option<String>,
    pub lock_risk: bool,
}

/// Resolve the operator's ground-rule [`RiskModel`] from the config default + CLI overrides.
fn resolve_risk(cfg: &AppConfig, args: &RiskArgs) -> Result<RiskModel> {
    let base = cfg.risk;
    let stop = match args.stop.as_deref() {
        Some(s) => s.parse().with_context(|| format!("bad --stop '{s}'"))?,
        None => base.stop,
    };
    let target1 = match args.target1.as_deref() {
        Some(s) => s.parse().with_context(|| format!("bad --target1 '{s}'"))?,
        None => base.target1,
    };
    let target2 = match args.target2.as_deref() {
        Some(s) if s.trim().eq_ignore_ascii_case("none") => None,
        Some(s) => Some(s.parse().with_context(|| format!("bad --target2 '{s}'"))?),
        None => base.target2,
    };
    Ok(RiskModel::new(stop, target1, target2))
}

/// Resolve the active horizon profile: `--horizon` flag overrides `SE_HORIZON`/config.
pub fn resolve_profile(cfg: &AppConfig, horizon: Option<&str>) -> Result<HorizonProfile> {
    match horizon {
        Some(h) => {
            let parsed: Horizon = h.parse().with_context(|| format!("bad horizon '{h}'"))?;
            Ok(HorizonProfile::for_horizon(parsed))
        }
        None => Ok(cfg.horizon),
    }
}

fn session_close(date: NaiveDate) -> DateTime<Utc> {
    Utc.from_utc_datetime(&date.and_hms_opt(21, 0, 0).expect("21:00:00 valid"))
}

fn parse_date(s: &str) -> Result<NaiveDate> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d").with_context(|| format!("bad date '{s}'"))
}

/// Ensure the ML worker is reachable; bail with a clear message otherwise.
async fn require_worker(harness: &ValidationHarness) -> Result<String> {
    let base = harness.client().base_url().to_string();
    match harness.client().health().await {
        Ok(h) if h.status == "ok" => Ok(base),
        _ => bail!(
            "ML worker not reachable at {base}. Start it:\n  \
             cd ml-worker && uv run uvicorn se_ml.server:app --port 8088"
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn run_search(
    cfg: &AppConfig,
    generations: u32,
    per_gen: usize,
    horizon: Option<String>,
    from: Option<String>,
    to: Option<String>,
    promote_dry_run: bool,
    risk_args: RiskArgs,
) -> Result<()> {
    let profile = resolve_profile(cfg, horizon.as_deref())?;
    let risk = resolve_risk(cfg, &risk_args)?;
    // The CLI flag wins; otherwise the config's SE_LOCK_RISK.
    let lock_risk = risk_args.lock_risk || cfg.lock_risk;

    let store = Store::connect(&cfg.database_url)
        .await
        .context("connect db")?;
    store.migrate().await.context("migrate")?;

    let harness = ValidationHarness::new(
        se_mlclient::MlClient::from_env().map_err(|e| anyhow::anyhow!("ml client: {e}"))?,
        std::env::temp_dir().join("se_search"),
    );
    let worker = require_worker(&harness).await?;

    let to_date = match to {
        Some(s) => parse_date(&s)?,
        None => Utc::now().date_naive(),
    };
    let from_date = match from {
        Some(s) => parse_date(&s)?,
        None => to_date - Duration::days(730),
    };

    let mut search_cfg = SearchConfig::new(profile, cfg.universe.clone());
    search_cfg.from = session_close(from_date);
    search_cfg.to = session_close(to_date);
    search_cfg.score = ScoreConfig::default();
    search_cfg.risk = risk;
    search_cfg.lock_risk = lock_risk;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        " se search │ worker={worker} │ horizon={} │ {} → {} │ gen={} per_gen={}",
        profile.horizon.as_str(),
        from_date,
        to_date,
        generations,
        per_gen
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "ranking key: OUT-OF-SAMPLE ONLY (oos_expectancy, dsr) — in-sample fit is never ranked."
    );
    println!(
        "operator ground-rule risk: {} │ {}",
        risk.describe(),
        if lock_risk {
            "LOCKED (conditions optimized; risk geometry fixed)"
        } else {
            "EXPLORED (search optimizes risk geometry on the OOS scoreboard)"
        }
    );

    println!("materializing per-bar feature windows across the universe (one-time) ...");
    use std::io::Write as _;
    std::io::stdout().flush().ok();
    let manager = PopulationManager::new(&store, &harness, search_cfg.clone()).await?;
    println!(
        "  done: {} features in catalog, {} windows",
        manager.catalog().len(),
        manager.windows().len()
    );
    if manager.catalog().is_empty() {
        bail!("no features in catalog — run `se scan` first to populate bars/features");
    }

    let outcome = manager.evolve(generations, per_gen).await?;
    print_leaderboard(&outcome, 12);

    if promote_dry_run {
        print_promotion_gate(&outcome, 8);
    }

    Ok(())
}

/// Print the leaderboard, best OOS first.
fn print_leaderboard(outcome: &se_search::EvolveOutcome, n: usize) {
    let board = outcome.leaderboard(n);
    println!();
    println!(
        "LEADERBOARD (horizon={}, top {} by cost-aware OOS expectancy):",
        outcome.profile.horizon.as_str(),
        n
    );
    println!("{}", "-".repeat(110));
    println!(
        "{:>3}  {:>10} {:>8} {:>8} {:>7} {:>6} {:>5}  {:<6}  genome",
        "#", "oos_exp_R", "dsr", "pbo", "pf", "reg+", "gate", "n"
    );
    println!("{}", "-".repeat(110));
    if board.is_empty() {
        println!("  (no genome produced enough labeled entries to validate — see logs)");
    }
    for (i, ev) in board.iter().enumerate() {
        let s = ev.score.as_ref().unwrap();
        println!(
            "{:>3}  {:>+10.4} {:>+8.3} {:>8.3} {:>7.2} {:>6} {:>5}  {:<6}  {}",
            i + 1,
            s.oos_expectancy_cost_aware,
            s.dsr,
            s.pbo,
            s.profit_factor,
            s.n_regimes_positive,
            if s.passed_gate { "PASS" } else { "fail" },
            ev.n_entries,
            ev.strategy.genome.describe(),
        );
    }
    println!("{}", "-".repeat(110));
    println!(
        "promoted (gate-passing): {} │ scored: {} │ evaluated total: {}",
        outcome.n_promoted(),
        outcome
            .evaluated
            .iter()
            .filter(|e| e.score.is_some())
            .count(),
        outcome.evaluated.len(),
    );
}

/// Print the full promotion-gate evaluation per condition for the top candidates.
fn print_promotion_gate(outcome: &se_search::EvolveOutcome, n: usize) {
    println!();
    println!("PROMOTION GATE (dry-run) — top {n} candidates, all four conditions:");
    println!("  required: DSR>0  AND  PBO<0.5  AND  cost-aware OOS expectancy>0  AND  >=2 positive regimes");
    println!("{}", "-".repeat(110));
    let board = outcome.leaderboard(n);
    if board.is_empty() {
        println!("  (nothing scored — no candidates to evaluate)");
        return;
    }
    for (i, ev) in board.iter().enumerate() {
        let s = ev.score.as_ref().unwrap();
        let g = &s.gate;
        let mark = |ok: bool| if ok { "✓" } else { "✗" };
        println!(
            "{:>2}. {}  [{}]",
            i + 1,
            ev.strategy.genome.describe(),
            if g.passed { "PROMOTE" } else { "REJECT" }
        );
        println!(
            "     {} DSR>0           dsr = {:+.4}",
            mark(g.dsr_ok),
            s.dsr
        );
        println!("     {} PBO<0.5         pbo = {:.4}", mark(g.pbo_ok), s.pbo);
        println!(
            "     {} OOS_exp>0       oos_expectancy_cost_aware = {:+.4} R",
            mark(g.expectancy_ok),
            s.oos_expectancy_cost_aware
        );
        println!(
            "     {} regimes>=2      n_regimes_positive = {}",
            mark(g.regime_ok),
            s.n_regimes_positive
        );
        if !g.reasons.is_empty() {
            for r in &g.reasons {
                println!("       • {r}");
            }
        }
    }
    println!("{}", "-".repeat(110));
}
