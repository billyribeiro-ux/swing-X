//! `se test-era-score` — the selection-bias meter (audit roadmap #1, second half).
//!
//! Scores each PROMOTED strategy EXACTLY ONCE on the locked out-of-time test era
//! (`SE_TEST_FROM`/`SE_TEST_TO`): the worker fits strictly PRE-era and measures precision /
//! cost-aware expectancy IN-era via its boundary-parameterized forward holdout. Results land in
//! `test_era_scores`, whose primary key `(strategy_id, test_from, test_to)` enforces the
//! once-only discipline — an already-scored strategy is refused, so the era can never be
//! iterated against.
//!
//! REPORT-ONLY by construction: this command never updates strategy status, never writes
//! `oos_scores`, and nothing in the engine reads `test_era_scores` for ranking, the gate,
//! survivor selection, or nightly. Its sole output is the OPTIMISM GAP
//! (`precision_oos − precision_test`) — the honest measure of how much the search's reported
//! OOS numbers flatter themselves through selection.

use anyhow::{bail, Context, Result};
use chrono::Duration;

use se_config::AppConfig;
use se_search::persist::{latest_oos_score, load_promoted};
use se_search::{PopulationManager, ScoreConfig, SearchConfig, MIN_ENTRIES_TO_VALIDATE};
use se_store::Store;
use se_validation::{EvaluateOptions, ValidationHarness};

use crate::search_cmd::resolve_profile;
use crate::session_close;

#[derive(clap::Args)]
pub struct TestEraArgs {
    /// Horizon override (P8 axis). Defaults to SE_HORIZON / config.
    #[arg(long)]
    pub horizon: Option<String>,
    /// Days of pre-era history the training fit may use (window = [test_from - N, test_to]).
    #[arg(long, default_value_t = 2200)]
    pub history_days: i64,
}

pub async fn run(cfg: &AppConfig, args: TestEraArgs) -> Result<()> {
    let Some((test_from, test_to)) = cfg.test_era() else {
        bail!(
            "no test era reserved; set SE_TEST_FROM/SE_TEST_TO (YYYY-MM-DD) to the locked \
             out-of-time window this command may score — exactly once per strategy"
        );
    };
    let profile = resolve_profile(cfg, args.horizon.as_deref())?;

    let store = Store::connect(&cfg.database_url)
        .await
        .context("connect db")?;
    store.migrate().await.context("migrate")?;

    let harness = ValidationHarness::new(
        se_mlclient::MlClient::from_env().map_err(|e| anyhow::anyhow!("ml client: {e}"))?,
        std::env::temp_dir().join("se_test_era"),
    );

    let promoted = load_promoted(&store, profile.horizon.as_str(), cfg.scanner).await?;
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        " se test-era-score │ {} │ era {test_from} → {test_to} │ promoted={} │ ONCE-ONLY, report-only",
        cfg.scanner.label(),
        promoted.len()
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    if promoted.is_empty() {
        println!("no promoted strategies for this horizon/scanner — nothing to score.");
        return Ok(());
    }

    // Deliberately construct the dataset WITHOUT the test-era firewall: this command is the
    // single sanctioned read of the reserved era. The worker still fits strictly PRE-era —
    // `forward_boundary_ts = test_from` — so in-era rows are only ever MEASURED, never trained on.
    let mut era_cfg = SearchConfig::new(profile, cfg.universe.clone());
    era_cfg.from = session_close(test_from) - Duration::days(args.history_days.max(1));
    era_cfg.to = session_close(test_to);
    era_cfg.scanner = cfg.scanner;
    era_cfg.test_era = None; // the sanctioned read (see module docs)
    let manager = PopulationManager::new(&store, &harness, era_cfg).await?;

    let boundary = session_close(test_from).to_rfc3339();
    let shape = ScoreConfig::default();
    let mut gaps: Vec<f64> = Vec::new();

    println!(
        "{:<10} {:>9} {:>9} {:>7} {:>9} {:>9} {:>7} {:>7}",
        "STRATEGY", "prec_oos", "prec_TEST", "gap", "exp_oos", "exp_TEST", "n_oos", "n_TEST"
    );
    println!("{}", "-".repeat(78));

    for strategy in &promoted {
        let sid = strategy.id.inner();
        let short = &sid.to_string()[..8];

        // Once-only: refuse to re-score (and refuse to even re-read the era) if a row exists.
        let already: Option<(i32,)> = se_store::sqlx::query_as(
            "SELECT 1 FROM test_era_scores WHERE strategy_id = $1 AND test_from = $2 AND test_to = $3",
        )
        .bind(sid)
        .bind(test_from)
        .bind(test_to)
        .fetch_optional(store.pool())
        .await
        .map_err(|e| anyhow::anyhow!("query test_era_scores: {e}"))?;
        if already.is_some() {
            println!("{short:<10} already scored on this era — refusing to re-score");
            continue;
        }

        let (rows, n_entries) = manager.dataset_for(&strategy.genome);
        if n_entries < MIN_ENTRIES_TO_VALIDATE {
            println!("{short:<10} too few labeled entries ({n_entries}) — skipped");
            continue;
        }
        let n_groups = shape.n_groups.min(rows.len() as u32).max(2);
        let opts = EvaluateOptions {
            n_groups,
            k_test_groups: shape.k_test_groups.min(n_groups.saturating_sub(1)).max(1),
            n_trials: shape.n_trials,
            n_search_trials: 1, // a single once-only evaluation, not a search
            forward_boundary_ts: Some(boundary.clone()),
        };
        let name = format!("test_era_{sid}.parquet");
        let outcome = match harness.evaluate_with(&rows, &profile, &name, &opts).await {
            Ok(o) => o,
            Err(e) => {
                println!("{short:<10} validation failed ({e}) — skipped");
                continue;
            }
        };
        let v = &outcome.validation;
        let (p_test, e_test, n_test) = (v.precision_forward, v.expectancy_forward, v.n_forward);

        se_store::sqlx::query(
            "INSERT INTO test_era_scores \
                (strategy_id, test_from, test_to, precision_test, expectancy_test, n_test) \
             VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT DO NOTHING",
        )
        .bind(sid)
        .bind(test_from)
        .bind(test_to)
        .bind(p_test)
        .bind(e_test)
        .bind(n_test as i32)
        .execute(store.pool())
        .await
        .map_err(|e| anyhow::anyhow!("insert test_era_scores: {e}"))?;

        let stored = latest_oos_score(&store, strategy.id).await?;
        let p_oos = stored.as_ref().and_then(|s| s.precision_oos);
        let e_oos = stored.as_ref().and_then(|s| s.oos_expectancy_cost_aware);
        let n_oos = stored.as_ref().and_then(|s| s.n_acted);
        if let Some(p) = p_oos {
            gaps.push(p - p_test);
        }
        println!(
            "{short:<10} {:>9} {:>9.3} {:>7} {:>9} {:>9.3} {:>7} {:>7}",
            p_oos
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "—".into()),
            p_test,
            p_oos
                .map(|v| format!("{:+.3}", v - p_test))
                .unwrap_or_else(|| "—".into()),
            e_oos
                .map(|v| format!("{v:+.3}"))
                .unwrap_or_else(|| "—".into()),
            e_test,
            n_oos.map(|v| v.to_string()).unwrap_or_else(|| "—".into()),
            n_test,
        );
    }

    println!("{}", "-".repeat(78));
    if gaps.is_empty() {
        println!("no comparable strategies scored (nothing new, or no stored OOS precision).");
    } else {
        let mean_gap = gaps.iter().sum::<f64>() / gaps.len() as f64;
        println!(
            "MEAN OPTIMISM GAP (precision_oos − precision_test) over {} strategies: {:+.3}",
            gaps.len(),
            mean_gap
        );
        println!(
            "A large positive gap = the search's reported OOS precision flatters itself through \
             selection; the TEST column is the number to trust."
        );
    }
    Ok(())
}
