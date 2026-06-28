//! `se trades` — replay each PROMOTED strategy's backtest and journal the actual
//! per-trade winners and losers (entry, exit, outcome, R), so you can SEE the
//! win/loss record rather than just the aggregate.
//!
//! These are journaled with `mode=backtest` (distinct from live `paper`/`live`),
//! and surface on the dashboard's Journal page. The configurable stop directly
//! decides each win vs loss — change `--stop`/re-run the search and the split shifts.

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, TimeZone, Utc};

use se_config::AppConfig;
use se_core::{HorizonProfile, Side};
use se_search::{backtest, build_window, load_promoted};
use se_store::sqlx;
use se_store::Store;
use uuid::Uuid;

use crate::search_cmd::resolve_profile;

fn session_close(d: NaiveDate) -> DateTime<Utc> {
    Utc.from_utc_datetime(&d.and_hms_opt(21, 0, 0).expect("21:00:00 valid"))
}

fn parse_date(s: &str) -> Result<NaiveDate> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d").with_context(|| format!("bad date '{s}'"))
}

fn side_str(side: Side) -> &'static str {
    match side {
        Side::Long => "long",
        Side::Short => "short",
    }
}

pub async fn run(
    cfg: &AppConfig,
    horizon: Option<String>,
    from: Option<String>,
    to: Option<String>,
) -> Result<()> {
    let profile: HorizonProfile = resolve_profile(cfg, horizon.as_deref())?;
    let store = Store::connect(&cfg.database_url).await.context("connect db")?;
    store.migrate().await.context("migrate")?;

    let to_date = match to {
        Some(s) => parse_date(&s)?,
        None => Utc::now().date_naive(),
    };
    let from_date = match from {
        Some(s) => parse_date(&s)?,
        None => to_date - chrono::Duration::days(730),
    };
    let (from_ts, to_ts) = (session_close(from_date), session_close(to_date));

    let promoted = load_promoted(&store, profile.horizon.as_str()).await?;
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        " se trades │ horizon={} │ {} → {} │ promoted strategies: {}",
        profile.horizon.as_str(),
        from_date,
        to_date,
        promoted.len()
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    if promoted.is_empty() {
        println!("No promoted strategies for this horizon — run `se search` (multi-regime window) first.");
        return Ok(());
    }

    // Idempotent: clear prior backtest trades so a re-run replaces them.
    sqlx::query("DELETE FROM trades_journal WHERE mode = 'backtest'")
        .execute(store.pool())
        .await
        .context("clear prior backtest trades")?;

    let cost = profile.cost.round_trip_frac();
    let mut total = 0usize;
    let mut wins = 0usize;
    let mut gross_win = 0.0f64;
    let mut gross_loss = 0.0f64;
    let mut sum_r = 0.0f64;
    let mut sample: Vec<(String, &'static str, f64, f64, &'static str)> = Vec::new();

    for strat in &promoted {
        let side = strat.genome.side;
        let sstr = side_str(side);
        for &ticker in &cfg.universe {
            let window = build_window(&store, ticker, from_ts, to_ts, profile)
                .await
                .with_context(|| format!("window {ticker}"))?;
            let res = backtest(&strat.genome, &window, &profile);
            for entry in &res.entries {
                let ev = &entry.event;
                let win = ev.ret_r > 0.0;
                if win {
                    wins += 1;
                    gross_win += ev.ret_r;
                } else {
                    gross_loss += -ev.ret_r;
                }
                total += 1;
                sum_r += ev.ret_r;
                let exit_px = match ev.outcome.as_str() {
                    "target" => ev.target_px,
                    "stop" => ev.stop_px,
                    // time exit: reconstruct from realized R and the risk distance.
                    _ => ev.entry_px + side.sign() * ev.ret_r * (ev.entry_px - ev.stop_px).abs(),
                };
                sqlx::query(
                    "INSERT INTO trades_journal \
                     (trade_id, strategy_id, ticker, side, mode, entry_ts, fill_px, fill_ts, \
                      exit_px, exit_ts, pnl_r, cost_frac, attribution) \
                     VALUES ($1,$2,$3,$4,'backtest',$5,$6,$5,$7,$8,$9,$10,$11)",
                )
                .bind(Uuid::new_v4())
                .bind(strat.id.inner())
                .bind(ticker.as_str())
                .bind(sstr)
                .bind(ev.entry_ts)
                .bind(ev.entry_px)
                .bind(exit_px)
                .bind(ev.t1)
                .bind(ev.ret_r)
                .bind(cost)
                .bind(serde_json::json!({
                    "outcome": ev.outcome.as_str(),
                    "regime": entry.regime,
                }))
                .execute(store.pool())
                .await
                .context("insert backtest trade")?;
                if sample.len() < 16 {
                    sample.push((ticker.to_string(), sstr, ev.entry_px, ev.ret_r, ev.outcome.as_str()));
                }
            }
        }
    }

    if total == 0 {
        println!("Promoted strategies fired no labelable entries in this window.");
        return Ok(());
    }

    let losses = total - wins;
    let win_rate = wins as f64 / total as f64 * 100.0;
    let avg_r = sum_r / total as f64;
    let pf = if gross_loss > 0.0 {
        gross_win / gross_loss
    } else {
        f64::INFINITY
    };

    println!("\nSAMPLE TRADES (first {}):", sample.len());
    println!("{:<6} {:<6} {:>10} {:>9} {:>8}", "TICKER", "SIDE", "ENTRY", "PnL(R)", "OUTCOME");
    println!("{}", "-".repeat(46));
    for (t, s, entry, r, outcome) in &sample {
        let tag = if *r > 0.0 { "WIN " } else { "LOSS" };
        println!("{t:<6} {s:<6} {entry:>10.2} {r:>+9.3} {outcome:>8}  {tag}");
    }

    println!("\n══════════════════ WINNERS / LOSERS ══════════════════");
    println!("  trades:      {total}");
    println!("  winners:     {wins}  ({win_rate:.1}%)");
    println!("  losers:      {losses}  ({:.1}%)", 100.0 - win_rate);
    println!("  avg trade:   {avg_r:+.3} R");
    println!("  profit factor: {pf:.2}   (gross wins {gross_win:.1}R / gross losses {gross_loss:.1}R)");
    println!("  expectancy:  {avg_r:+.3} R/trade");
    println!("\n→ journaled as mode=backtest; view on the dashboard Journal page.");
    println!("(win rate is descriptive only — the engine ranks on expectancy/PF/CVaR, never win rate.)");

    Ok(())
}
