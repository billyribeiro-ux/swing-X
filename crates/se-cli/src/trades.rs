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

/// Reconstruct the realized exit price for a resolved backtest entry. Target/stop exits use the
/// stored barrier price; a time exit is reconstructed from realized R and the risk (stop) distance
/// — `|entry − stop|` is exactly one R by construction, so this inverts
/// `ret_r = side·(exit − entry) / R`.
fn reconstruct_exit_px(
    outcome: &str,
    entry_px: f64,
    stop_px: f64,
    target_px: f64,
    ret_r: f64,
    side: Side,
) -> f64 {
    match outcome {
        "target" => target_px,
        "stop" => stop_px,
        // time exit: reconstruct from realized R and the risk distance.
        _ => entry_px + side.sign() * ret_r * (entry_px - stop_px).abs(),
    }
}

/// Running winners/losers tally over per-trade R outcomes. A trade is a winner iff its realized R
/// is strictly positive; everything else (including a breakeven time exit) is a loser — matching
/// the journal's win/loss split. `win_rate` here is descriptive only; the engine never ranks on it.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct TradeStats {
    total: usize,
    wins: usize,
    gross_win: f64,
    gross_loss: f64,
    sum_r: f64,
}

impl TradeStats {
    fn push(&mut self, ret_r: f64) {
        self.total += 1;
        self.sum_r += ret_r;
        if ret_r > 0.0 {
            self.wins += 1;
            self.gross_win += ret_r;
        } else {
            self.gross_loss += -ret_r;
        }
    }
    fn losses(&self) -> usize {
        self.total - self.wins
    }
    fn avg_r(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.sum_r / self.total as f64
        }
    }
    fn win_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.wins as f64 / self.total as f64 * 100.0
        }
    }
    /// Gross wins ÷ gross losses; `+∞` when there are no losing R (an unbeaten record).
    fn profit_factor(&self) -> f64 {
        if self.gross_loss > 0.0 {
            self.gross_win / self.gross_loss
        } else {
            f64::INFINITY
        }
    }
}

pub async fn run(
    cfg: &AppConfig,
    horizon: Option<String>,
    from: Option<String>,
    to: Option<String>,
) -> Result<()> {
    let profile: HorizonProfile = resolve_profile(cfg, horizon.as_deref())?;
    let store = Store::connect(&cfg.database_url)
        .await
        .context("connect db")?;
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
    let mut stats = TradeStats::default();
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
                stats.push(ev.ret_r);
                let exit_px = reconstruct_exit_px(
                    ev.outcome.as_str(),
                    ev.entry_px,
                    ev.stop_px,
                    ev.target_px,
                    ev.ret_r,
                    side,
                );
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
                    sample.push((
                        ticker.to_string(),
                        sstr,
                        ev.entry_px,
                        ev.ret_r,
                        ev.outcome.as_str(),
                    ));
                }
            }
        }
    }

    if stats.total == 0 {
        println!("Promoted strategies fired no labelable entries in this window.");
        return Ok(());
    }

    let total = stats.total;
    let wins = stats.wins;
    let losses = stats.losses();
    let win_rate = stats.win_rate();
    let avg_r = stats.avg_r();
    let pf = stats.profit_factor();
    let gross_win = stats.gross_win;
    let gross_loss = stats.gross_loss;

    println!("\nSAMPLE TRADES (first {}):", sample.len());
    println!(
        "{:<6} {:<6} {:>10} {:>9} {:>8}",
        "TICKER", "SIDE", "ENTRY", "PnL(R)", "OUTCOME"
    );
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
    println!(
        "  profit factor: {pf:.2}   (gross wins {gross_win:.1}R / gross losses {gross_loss:.1}R)"
    );
    println!("  expectancy:  {avg_r:+.3} R/trade");

    // Honesty: the record above is IN-SAMPLE (the raw backtest, memorization-prone).
    // Contrast it with the purged OOS expectancy — the edge the engine actually trusts.
    // Use the LATEST score per strategy (as persist.rs / the API / the monitor all do), not a
    // flat avg over every historical row — re-evaluation appends rows, so a flat avg would mix in
    // superseded scores and weight strategies by their evaluation count.
    let oos_exp: Option<f64> = sqlx::query_scalar::<_, Option<f64>>(
        "SELECT avg(latest.oos_expectancy_cost_aware) \
         FROM strategies s \
         JOIN LATERAL ( \
             SELECT oos_expectancy_cost_aware FROM oos_scores \
             WHERE strategy_id = s.strategy_id \
             ORDER BY evaluated_at DESC LIMIT 1 \
         ) latest ON TRUE \
         WHERE s.status = 'promoted' AND s.horizon = $1",
    )
    .bind(profile.horizon.as_str())
    .fetch_one(store.pool())
    .await
    .ok()
    .flatten();

    println!("\n⚠ the above is the IN-SAMPLE backtest (memorization-prone).");
    if let Some(oos) = oos_exp {
        println!(
            "  OOS-VALIDATED expectancy (purged CPCV, latest per strategy — the edge the engine trusts): {oos:+.3} R",
        );
        // The in-sample figure is GROSS R; the OOS figure is cost-aware (net). So the gap is
        // overfit decay PLUS the round-trip cost wedge — both already discounted by the gate.
        println!(
            "  the gap (in-sample {avg_r:+.3}R gross → OOS {oos:+.3}R cost-aware) is overfit decay + the cost wedge the gate already discounts."
        );
    }
    println!("→ journaled as mode=backtest; view on the dashboard Journal page.");
    println!(
        "(win rate is descriptive only — the engine ranks on expectancy/PF/CVaR, never win rate.)"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trade_stats_tallies_winners_losers_and_expectancy() {
        let mut s = TradeStats::default();
        for r in [2.0, 2.0, -1.0, 2.0, -1.0] {
            s.push(r);
        }
        assert_eq!(s.total, 5);
        assert_eq!(s.wins, 3);
        assert_eq!(s.losses(), 2);
        assert!((s.sum_r - 4.0).abs() < 1e-12);
        assert!((s.avg_r() - 0.8).abs() < 1e-12);
        assert!((s.win_rate() - 60.0).abs() < 1e-12);
        // gross wins 6.0 / gross losses 2.0 = 3.0
        assert!((s.profit_factor() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn breakeven_time_exit_counts_as_loss_not_win() {
        // ret_r == 0 is NOT a win (strictly-positive rule) and adds 0 to gross_loss magnitude.
        let mut s = TradeStats::default();
        s.push(0.0);
        assert_eq!(s.wins, 0);
        assert_eq!(s.losses(), 1);
        assert_eq!(s.gross_loss, 0.0);
    }

    #[test]
    fn empty_stats_are_zero_not_nan() {
        let s = TradeStats::default();
        assert_eq!(s.avg_r(), 0.0);
        assert_eq!(s.win_rate(), 0.0);
        // no losses -> profit factor is +inf, never NaN.
        assert!(s.profit_factor().is_infinite());
    }

    #[test]
    fn unbeaten_record_has_infinite_profit_factor() {
        let mut s = TradeStats::default();
        s.push(2.0);
        s.push(1.5);
        assert!(s.profit_factor().is_infinite());
        assert_eq!(s.losses(), 0);
    }

    #[test]
    fn reconstruct_exit_uses_stored_barrier_for_target_and_stop() {
        // target/stop branches return the stored barrier price verbatim, regardless of ret_r.
        assert_eq!(
            reconstruct_exit_px("target", 100.0, 98.0, 104.0, 2.0, Side::Long),
            104.0
        );
        assert_eq!(
            reconstruct_exit_px("stop", 100.0, 98.0, 104.0, -1.0, Side::Long),
            98.0
        );
    }

    #[test]
    fn reconstruct_time_exit_inverts_ret_r_exactly_long() {
        // Long, entry 100, stop 98 => R = |100-98| = 2. A +0.5R time exit lands at 100 + 0.5*2 = 101.
        let px = reconstruct_exit_px("time", 100.0, 98.0, 104.0, 0.5, Side::Long);
        assert!((px - 101.0).abs() < 1e-9, "px={px}");
        // Round-trip: recompute ret_r the labeler way and confirm it matches.
        let r = (px - 100.0) / (100.0_f64 - 98.0).abs(); // long sign = +1
        assert!((r - 0.5).abs() < 1e-9);
    }

    #[test]
    fn reconstruct_time_exit_inverts_ret_r_exactly_short() {
        // Short, entry 100, stop 102 => R = 2. A +0.5R (favorable) exit: 100 + (-1)*0.5*2 = 99.
        let px = reconstruct_exit_px("time", 100.0, 102.0, 96.0, 0.5, Side::Short);
        assert!((px - 99.0).abs() < 1e-9, "px={px}");
        // Round-trip with short sign (-1): ret_r = -(px-entry)/R.
        let r = -(px - 100.0) / (100.0_f64 - 102.0).abs();
        assert!((r - 0.5).abs() < 1e-9);
    }
}
