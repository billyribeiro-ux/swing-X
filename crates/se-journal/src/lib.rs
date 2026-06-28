//! `se-journal` (P7) — the paper-trade journal with realistic fills + attribution.
//!
//! From a [`se_core::Signal`] decided at bar `T`, open a paper trade filled at the NEXT bar's
//! open made adverse by the horizon's round-trip cost (see [`fills`]). Walk bars forward and
//! close the trade when the triple-barrier outcome resolves (target / stop / time), then record
//! the realized return in R units (`pnl_r`), net of cost. Trades persist to `trades_journal`
//! with their signal/strategy links. [`realized_stats`] reads the journal back to give the
//! monitor per-strategy realized expectancy / profit-factor / CVaR.

pub mod fills;

use chrono::{DateTime, Utc};
use se_core::{
    DecisionTs, HorizonProfile, Result, Side, Signal, Ticker, Trade, TradeId, TradeMode,
};
use se_labeler::TripleBarrier;
use se_store::Store;

use crate::fills::{adverse_price, Leg};

/// The result of opening + resolving a paper trade from a signal.
#[derive(Debug, Clone)]
pub struct PaperTrade {
    pub trade: Trade,
    /// The bar at which the trade was filled (next bar open after the signal).
    pub fill_ts: DateTime<Utc>,
    /// The resolved exit, if the barrier was hit within the loaded forward window.
    pub resolved: bool,
}

/// Open a paper trade from `signal` and resolve it against forward bars, using `profile` for the
/// barrier geometry and cost. Returns `None` if there is no next bar to fill against.
///
/// Fill convention: entry at the NEXT bar's OPEN after the signal's decision bar, worsened by
/// half the round-trip cost. The triple-barrier outcome (computed by the labeler from that next
/// bar onward) gives the R return; we then subtract the full round-trip cost in R units so the
/// journaled `pnl_r` is net of costs.
pub async fn open_and_resolve(
    store: &Store,
    signal: &Signal,
    profile: &HorizonProfile,
) -> Result<Option<PaperTrade>> {
    // Load forward bars from the signal's decision bar (PIT-safe at the *latest* available
    // cutoff — the journal resolves trades using data that has since become known).
    let bars = load_forward_bars(store, signal.ticker, signal.decision_ts, profile).await?;
    // We need the decision bar plus at least one forward bar to fill.
    let entry_idx = bars.iter().position(|b| b.ts >= signal.decision_ts.inner());
    let Some(entry_idx) = entry_idx else {
        return Ok(None);
    };
    if entry_idx + 1 >= bars.len() {
        // No next bar to fill against yet.
        return Ok(None);
    }

    let side = signal.side;
    let cost = &profile.cost;

    // Fill at next-bar open, made adverse.
    let raw_fill = bars[entry_idx + 1].open;
    let fill_px = adverse_price(raw_fill, side, Leg::Entry, cost);
    let fill_ts = bars[entry_idx + 1].ts;

    // Resolve via triple-barrier from the FILL bar (entry executes at the fill bar's price; the
    // first touchable bar is the one after, matching the labeler's no-look-ahead convention).
    // The labeler needs a positive ATR; recompute it at the fill bar.
    let atr = se_features_atr(&bars[..=(entry_idx + 1)], profile.atr_lookback as usize);
    let labeler = TripleBarrier::new(*profile);

    let (exit_ts, exit_px, pnl_r_gross, resolved) = match atr {
        Some(atr) if atr > 0.0 => {
            match labeler.label_one(&bars, entry_idx + 1, side, atr) {
                Ok(ev) => {
                    // Exit at the barrier bar's relevant price (target/stop at the barrier; time
                    // at the barrier bar's close), made adverse by exit slippage.
                    let raw_exit = match ev.outcome {
                        se_labeler::Outcome::Target => ev.target_px,
                        se_labeler::Outcome::Stop => ev.stop_px,
                        se_labeler::Outcome::Time => bars
                            .iter()
                            .find(|b| b.ts == ev.t1)
                            .map(|b| b.close)
                            .unwrap_or(ev.entry_px),
                    };
                    let exit_px = adverse_price(raw_exit, side, Leg::Exit, cost);
                    (Some(ev.t1), Some(exit_px), ev.ret_r, true)
                }
                Err(_) => (None, None, 0.0, false),
            }
        }
        _ => (None, None, 0.0, false),
    };

    // Net the full round-trip cost out of the R return. One R in price = stop distance =
    // stop_atr_mult * ATR; express the cost (a fraction of notional) in R units.
    let pnl_r = if resolved {
        Some(pnl_r_gross - cost_in_r(fill_px, profile, atr))
    } else {
        None
    };

    let trade = Trade {
        id: TradeId::new(),
        signal_id: Some(signal.id),
        strategy_id: Some(signal.strategy_id),
        ticker: signal.ticker,
        side,
        mode: TradeMode::Paper,
        entry_ts: signal.decision_ts,
        fill_px,
        fill_ts: DecisionTs::new(fill_ts),
        exit_ts: exit_ts.map(DecisionTs::new),
        exit_px,
        pnl_r,
        cost_frac: cost.round_trip_frac(),
    };

    Ok(Some(PaperTrade {
        trade,
        fill_ts,
        resolved,
    }))
}

/// Express the round-trip cost fraction as a loss in R units. One R (in price) is
/// `stop_atr_mult * ATR`; the cost is `round_trip_frac * fill_px` (a price), so
/// `cost_R = cost_price / R_price`.
fn cost_in_r(fill_px: f64, profile: &HorizonProfile, atr: Option<f64>) -> f64 {
    let r_price = profile.stop_atr_mult * atr.unwrap_or(0.0);
    if r_price <= 0.0 {
        return 0.0;
    }
    profile.cost.round_trip_frac() * fill_px / r_price
}

/// Wilder ATR over the bars (thin wrapper so we don't pull in the whole features crate here).
fn se_features_atr(bars: &[se_core::Bar], lookback: usize) -> Option<f64> {
    if lookback == 0 || bars.len() <= lookback {
        return None;
    }
    let trs: Vec<f64> = bars
        .windows(2)
        .map(|w| w[1].true_range(w[0].close))
        .collect();
    let slice = &trs[trs.len() - lookback..];
    Some(slice.iter().sum::<f64>() / lookback as f64)
}

/// Load bars for `ticker` spanning the signal's decision bar forward, with enough warmup for the
/// ATR. PIT cutoff is the latest stored bar (resolution uses data that has since become known).
async fn load_forward_bars(
    store: &Store,
    ticker: Ticker,
    decision_ts: DecisionTs,
    profile: &HorizonProfile,
) -> Result<Vec<se_core::Bar>> {
    // Warmup before the decision bar (for ATR) + forward bars (>= max_hold) to resolve.
    let warmup = (profile.atr_lookback as i64) + 5;
    let forward = (profile.max_hold_bars as i64) + 5;
    type Row = (String, DateTime<Utc>, f64, f64, f64, f64, f64);
    let rows: Vec<Row> = se_store::sqlx::query_as(
        "SELECT ticker, ts, open, high, low, close, volume FROM ( \
             ( SELECT ticker, ts, open, high, low, close, volume FROM bars \
               WHERE ticker = $1 AND cadence = 'daily' AND ts <= $2 \
               ORDER BY ts DESC LIMIT $3 ) \
             UNION ALL \
             ( SELECT ticker, ts, open, high, low, close, volume FROM bars \
               WHERE ticker = $1 AND cadence = 'daily' AND ts > $2 \
               ORDER BY ts ASC LIMIT $4 ) \
         ) u ORDER BY ts ASC",
    )
    .bind(ticker.as_str())
    .bind(decision_ts.inner())
    .bind(warmup)
    .bind(forward)
    .fetch_all(store.pool())
    .await
    .map_err(|e| se_core::Error::Store(e.to_string()))?;

    Ok(rows
        .into_iter()
        .filter_map(|(tk, ts, open, high, low, close, volume)| {
            let ticker: Ticker = tk.parse().ok()?;
            Some(se_core::Bar {
                ticker,
                ts,
                open,
                high,
                low,
                close,
                volume,
            })
        })
        .collect())
}

/// Persist a paper trade to `trades_journal` (idempotent on `(trade_id, entry_ts)`).
pub async fn persist_trade(store: &Store, pt: &PaperTrade) -> Result<()> {
    let t = &pt.trade;
    let side = match t.side {
        Side::Long => "long",
        Side::Short => "short",
    };
    let attribution = serde_json::json!({
        "signal_id": t.signal_id.map(|s| s.to_string()),
        "strategy_id": t.strategy_id.map(|s| s.to_string()),
        "resolved": pt.resolved,
    });
    se_store::sqlx::query(
        "INSERT INTO trades_journal \
            (trade_id, signal_id, strategy_id, ticker, side, mode, entry_ts, fill_px, fill_ts, \
             exit_ts, exit_px, pnl_r, cost_frac, attribution) \
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14) \
         ON CONFLICT (trade_id, entry_ts) DO UPDATE SET \
             exit_ts = EXCLUDED.exit_ts, exit_px = EXCLUDED.exit_px, pnl_r = EXCLUDED.pnl_r",
    )
    .bind(t.id.inner())
    .bind(t.signal_id.map(|s| s.inner()))
    .bind(t.strategy_id.map(|s| s.inner()))
    .bind(t.ticker.as_str())
    .bind(side)
    .bind(t.mode.as_str())
    .bind(t.entry_ts.inner())
    .bind(t.fill_px)
    .bind(t.fill_ts.inner())
    .bind(t.exit_ts.map(|x| x.inner()))
    .bind(t.exit_px)
    .bind(t.pnl_r)
    .bind(t.cost_frac)
    .bind(attribution)
    .execute(store.pool())
    .await
    .map_err(|e| se_core::Error::Store(e.to_string()))?;
    Ok(())
}

/// Open + resolve + persist a paper trade from a signal in one call.
pub async fn journal_signal(
    store: &Store,
    signal: &Signal,
    profile: &HorizonProfile,
) -> Result<Option<PaperTrade>> {
    match open_and_resolve(store, signal, profile).await? {
        Some(pt) => {
            persist_trade(store, &pt).await?;
            Ok(Some(pt))
        }
        None => Ok(None),
    }
}

/// Per-strategy realized performance read back from the journal — what the monitor consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct RealizedStats {
    pub strategy_id: se_core::StrategyId,
    /// Number of resolved (closed) trades.
    pub n: usize,
    /// Realized expectancy in R (mean `pnl_r`).
    pub expectancy_r: f64,
    /// Profit factor = sum(wins) / |sum(losses)| (`inf` if no losses, `0` if no wins).
    pub profit_factor: f64,
    /// CVaR at 5% in R (mean of the worst 5% of `pnl_r`; `0` if too few trades).
    pub cvar5: f64,
}

/// Compute realized stats for a strategy from its closed trades in `trades_journal`.
pub async fn realized_stats(
    store: &Store,
    strategy_id: se_core::StrategyId,
) -> Result<RealizedStats> {
    let rows: Vec<(Option<f64>,)> = se_store::sqlx::query_as(
        "SELECT pnl_r FROM trades_journal \
         WHERE strategy_id = $1 AND pnl_r IS NOT NULL ORDER BY entry_ts ASC",
    )
    .bind(strategy_id.inner())
    .fetch_all(store.pool())
    .await
    .map_err(|e| se_core::Error::Store(e.to_string()))?;

    let pnls: Vec<f64> = rows.into_iter().filter_map(|r| r.0).collect();
    Ok(compute_stats(strategy_id, &pnls))
}

/// Pure stats computation over a slice of realized R returns (testable without a DB).
pub fn compute_stats(strategy_id: se_core::StrategyId, pnls: &[f64]) -> RealizedStats {
    let n = pnls.len();
    if n == 0 {
        return RealizedStats {
            strategy_id,
            n: 0,
            expectancy_r: 0.0,
            profit_factor: 0.0,
            cvar5: 0.0,
        };
    }
    let expectancy_r = pnls.iter().sum::<f64>() / n as f64;
    let wins: f64 = pnls.iter().filter(|x| **x > 0.0).sum();
    let losses: f64 = pnls.iter().filter(|x| **x < 0.0).map(|x| x.abs()).sum();
    let profit_factor = if losses > 0.0 {
        wins / losses
    } else if wins > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    // CVaR(5%): mean of the worst 5% of returns (at least one observation).
    let mut sorted = pnls.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let tail = ((n as f64 * 0.05).ceil() as usize).max(1).min(n);
    let cvar5 = sorted[..tail].iter().sum::<f64>() / tail as f64;

    RealizedStats {
        strategy_id,
        n,
        expectancy_r,
        profit_factor,
        cvar5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use se_core::StrategyId;

    #[test]
    fn stats_basic_expectancy_and_pf() {
        let id = StrategyId::new();
        let pnls = [2.0, -1.0, 2.0, -1.0, -1.0]; // 3 wins-ish? sum = 1.0, mean 0.2
        let s = compute_stats(id, &pnls);
        assert_eq!(s.n, 5);
        assert!((s.expectancy_r - 0.2).abs() < 1e-9);
        // wins = 4.0, losses = 3.0 -> PF = 1.333...
        assert!((s.profit_factor - 4.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn stats_all_wins_is_inf_pf() {
        let s = compute_stats(StrategyId::new(), &[1.0, 2.0, 3.0]);
        assert!(s.profit_factor.is_infinite());
    }

    #[test]
    fn stats_empty_is_zeroed() {
        let s = compute_stats(StrategyId::new(), &[]);
        assert_eq!(s.n, 0);
        assert_eq!(s.expectancy_r, 0.0);
        assert_eq!(s.profit_factor, 0.0);
    }

    #[test]
    fn cvar_picks_worst_tail() {
        // 20 trades: worst 5% = 1 trade = the minimum.
        let mut pnls: Vec<f64> = (0..20).map(|i| i as f64).collect();
        pnls[0] = -10.0;
        let s = compute_stats(StrategyId::new(), &pnls);
        assert_eq!(s.cvar5, -10.0);
    }
}
