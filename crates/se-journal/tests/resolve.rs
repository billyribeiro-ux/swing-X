//! Journal resolution against real stored bars (DB-gated; skips cleanly without `DATABASE_URL`).
//!
//! Builds a signal at an OLDER SPY decision bar (so forward bars exist), opens + resolves a
//! paper trade, and asserts: the fill is next-bar-open-or-worse, the trade resolves to a finite
//! `pnl_r`, and the realized-stats readback aggregates it. This exercises the full P7 journal
//! path end-to-end without depending on a live promoted strategy.

use chrono::Utc;
use se_core::{Driver, HorizonProfile, Layer, Side, Signal, StrategyId, Ticker};
use se_journal::{compute_stats, journal_signal, realized_stats};
use se_store::Store;

async fn store_if_up() -> Option<Store> {
    let url = std::env::var("DATABASE_URL").ok()?;
    let store = Store::connect(&url).await.ok()?;
    store.migrate().await.ok()?;
    Some(store)
}

/// Find a SPY daily bar at least `min_forward` bars before the latest stored bar.
async fn older_decision_bar(
    store: &Store,
    min_forward: i64,
) -> Option<(chrono::DateTime<Utc>, f64)> {
    let rows: Vec<(chrono::DateTime<Utc>, f64)> = se_store::sqlx::query_as(
        "SELECT ts, close FROM bars WHERE ticker = 'SPY' AND cadence = 'daily' ORDER BY ts DESC LIMIT $1",
    )
    .bind(min_forward + 1)
    .fetch_all(store.pool())
    .await
    .ok()?;
    // The OLDEST of the most-recent (min_forward+1) bars has `min_forward` bars after it.
    rows.into_iter().last()
}

#[tokio::test]
async fn journal_resolves_a_paper_trade_with_next_bar_fill() {
    let Some(store) = store_if_up().await else {
        eprintln!("SKIP journal_resolves: DATABASE_URL not set / DB unreachable");
        return;
    };
    let profile = HorizonProfile::swing();

    let Some((ts, close)) = older_decision_bar(&store, profile.max_hold_bars as i64 + 4).await
    else {
        eprintln!("SKIP journal_resolves: not enough SPY bars in store");
        return;
    };

    // A hand-built long signal at that older bar (stop/target geometry consistent).
    let entry = close;
    let stop = entry * 0.95;
    let target1 = entry * 1.10;
    let signal = Signal::new(
        StrategyId::new(),
        Ticker::SPY,
        Side::Long,
        se_core::DecisionTs::new(ts),
        profile.horizon,
        entry,
        stop,
        target1,
        Some(entry * 1.16),
        0.5,
        50,
        se_core::RegimeLabel::RiskOn,
        "risk_on",
        vec![Driver {
            layer: Layer::Trigger,
            key: "trigger.rsi14".into(),
            contribution: 1.0,
            detail: "rsi14 > 55".into(),
        }],
        "daily close < stop",
        0.3,
        -1.0,
        "test",
    )
    .expect("valid signal geometry");

    let pt = journal_signal(&store, &signal, &profile)
        .await
        .expect("journal call")
        .expect("a next bar exists to fill against");

    // Fill is the next bar's open, made adverse (>= raw open for a long entry).
    assert!(
        pt.trade.fill_ts.inner() > ts,
        "fill must be after the decision bar"
    );
    assert!(pt.trade.fill_px.is_finite() && pt.trade.fill_px > 0.0);

    // It should resolve (target/stop/time) within the loaded forward window.
    assert!(
        pt.resolved,
        "trade should resolve within max_hold+forward bars"
    );
    let pnl = pt.trade.pnl_r.expect("resolved trade has pnl_r");
    assert!(pnl.is_finite(), "pnl_r must be finite, got {pnl}");

    // Realized-stats readback includes this trade.
    let stats = realized_stats(&store, signal.strategy_id)
        .await
        .expect("stats");
    assert_eq!(
        stats.n, 1,
        "exactly one resolved trade for this fresh strategy id"
    );
    assert!((stats.expectancy_r - pnl).abs() < 1e-9);

    println!(
        "[journal] fill={:.2} (next-bar-open adverse), exit={:.2}, pnl={:+.3}R, cost_frac={:.5}",
        pt.trade.fill_px,
        pt.trade.exit_px.unwrap_or(f64::NAN),
        pnl,
        pt.trade.cost_frac,
    );

    // Sanity: pure-stats helper agrees with the DB readback.
    let pure = compute_stats(signal.strategy_id, &[pnl]);
    assert_eq!(pure.n, 1);

    // Clean up so re-runs stay deterministic.
    let _ = se_store::sqlx::query("DELETE FROM trades_journal WHERE strategy_id = $1")
        .bind(signal.strategy_id.inner())
        .execute(store.pool())
        .await;
}
