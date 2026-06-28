//! `se signals` — generate + print current executable signals from PROMOTED strategies that
//! fire at each ticker's latest decision bar. Prints the human §6 format and a count. If nothing
//! promotes (or nothing fires), it says so plainly and does NOT fabricate a signal.

use anyhow::{Context, Result};

use se_config::AppConfig;
use se_core::HorizonProfile;
use se_search::persist::load_promoted;
use se_store::Store;

use crate::search_cmd::resolve_profile;

pub async fn run_signals(cfg: &AppConfig, horizon: Option<String>, journal: bool) -> Result<()> {
    let profile: HorizonProfile = resolve_profile(cfg, horizon.as_deref())?;

    let store = Store::connect(&cfg.database_url)
        .await
        .context("connect db")?;
    store.migrate().await.context("migrate")?;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(" se signals │ horizon={}", profile.horizon.as_str());
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let promoted = load_promoted(&store, profile.horizon.as_str()).await?;
    println!("promoted strategies for this horizon: {}", promoted.len());
    if promoted.is_empty() {
        println!(
            "\nNo promoted strategies for horizon `{}` — nothing to surface.\n\
             Run `se search --generations N --per-gen M` and, if a candidate passes the gate,\n\
             it is promoted automatically; signals will then appear here.",
            profile.horizon.as_str()
        );
        return Ok(());
    }

    let signals = se_signal::generate_signals(&store, &profile, &cfg.universe).await?;

    if signals.is_empty() {
        println!(
            "\nNo concrete signal at the latest bar: promoted strategies exist but none fired\n\
             on tradeable-regime data right now (geometry/regime/data gates not all satisfied)."
        );
        return Ok(());
    }

    println!(
        "\n{} signal(s) at the latest decision bar:\n",
        signals.len()
    );
    for (i, sig) in signals.iter().enumerate() {
        println!(
            "┌─ signal {} ─────────────────────────────────────────────",
            i + 1
        );
        for line in sig.to_human().lines() {
            println!("│ {line}");
        }
        println!("└────────────────────────────────────────────────────────");
        println!();

        if journal {
            match se_journal::journal_signal(&store, sig, &profile).await? {
                Some(pt) if pt.resolved => {
                    let t = &pt.trade;
                    println!(
                        "  ↳ paper trade: filled {:.2} @ next-bar-open(adverse), exit {:.2}, pnl {:+.2}R (cost {:.4})",
                        t.fill_px,
                        t.exit_px.unwrap_or(f64::NAN),
                        t.pnl_r.unwrap_or(f64::NAN),
                        t.cost_frac,
                    );
                }
                Some(pt) => {
                    println!(
                        "  ↳ paper trade opened (fill {:.2}); not yet resolved (awaiting forward bars)",
                        pt.trade.fill_px
                    );
                }
                None => println!(
                    "  ↳ no next bar to fill against yet (signal is at the very latest bar)"
                ),
            }
        }
    }

    Ok(())
}
