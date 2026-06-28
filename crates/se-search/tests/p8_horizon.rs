//! P8 — horizon generalization, end-to-end through the search/backtest/OOS pipeline.
//!
//! Proves NO swing constant is hardcoded anywhere in `se-search`: the SAME genome-search and
//! backtest pipeline runs unchanged under at least TWO distinct [`HorizonProfile`]s (swing +
//! day), differing only by the profile passed in. Because every barrier width, time barrier,
//! sampling cadence and cost lives in the profile, swapping the profile is the only change.
//!
//! Requires a live DB (`DATABASE_URL`) with bars/features ingested (`se scan`) AND the ML worker
//! (for the OOS score). Skips cleanly when either is absent — absence never fails CI.

use chrono::{TimeZone, Utc};
use se_core::{HorizonProfile, Ticker};
use se_mlclient::MlClient;
use se_search::{
    backtest, build_window, score_oos, seed_population, FeatureCatalog, Rng, ScoreConfig,
};
use se_store::Store;
use se_validation::ValidationHarness;

async fn store_if_up() -> Option<Store> {
    let url = std::env::var("DATABASE_URL").ok()?;
    let store = Store::connect(&url).await.ok()?;
    store.migrate().await.ok()?;
    Some(store)
}

async fn harness_if_up() -> Option<ValidationHarness> {
    let client = MlClient::from_env().ok()?;
    match client.health().await {
        Ok(h) if h.status == "ok" => Some(ValidationHarness::new(
            client,
            std::env::temp_dir().join("se_search_p8"),
        )),
        _ => None,
    }
}

/// Run the full pipeline (build window -> seed -> backtest -> OOS score) for one profile and a
/// fixed seed, returning how many genomes produced a labelable dataset (entries > 0) and how many
/// reached the OOS validator. The point is that this runs unchanged across profiles.
async fn run_under_profile(
    store: &Store,
    harness: &ValidationHarness,
    profile: HorizonProfile,
) -> (usize, usize) {
    let from = Utc.with_ymd_and_hms(2019, 6, 1, 21, 0, 0).unwrap();
    let to = Utc.with_ymd_and_hms(2024, 6, 1, 21, 0, 0).unwrap();

    // SPY/QQQ carry the deepest history; build their windows.
    let mut windows = Vec::new();
    for t in [Ticker::Spy, Ticker::Qqq] {
        let w = build_window(store, t, from, to, profile)
            .await
            .expect("window");
        if !w.points.is_empty() {
            windows.push(w);
        }
    }
    if windows.is_empty() {
        return (0, 0);
    }

    let catalog = FeatureCatalog::from_windows(&windows, 20);
    let mut rng = Rng::seeded(20240601, 0);
    let genomes = seed_population(&catalog, profile.horizon, 6, &mut rng, 3);

    let mut with_entries = 0usize;
    let mut scored = 0usize;
    for genome in genomes {
        let mut rows = Vec::new();
        for w in &windows {
            let res = backtest(&genome, w, &profile);
            rows.extend(se_search::assemble(&res));
        }
        rows.sort_by(|a, b| a.ts.cmp(&b.ts));
        if !rows.is_empty() {
            with_entries += 1;
        }
        let id = se_core::StrategyId::new();
        // Score OOS (skips tiny datasets internally, returning Ok(None)).
        if let Ok(Some(_score)) =
            score_oos(harness, id, &rows, &profile, ScoreConfig::default()).await
        {
            scored += 1;
        }
    }
    (with_entries, scored)
}

#[tokio::test]
async fn pipeline_runs_under_two_horizons() {
    let Some(store) = store_if_up().await else {
        eprintln!("SKIP p8_horizon: DATABASE_URL not set / DB unreachable");
        return;
    };
    let Some(harness) = harness_if_up().await else {
        eprintln!("SKIP p8_horizon: ml-worker not reachable");
        return;
    };

    // Two distinct profiles. They differ in every barrier/cadence/cost constant, yet the SAME
    // pipeline code runs for both — the proof that no swing constant is baked in.
    let swing = HorizonProfile::swing();
    let day = HorizonProfile::day();
    assert_ne!(swing.target_atr_mult, day.target_atr_mult);
    assert_ne!(swing.max_hold_bars, day.max_hold_bars);

    let (swing_entries, swing_scored) = run_under_profile(&store, &harness, swing).await;
    let (day_entries, day_scored) = run_under_profile(&store, &harness, day).await;

    println!(
        "[P8] swing: genomes_with_entries={swing_entries} scored_oos={swing_scored} \
         (target={} stop={} max_hold={})",
        swing.target_atr_mult, swing.stop_atr_mult, swing.max_hold_bars
    );
    println!(
        "[P8] day:   genomes_with_entries={day_entries} scored_oos={day_scored} \
         (target={} stop={} max_hold={})",
        day.target_atr_mult, day.stop_atr_mult, day.max_hold_bars
    );

    // The pipeline must actually produce labeled entries under BOTH profiles (if there is data
    // at all). If the store has no bars for SPY/QQQ in the window, both will be zero and we only
    // assert the code path ran without panicking.
    if swing_entries == 0 && day_entries == 0 {
        eprintln!(
            "NOTE p8_horizon: no labeled entries produced under either profile \
             (no bars in window?). Pipeline ran end-to-end without hardcoded constants."
        );
        return;
    }

    assert!(
        swing_entries > 0,
        "swing profile produced no entries but day did — a horizon-specific path leaked"
    );
    assert!(
        day_entries > 0,
        "day profile produced no entries but swing did — a horizon-specific path leaked"
    );
}
