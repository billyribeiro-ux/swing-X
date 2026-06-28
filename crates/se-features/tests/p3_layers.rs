//! P3 feature-layer integration proof: ingest a short MOCK daily-bar window for
//! the whole universe, then run LocationModule / TriggerModule / EventOverlay at
//! the latest decision bar and assert each emits at least one feature and every
//! emitted value is finite. Trigger exercises the cross-ticker `bars_for` reads
//! (relative strength vs SPY + universe breadth).
//!
//! Requires a live database; set `DATABASE_URL`. Skips (passes) if unset.

use chrono::NaiveDate;
use se_core::{DecisionTs, HorizonProfile, Ticker};
use se_features::{EventOverlay, FeatureContext, FeatureModule, LocationModule, TriggerModule};
use se_provider::{DataProvider, MockProvider, NullProprietary};
use se_store::Store;

async fn connect() -> Option<Store> {
    let url = std::env::var("DATABASE_URL").ok()?;
    let store = Store::connect(&url).await.expect("connect");
    store.migrate().await.expect("migrate");
    Some(store)
}

#[tokio::test]
async fn p3_layers_produce_finite_features() {
    let Some(store) = connect().await else {
        eprintln!("SKIP p3_layers_produce_finite_features: DATABASE_URL not set");
        return;
    };

    // Use a dedicated provenance tag so the test data is isolated + cleanable.
    let source = "p3_layers_test";
    store.delete_features_by_source(source).await.unwrap();

    // ~1 year of synthetic daily bars so SMA200 / ATR / breadth windows fill.
    let provider = MockProvider;
    let from = NaiveDate::from_ymd_opt(2024, 6, 3).unwrap();
    let to = NaiveDate::from_ymd_opt(2025, 6, 20).unwrap();

    let mut latest: Option<chrono::DateTime<chrono::Utc>> = None;
    for &t in &Ticker::ALL {
        let bars = provider.daily_bars(t, from, to).await.unwrap();
        assert!(!bars.is_empty(), "mock must produce bars for {t}");
        let last_ts = bars.last().unwrap().ts;
        latest = Some(last_ts);
        store.upsert_bars(&bars, "daily", source).await.unwrap();
    }
    let last_ts = latest.expect("at least one bar");
    let decision = DecisionTs::new(last_ts);

    let prop = NullProprietary;
    let profile = HorizonProfile::swing();

    // Evaluate against a non-benchmark sector ETF so rs_vs_spy goes through the
    // real cross-ticker path (SPY would short-circuit to 0).
    let pit = store.pit(Ticker::XLK, decision);
    let ctx = FeatureContext::new(&pit, &prop, profile);

    let location = LocationModule::new();
    let trigger = TriggerModule::new();
    let events = EventOverlay::new();

    let loc = location.compute(&ctx).await.unwrap();
    let trig = trigger.compute(&ctx).await.unwrap();
    let evt = events.compute(&ctx).await.unwrap();

    assert!(!loc.is_empty(), "location layer must emit features");
    assert!(!trig.is_empty(), "trigger layer must emit features");
    assert!(!evt.is_empty(), "event overlay must emit features");

    for f in loc.iter().chain(trig.iter()).chain(evt.iter()) {
        assert!(
            f.value.is_finite(),
            "feature {} must be finite, got {}",
            f.key,
            f.value
        );
    }

    // Spot-check the layer tags + that the headline keys are present.
    let loc_keys: Vec<&str> = loc.iter().map(|f| f.key.as_str()).collect();
    assert!(loc_keys.contains(&"location.dist_50dma"));
    assert!(loc_keys.contains(&"location.pct_range_position"));

    let trig_keys: Vec<&str> = trig.iter().map(|f| f.key.as_str()).collect();
    assert!(trig_keys.contains(&"trigger.rs_vs_spy"));
    assert!(trig_keys.contains(&"trigger.breadth_thrust"));
    assert!(trig_keys.contains(&"trigger.rsi14"));

    let evt_keys: Vec<&str> = evt.iter().map(|f| f.key.as_str()).collect();
    assert!(evt_keys.contains(&"event.is_opex"));
    assert!(evt_keys.contains(&"event.pre_fomc"));

    // breadth_thrust is a fraction in [0,1].
    let breadth = trig
        .iter()
        .find(|f| f.key == "trigger.breadth_thrust")
        .unwrap()
        .value;
    assert!((0.0..=1.0).contains(&breadth), "breadth in [0,1]");

    // Persist through the real write path (mirrors the CLI) and confirm the round
    // trip works. These features carry their module's own source ("derived" /
    // "calendar"), so we don't blanket-delete by source (that would touch real
    // scan data); the idempotent upsert simply overwrites this one bar's keys.
    let writes: Vec<se_store::FeatureWrite> = loc
        .iter()
        .chain(trig.iter())
        .chain(evt.iter())
        .map(|f| se_store::FeatureWrite::from_feature(Ticker::XLK, decision, f))
        .collect();
    store.insert_features(&writes).await.unwrap();

    // Clean up the test bars (tagged with our isolated source) and any features
    // we inserted under that same tag.
    store.delete_features_by_source(source).await.unwrap();
    se_store::sqlx::query("DELETE FROM bars WHERE source = $1")
        .bind(source)
        .execute(store.pool())
        .await
        .unwrap();
}
