//! Data-layer leakage proof: a feature whose `as_of` is AFTER the decision bar
//! must be invisible to a `PitContext` at that bar, and only become visible once
//! the decision instant has advanced past its `as_of`.
//!
//! Requires a live database; set `DATABASE_URL`. Skips (passes) if unset so the
//! suite still runs in environments without a DB.

use chrono::{Duration, TimeZone, Utc};
use se_core::{AsOf, DecisionTs, Layer, LeadTimeTag, Ticker};
use se_store::{FeatureWrite, Store};

async fn connect() -> Option<Store> {
    let url = std::env::var("DATABASE_URL").ok()?;
    let store = Store::connect(&url).await.expect("connect");
    store.migrate().await.expect("migrate");
    Some(store)
}

#[tokio::test]
async fn pit_excludes_future_knowledge() {
    let Some(store) = connect().await else {
        eprintln!("SKIP pit_excludes_future_knowledge: DATABASE_URL not set");
        return;
    };

    // clean up any leftovers from a prior run
    store.delete_features_by_source("pit_test").await.unwrap();

    let ticker = Ticker::Spy;
    let decision = DecisionTs::new(Utc.with_ymd_and_hms(2025, 1, 10, 21, 0, 0).unwrap());

    // Clean: knowable exactly at the decision bar.
    let clean = FeatureWrite {
        ticker,
        feature_key: "pit_test.clean".into(),
        layer: Layer::Regime,
        decision_ts: decision,
        as_of: AsOf::new(decision.inner()),
        value: 1.0,
        lead_time: LeadTimeTag::EndOfDay,
        source: "pit_test".into(),
    };
    // Leaky: same decision bar, but only knowable a day LATER (a future peek).
    let leaky = FeatureWrite {
        ticker,
        feature_key: "pit_test.leaky".into(),
        layer: Layer::Regime,
        decision_ts: decision,
        as_of: AsOf::new(decision.inner() + Duration::days(1)),
        value: 9.0,
        lead_time: LeadTimeTag::EndOfDay,
        source: "pit_test".into(),
    };

    store
        .insert_features(&[clean, leaky])
        .await
        .expect("insert");

    // At the decision bar: clean visible, leaky invisible.
    let pit = store.pit(ticker, decision);
    let feats = pit.features(None).await.unwrap();
    let keys: Vec<&str> = feats.iter().map(|f| f.key.as_str()).collect();
    assert!(
        keys.contains(&"pit_test.clean"),
        "clean feature must be visible"
    );
    assert!(
        !keys.contains(&"pit_test.leaky"),
        "LEAKY feature (as_of after decision) must NOT be visible"
    );
    assert!(pit.feature("pit_test.clean").await.unwrap().is_some());
    assert!(
        pit.feature("pit_test.leaky").await.unwrap().is_none(),
        "direct lookup of leaky feature must also be hidden"
    );

    // Two days later, the value's as_of has passed -> it becomes visible.
    let later = DecisionTs::new(decision.inner() + Duration::days(2));
    let pit2 = store.pit(ticker, later);
    assert!(
        pit2.feature("pit_test.leaky").await.unwrap().is_some(),
        "once as_of <= decision, the value is observable"
    );

    store.delete_features_by_source("pit_test").await.unwrap();
}
