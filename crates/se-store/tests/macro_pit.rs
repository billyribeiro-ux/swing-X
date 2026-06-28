//! Macro-store leakage proof: a macro observation whose `as_of` is AFTER the
//! decision bar must be invisible to `macro_value_as_of` / `macro_history` at
//! that bar, and become visible only once the decision instant passes its `as_of`.
//!
//! Requires a live database; set `DATABASE_URL`. Skips (passes) if unset.

use chrono::{Duration, TimeZone, Utc};
use se_core::{DecisionTs, LeadTimeTag, Ticker};
use se_store::{MacroWrite, Store};

async fn connect() -> Option<Store> {
    let url = std::env::var("DATABASE_URL").ok()?;
    let store = Store::connect(&url).await.expect("connect");
    store.migrate().await.expect("migrate");
    Some(store)
}

#[tokio::test]
async fn macro_pit_excludes_future_knowledge() {
    let Some(store) = connect().await else {
        eprintln!("SKIP macro_pit_excludes_future_knowledge: DATABASE_URL not set");
        return;
    };

    // Use a unique source tag so cleanup is isolated from other runs.
    let source = "macro_pit_test";
    store.delete_macro_by_source(source).await.unwrap();

    let series = "macro_pit_test_series";
    let decision = DecisionTs::new(Utc.with_ymd_and_hms(2025, 1, 10, 21, 0, 0).unwrap());

    // A point for an earlier date, knowable exactly at the decision bar.
    let clean = MacroWrite {
        series: series.into(),
        ts: Utc.with_ymd_and_hms(2025, 1, 9, 21, 0, 0).unwrap(),
        as_of: decision.inner(),
        value: 1.0,
        lead_time: LeadTimeTag::EndOfDay,
        source: source.into(),
    };
    // A point dated ON the decision bar, but only PUBLISHED a week later (FRED-like
    // lag). It must NOT be visible at the decision bar — that would be leakage.
    let lagged = MacroWrite {
        series: series.into(),
        ts: decision.inner(),
        as_of: decision.inner() + Duration::days(7),
        value: 9.0,
        lead_time: LeadTimeTag::LaggedDays(7),
        source: source.into(),
    };

    let n = store
        .upsert_macro(&[clean.clone(), lagged.clone()])
        .await
        .unwrap();
    assert_eq!(n, 2, "two distinct (series, ts, as_of) rows should insert");

    // The macro store ignores the ticker; any ticker yields the same cutoff read.
    let pit = store.pit(Ticker::SPY, decision);

    // At the decision bar: the clean value is the latest visible; the lagged one
    // (as_of in the future) is hidden -> value is the clean 1.0, not 9.0.
    let v = pit.macro_value_as_of(series).await.unwrap();
    assert_eq!(
        v,
        Some(1.0),
        "latest knowable value must be the clean point; lagged future release hidden"
    );

    // History likewise excludes the lagged future release.
    let hist = pit.macro_history(series, 50).await.unwrap();
    assert_eq!(
        hist.len(),
        1,
        "only the clean point is knowable at the decision bar"
    );
    assert_eq!(hist[0].1, 1.0);

    // Eight days later, the lagged release's as_of has passed -> it becomes the
    // latest visible value for its (later) reference date.
    let later = DecisionTs::new(decision.inner() + Duration::days(8));
    let pit2 = store.pit(Ticker::QQQ, later);
    let v2 = pit2.macro_value_as_of(series).await.unwrap();
    assert_eq!(
        v2,
        Some(9.0),
        "once as_of <= decision, the lagged release (newer ts) is observable"
    );
    let hist2 = pit2.macro_history(series, 50).await.unwrap();
    assert_eq!(hist2.len(), 2, "both points knowable after the lag passes");
    // Chronological order: clean (older ts) then lagged.
    assert_eq!(hist2[0].1, 1.0);
    assert_eq!(hist2[1].1, 9.0);

    store.delete_macro_by_source(source).await.unwrap();
}

#[tokio::test]
async fn macro_history_keeps_latest_vintage_per_ts() {
    let Some(store) = connect().await else {
        eprintln!("SKIP macro_history_keeps_latest_vintage_per_ts: DATABASE_URL not set");
        return;
    };
    let source = "macro_vintage_test";
    let series = "macro_vintage_series";
    store.delete_macro_by_source(source).await.unwrap();

    let decision = DecisionTs::new(Utc.with_ymd_and_hms(2025, 2, 1, 21, 0, 0).unwrap());
    let ts = Utc.with_ymd_and_hms(2025, 1, 15, 21, 0, 0).unwrap();

    // Two vintages of the SAME reference date, both knowable by the decision bar:
    // an initial print and a later revision. The read must pick the latest as_of.
    let initial = MacroWrite {
        series: series.into(),
        ts,
        as_of: ts,
        value: 5.0,
        lead_time: LeadTimeTag::EndOfDay,
        source: source.into(),
    };
    let revision = MacroWrite {
        series: series.into(),
        ts,
        as_of: ts + Duration::days(2),
        value: 5.5,
        lead_time: LeadTimeTag::EndOfDay,
        source: source.into(),
    };
    store.upsert_macro(&[initial, revision]).await.unwrap();

    let pit = store.pit(Ticker::SPY, decision);
    assert_eq!(
        pit.macro_value_as_of(series).await.unwrap(),
        Some(5.5),
        "must read the latest vintage (revision) for the reference date"
    );
    let hist = pit.macro_history(series, 10).await.unwrap();
    assert_eq!(
        hist.len(),
        1,
        "one reference date -> one history row (latest vintage)"
    );
    assert_eq!(hist[0].1, 5.5);

    store.delete_macro_by_source(source).await.unwrap();
}
