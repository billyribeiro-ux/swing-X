//! Smoke integration test for the axum router.
//!
//! Gated on `DATABASE_URL`: when unset the test is a no-op (so `cargo test` is green
//! without a database). When set, it boots the `Router` and drives it in-process via
//! `tower::ServiceExt::oneshot`, asserting `/api/health` and `/api/signals` return
//! 200 with a valid JSON shape (an empty `signals` array is acceptable).

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use se_api::{router, AppState};
use se_store::Store;
use tower::ServiceExt;

fn database_url() -> Option<String> {
    std::env::var("DATABASE_URL").ok().filter(|s| !s.is_empty())
}

#[tokio::test]
async fn health_and_signals_smoke() {
    let Some(url) = database_url() else {
        eprintln!("DATABASE_URL unset; skipping se-api smoke test");
        return;
    };
    let store = Store::connect(&url).await.expect("connect to DATABASE_URL");
    let app = router(AppState::new(store));

    // /api/health -> 200, { status: "ok" | "degraded" }
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&body).expect("health is JSON");
    assert!(v.get("status").and_then(|s| s.as_str()).is_some());

    // /api/signals -> 200, JSON array (possibly empty)
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/signals")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&body).expect("signals is JSON");
    assert!(v.is_array(), "GET /api/signals must return a JSON array");

    // /api/signals with a date window -> 200, JSON array. The window filters
    // `decision_ts`; the result is a (possibly empty) array, never an error.
    let res = app
        .oneshot(
            Request::builder()
                .uri("/api/signals?from=2026-01-01&to=2026-12-31")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&body).expect("ranged signals is JSON");
    assert!(
        v.is_array(),
        "GET /api/signals?from&to must return a JSON array"
    );
}
