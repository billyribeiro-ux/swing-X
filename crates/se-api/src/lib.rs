//! `se-api` — axum HTTP + WebSocket backend for the operator dashboard.
//!
//! [`router`] builds the full `axum::Router` over an [`AppState`] holding a shared
//! [`se_store::Store`]. REST handlers map DB rows to the camelCase DTOs the
//! SvelteKit app validates with its zod schemas (see [`dto`]).
//!
//! ## WebSocket push (`/api/stream`)
//! v1 uses a simple server-side **DB poll + broadcast**: a background task polls the
//! newest `signals.created_at` and `monitor_events.ts` every [`STREAM_POLL`]ms; when
//! a newer row appears it serializes the new [`dto::SignalDto`] / [`dto::MonitorEventDto`]
//! and publishes it on a `tokio::sync::broadcast` channel. Each `/api/stream` socket
//! subscribes to that channel and forwards frames as text JSON envelopes
//! `{ "kind": "signal" | "monitor", "data": <dto> }`. This is intentionally simple
//! (no LISTEN/NOTIFY, no per-client cursors) and adequate for a single-operator
//! dashboard; a future version can swap the poll for Postgres `LISTEN/NOTIFY`.

pub mod dto;
pub mod queries;
pub mod stream;

use std::time::Duration;

use axum::extract::ws::WebSocketUpgrade;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::get;
use axum::Router;
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use serde::Deserialize;
use se_store::Store;
use tower_http::cors::CorsLayer;

use crate::queries::DateBounds;
pub use stream::{StreamHub, STREAM_POLL};

/// Optional `?from=&to=` date-window query, shared by the list endpoints
/// (`/api/signals`, `/api/population`, `/api/journal`, `/api/monitor`).
///
/// Each bound accepts either an ISO calendar date (`YYYY-MM-DD`) or a full
/// RFC-3339 timestamp. Bounds are **inclusive**: a date-only `from` expands to
/// `00:00:00Z` of that day and a date-only `to` expands to `23:59:59.999999999Z`
/// of that day so the whole day is covered. When a param is absent or empty that
/// side is unconstrained; when both are absent the endpoint is unfiltered, so
/// existing callers see no behavior change. An unparseable value is ignored
/// (treated as absent) rather than erroring, keeping the dashboard resilient.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct DateRange {
    #[serde(default)]
    pub from: Option<String>,
    #[serde(default)]
    pub to: Option<String>,
}

impl DateRange {
    /// Resolve the string params into concrete UTC bounds for the query layer.
    pub fn bounds(&self) -> DateBounds {
        DateBounds {
            from: self.from.as_deref().and_then(|s| parse_bound(s, false)),
            to: self.to.as_deref().and_then(|s| parse_bound(s, true)),
        }
    }
}

/// Parse one bound. Tries RFC-3339 first, then a bare `YYYY-MM-DD` date which is
/// expanded to the start of the day (`from`) or the inclusive end of the day
/// (`to`). Blank or malformed input yields `None` (no constraint on that side).
fn parse_bound(raw: &str, is_upper: bool) -> Option<DateTime<Utc>> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&Utc));
    }
    let date = NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()?;
    let naive = if is_upper {
        // Inclusive end of day, nanosecond precision (timestamps compare <=).
        date.and_hms_nano_opt(23, 59, 59, 999_999_999)?
    } else {
        date.and_hms_opt(0, 0, 0)?
    };
    Some(Utc.from_utc_datetime(&naive))
}

/// Shared application state. Cheap to clone (the store wraps an `Arc` pool, the hub
/// holds a `broadcast::Sender`).
#[derive(Clone)]
pub struct AppState {
    pub store: Store,
    pub hub: StreamHub,
}

impl AppState {
    pub fn new(store: Store) -> Self {
        AppState {
            store,
            hub: StreamHub::new(),
        }
    }
}

/// Build the API router. CORS is permissive for local dev.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/signals", get(get_signals))
        .route("/api/signals/{id}", get(get_signal))
        .route("/api/population", get(get_population))
        .route("/api/monitor", get(get_monitor))
        .route("/api/journal", get(get_journal))
        .route("/api/changelog", get(get_changelog))
        .route("/api/stream", get(get_stream))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Map a `se_core::Error` to a 500 with a JSON body. NotFound is handled inline by
/// the relevant handler (returns 404).
fn err_response(e: se_core::Error) -> Response {
    tracing::error!(error = %e, "api handler error");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({ "error": e.to_string() })),
    )
        .into_response()
}

async fn health(State(state): State<AppState>) -> Response {
    // Report degraded (but still 200) if the DB ping fails, so the dashboard's
    // health badge reflects reality without the endpoint itself hard-failing.
    let status = match state.store.ping().await {
        Ok(()) => "ok",
        Err(_) => "degraded",
    };
    Json(dto::HealthDto {
        status: status.to_string(),
    })
    .into_response()
}

async fn get_signals(State(state): State<AppState>, Query(range): Query<DateRange>) -> Response {
    match queries::signals(&state.store, range.bounds()).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => err_response(e),
    }
}

async fn get_signal(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    match queries::signal_by_id(&state.store, &id).await {
        Ok(Some(s)) => Json(s).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("signal {id} not found") })),
        )
            .into_response(),
        Err(e) => err_response(e),
    }
}

async fn get_population(State(state): State<AppState>, Query(range): Query<DateRange>) -> Response {
    match queries::population(&state.store, range.bounds()).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => err_response(e),
    }
}

async fn get_monitor(State(state): State<AppState>, Query(range): Query<DateRange>) -> Response {
    match queries::monitor_events(&state.store, range.bounds()).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => err_response(e),
    }
}

async fn get_journal(State(state): State<AppState>, Query(range): Query<DateRange>) -> Response {
    match queries::journal(&state.store, range.bounds()).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => err_response(e),
    }
}

async fn get_changelog(State(state): State<AppState>) -> Response {
    match queries::changelog(&state.store).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => err_response(e),
    }
}

async fn get_stream(State(state): State<AppState>, ws: WebSocketUpgrade) -> Response {
    let rx = state.hub.subscribe();
    ws.on_upgrade(move |socket| stream::serve_socket(socket, rx))
}

/// Spawn the background poller that drives `/api/stream`. Call once after building
/// the router (the poller shares the same `Store` and `StreamHub`).
pub fn spawn_stream_poller(state: &AppState) {
    let store = state.store.clone();
    let hub = state.hub.clone();
    tokio::spawn(async move {
        stream::poll_loop(store, hub, Duration::from_millis(STREAM_POLL)).await;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_iso_date_bounds_inclusively() {
        let r = DateRange {
            from: Some("2026-01-01".into()),
            to: Some("2026-12-31".into()),
        };
        let b = r.bounds();
        assert_eq!(b.from.unwrap().to_rfc3339(), "2026-01-01T00:00:00+00:00");
        // upper bound expands to the inclusive end of the day
        let to = b.to.unwrap();
        assert_eq!(to.date_naive().to_string(), "2026-12-31");
        assert_eq!(to.format("%H:%M:%S").to_string(), "23:59:59");
    }

    #[test]
    fn parses_rfc3339_bounds_verbatim() {
        let r = DateRange {
            from: Some("2026-06-01T12:30:00Z".into()),
            to: None,
        };
        let b = r.bounds();
        assert_eq!(b.from.unwrap().to_rfc3339(), "2026-06-01T12:30:00+00:00");
        assert!(b.to.is_none());
    }

    #[test]
    fn missing_or_blank_or_garbage_bounds_are_unconstrained() {
        let r = DateRange {
            from: Some("   ".into()),
            to: Some("not-a-date".into()),
        };
        let b = r.bounds();
        assert!(b.from.is_none());
        assert!(b.to.is_none());
        assert!(b.is_unbounded());

        let empty = DateRange::default().bounds();
        assert!(empty.is_unbounded());
    }
}
