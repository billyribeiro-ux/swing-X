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
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::get;
use axum::Router;
use se_store::Store;
use tower_http::cors::CorsLayer;

pub use stream::{StreamHub, STREAM_POLL};

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

async fn get_signals(State(state): State<AppState>) -> Response {
    match queries::signals(&state.store).await {
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

async fn get_population(State(state): State<AppState>) -> Response {
    match queries::population(&state.store).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => err_response(e),
    }
}

async fn get_monitor(State(state): State<AppState>) -> Response {
    match queries::monitor_events(&state.store).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => err_response(e),
    }
}

async fn get_journal(State(state): State<AppState>) -> Response {
    match queries::journal(&state.store).await {
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
