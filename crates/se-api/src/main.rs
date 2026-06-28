//! `se-api` binary — boots the axum dashboard backend.
//!
//! Reads `DATABASE_URL` (required) and `SE_API_BIND` (default `0.0.0.0:8080`),
//! connects the store lazily, builds the router, spawns the `/api/stream` poller,
//! and serves until interrupted.

use std::net::SocketAddr;

use se_api::{router, spawn_stream_poller, AppState};
use se_store::Store;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,se_api=debug".into()),
        )
        .init();

    let database_url = std::env::var("DATABASE_URL").map_err(|_| {
        "DATABASE_URL must be set (e.g. postgres://swing:swing@localhost:5433/swing)"
    })?;
    let bind = std::env::var("SE_API_BIND").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    let addr: SocketAddr = bind
        .parse()
        .map_err(|e| format!("invalid SE_API_BIND `{bind}`: {e}"))?;

    // connect_lazy so the server boots even if the DB is briefly unavailable; the
    // first request (or the poller) establishes the connection.
    let store = Store::connect_lazy(&database_url)?;
    let state = AppState::new(store);
    spawn_stream_poller(&state);
    let app = router(state);

    tracing::info!(%addr, "se-api listening");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
