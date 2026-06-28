//! WebSocket push for `/api/stream`.
//!
//! Design (v1): a single background [`poll_loop`] polls the DB on a fixed interval
//! for rows newer than the high-water marks it has already seen, converts them to
//! DTOs, and publishes JSON envelopes on a `broadcast` channel ([`StreamHub`]). Each
//! connected socket subscribes and forwards frames. Simple and good enough for a
//! single-operator dashboard; see the crate docs for the upgrade path.

use std::time::Duration;

use axum::extract::ws::{Message, WebSocket};
use chrono::{DateTime, Utc};
use se_store::sqlx;
use se_store::sqlx::Row;
use se_store::Store;
use serde::Serialize;
use tokio::sync::broadcast;
use uuid::Uuid;

use crate::dto::{monitor_action_dto, MonitorEventDto, SignalDto};

/// Poll interval for the stream poller, in milliseconds.
pub const STREAM_POLL: u64 = 2000;
/// Channel capacity. Slow sockets that lag this far behind get a `Lagged` and skip
/// ahead rather than blocking the publisher.
const CHANNEL_CAP: usize = 256;

/// A JSON envelope pushed over the socket. `kind` discriminates the payload.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamFrame {
    pub kind: String,
    pub data: serde_json::Value,
}

/// Broadcast hub shared by the poller (publisher) and every socket (subscriber).
#[derive(Clone)]
pub struct StreamHub {
    tx: broadcast::Sender<StreamFrame>,
}

impl Default for StreamHub {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamHub {
    pub fn new() -> Self {
        let (tx, _rx) = broadcast::channel(CHANNEL_CAP);
        StreamHub { tx }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<StreamFrame> {
        self.tx.subscribe()
    }

    /// Publish a frame. Errors (no subscribers) are ignored — the poller keeps
    /// running so a later-connecting socket still gets future frames.
    fn publish(&self, frame: StreamFrame) {
        let _ = self.tx.send(frame);
    }
}

/// Serve one WebSocket: forward every broadcast frame as text JSON until the client
/// disconnects or the channel closes. Lag (slow client) is tolerated by skipping.
pub async fn serve_socket(mut socket: WebSocket, mut rx: broadcast::Receiver<StreamFrame>) {
    loop {
        match rx.recv().await {
            Ok(frame) => {
                let text = match serde_json::to_string(&frame) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                if socket.send(Message::text(text)).await.is_err() {
                    break; // client gone
                }
            }
            Err(broadcast::error::RecvError::Lagged(_)) => continue,
            Err(broadcast::error::RecvError::Closed) => break,
        }
    }
}

/// Background loop: every `interval`, push DB rows newer than the last-seen marks.
/// Initializes its high-water marks to "now-ish" by reading the current maxima so
/// it only streams rows that appear *after* startup (no replay storm on connect).
pub async fn poll_loop(store: Store, hub: StreamHub, interval: Duration) {
    let mut last_signal: DateTime<Utc> = max_signal_ts(&store).await.unwrap_or_else(Utc::now);
    let mut last_event: DateTime<Utc> = max_event_ts(&store).await.unwrap_or_else(Utc::now);
    let mut ticker = tokio::time::interval(interval);
    loop {
        ticker.tick().await;
        if let Ok((sigs, hw)) = new_signals(&store, last_signal).await {
            if let Some(hw) = hw {
                last_signal = hw;
            }
            for s in sigs {
                hub.publish(StreamFrame {
                    kind: "signal".into(),
                    data: serde_json::to_value(&s).unwrap_or(serde_json::Value::Null),
                });
            }
        }
        if let Ok((events, hw)) = new_events(&store, last_event).await {
            if let Some(hw) = hw {
                last_event = hw;
            }
            for e in events {
                hub.publish(StreamFrame {
                    kind: "monitor".into(),
                    data: serde_json::to_value(&e).unwrap_or(serde_json::Value::Null),
                });
            }
        }
    }
}

async fn max_signal_ts(store: &Store) -> Option<DateTime<Utc>> {
    sqlx::query_scalar::<_, Option<DateTime<Utc>>>("SELECT MAX(created_at) FROM signals")
        .fetch_one(store.pool())
        .await
        .ok()
        .flatten()
}

async fn max_event_ts(store: &Store) -> Option<DateTime<Utc>> {
    sqlx::query_scalar::<_, Option<DateTime<Utc>>>("SELECT MAX(ts) FROM monitor_events")
        .fetch_one(store.pool())
        .await
        .ok()
        .flatten()
}

/// Signals created strictly after `since`, plus the new high-water mark.
async fn new_signals(
    store: &Store,
    since: DateTime<Utc>,
) -> Result<(Vec<SignalDto>, Option<DateTime<Utc>>), sqlx::Error> {
    let rows = sqlx::query(
        "SELECT signal_id, strategy_id, ticker, side, decision_ts, horizon, entry, stop, \
                target1, target2, rr1, rr2, conviction, cohort_n, regime_desc, why, \
                invalidation, cohort_expectancy, cvar5, lead_time, payload_json, created_at \
         FROM signals WHERE created_at > $1 ORDER BY created_at ASC LIMIT 100",
    )
    .bind(since)
    .fetch_all(store.pool())
    .await?;
    let mut hw = None;
    let mut out = Vec::with_capacity(rows.len());
    for row in &rows {
        let created: DateTime<Utc> = row.get("created_at");
        hw = Some(created);
        out.push(signal_dto_from_row(row));
    }
    Ok((out, hw))
}

/// Monitor events with `ts` strictly after `since`, plus the new high-water mark.
async fn new_events(
    store: &Store,
    since: DateTime<Utc>,
) -> Result<(Vec<MonitorEventDto>, Option<DateTime<Utc>>), sqlx::Error> {
    let rows = sqlx::query(
        "SELECT ts, detector, ticker, strategy_id, metric_value, threshold, action_taken, detail \
         FROM monitor_events WHERE ts > $1 ORDER BY ts ASC LIMIT 100",
    )
    .bind(since)
    .fetch_all(store.pool())
    .await?;
    let mut hw = None;
    let mut out = Vec::with_capacity(rows.len());
    for row in &rows {
        let ts: DateTime<Utc> = row.get("ts");
        hw = Some(ts);
        out.push(event_dto_from_row(row));
    }
    Ok((out, hw))
}

// Local row->DTO converters (mirror queries.rs; kept here so the poller is
// self-contained and the SQL column lists live next to their use).

fn signal_dto_from_row(row: &sqlx::postgres::PgRow) -> SignalDto {
    let why: serde_json::Value = row.try_get("why").unwrap_or(serde_json::Value::Null);
    let payload_json: serde_json::Value = row
        .try_get("payload_json")
        .unwrap_or(serde_json::Value::Null);
    let lead_time: Option<f64> = row
        .try_get::<Option<String>, _>("lead_time")
        .ok()
        .flatten()
        .and_then(|s| {
            let d: String = s
                .trim()
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                .collect();
            d.parse::<f64>().ok()
        });
    SignalDto {
        signal_id: row.get::<Uuid, _>("signal_id").to_string(),
        strategy_id: row.get::<Uuid, _>("strategy_id").to_string(),
        ticker: row.get("ticker"),
        side: row.get("side"),
        decision_ts: row.get::<DateTime<Utc>, _>("decision_ts").to_rfc3339(),
        horizon: row.get("horizon"),
        entry: row.get("entry"),
        stop: row.get("stop"),
        target1: row.get("target1"),
        target2: row.try_get("target2").ok().flatten(),
        rr1: row.try_get("rr1").ok().flatten(),
        rr2: row.try_get("rr2").ok().flatten(),
        conviction: row.get("conviction"),
        cohort_n: row.get::<i32, _>("cohort_n") as i64,
        regime_desc: row.get("regime_desc"),
        why: why
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|d| {
                        Some(crate::dto::DriverDto {
                            layer: d.get("layer")?.as_str()?.to_string(),
                            key: d.get("key")?.as_str()?.to_string(),
                            contribution: d
                                .get("contribution")
                                .and_then(|x| x.as_f64())
                                .unwrap_or(0.0),
                            detail: d
                                .get("detail")
                                .and_then(|x| x.as_str())
                                .unwrap_or("")
                                .to_string(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default(),
        invalidation: row.get("invalidation"),
        cohort_expectancy: row.try_get("cohort_expectancy").ok().flatten(),
        cvar5: row.try_get("cvar5").ok().flatten(),
        lead_time,
        payload_json,
    }
}

fn event_dto_from_row(row: &sqlx::postgres::PgRow) -> MonitorEventDto {
    let detail: serde_json::Value = row.try_get("detail").unwrap_or(serde_json::Value::Null);
    let detail_str = detail
        .get("note")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            if detail.is_null() {
                String::new()
            } else {
                detail.to_string()
            }
        });
    MonitorEventDto {
        ts: row.get::<DateTime<Utc>, _>("ts").to_rfc3339(),
        detector: row.get("detector"),
        ticker: row.try_get("ticker").ok().flatten(),
        strategy_id: row
            .try_get::<Option<Uuid>, _>("strategy_id")
            .ok()
            .flatten()
            .map(|u| u.to_string()),
        metric_value: row.try_get("metric_value").ok().flatten(),
        threshold: row.try_get("threshold").ok().flatten(),
        action_taken: monitor_action_dto(&row.get::<String, _>("action_taken")).to_string(),
        detail: detail_str,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn hub_publish_reaches_subscribers() {
        let hub = StreamHub::new();
        let mut rx = hub.subscribe();
        hub.publish(StreamFrame {
            kind: "signal".into(),
            data: json!({ "x": 1 }),
        });
        let got = rx.try_recv().expect("frame delivered");
        assert_eq!(got.kind, "signal");
    }

    #[test]
    fn frame_serializes_camelcase() {
        let f = StreamFrame {
            kind: "monitor".into(),
            data: json!({ "a": 1 }),
        };
        let v = serde_json::to_value(&f).unwrap();
        assert_eq!(v.get("kind").unwrap(), "monitor");
        assert!(v.get("data").is_some());
    }
}
