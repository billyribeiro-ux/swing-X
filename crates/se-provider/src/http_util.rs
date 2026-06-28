//! Small shared helpers for the live HTTP adapters (FMP/FRED).
//!
//! Kept private to the crate. Holds only pure, dependency-light utilities so the
//! adapter files stay focused on endpoint wiring + parsing.

use chrono::{DateTime, NaiveDate, TimeZone, Utc};

/// The session-close convention used across this repo: a NY trading day maps to
/// 21:00:00 UTC (≈ 16:00 ET when DST, 17:00 UTC offset aside this is the agreed
/// fixed convention the mock provider and store also use).
pub(crate) const SESSION_CLOSE_HOUR_UTC: u32 = 21;

/// Map a (naive) trading date to its UTC session-close timestamp.
pub(crate) fn session_close_ts(date: NaiveDate) -> DateTime<Utc> {
    Utc.from_utc_datetime(
        &date
            .and_hms_opt(SESSION_CLOSE_HOUR_UTC, 0, 0)
            .expect("21:00:00 is a valid time"),
    )
}

/// Parse a `YYYY-MM-DD` string (the date format both FMP and FRED return).
pub(crate) fn parse_ymd(s: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").ok()
}
