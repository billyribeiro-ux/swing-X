//! EVENT OVERLAY — calendar-derived modifiers (Layer::Event).
//!
//! These are NOT standalone signals: they are modifiers/sizing constraints that
//! tell downstream layers a decision bar sits on or near a structurally important
//! calendar date (options expiration, quad-witch, month/quarter end, the run-up to
//! an FOMC decision). Everything here is computed purely from `decision_ts` date
//! arithmetic plus a maintained list of scheduled FOMC dates — there is no market
//! data input and therefore NO leakage risk.
//!
//! All emitted features carry `LeadTimeTag::EndOfDay` and `source = "calendar"`.

use async_trait::async_trait;
use chrono::{Datelike, Duration, NaiveDate, Weekday};
use se_core::{AsOf, Feature, Layer, LeadTimeTag, Result};

use crate::module::{FeatureContext, FeatureModule};

/// Scheduled FOMC **decision** dates (the second/last day of each meeting),
/// 2018–2026. Maintained by hand from the Federal Reserve's published calendar.
/// `pre_fomc` fires on the 1–2 sessions BEFORE one of these dates.
///
/// NOTE: 2026 dates beyond those already published are the Fed's announced
/// schedule; revise if the Fed reschedules. Format: (year, month, day).
const FOMC_DATES: &[(i32, u32, u32)] = &[
    // 2018
    (2018, 1, 31),
    (2018, 3, 21),
    (2018, 5, 2),
    (2018, 6, 13),
    (2018, 8, 1),
    (2018, 9, 26),
    (2018, 11, 8),
    (2018, 12, 19),
    // 2019
    (2019, 1, 30),
    (2019, 3, 20),
    (2019, 5, 1),
    (2019, 6, 19),
    (2019, 7, 31),
    (2019, 9, 18),
    (2019, 10, 30),
    (2019, 12, 11),
    // 2020
    (2020, 1, 29),
    (2020, 3, 18),
    (2020, 4, 29),
    (2020, 6, 10),
    (2020, 7, 29),
    (2020, 9, 16),
    (2020, 11, 5),
    (2020, 12, 16),
    // 2021
    (2021, 1, 27),
    (2021, 3, 17),
    (2021, 4, 28),
    (2021, 6, 16),
    (2021, 7, 28),
    (2021, 9, 22),
    (2021, 11, 3),
    (2021, 12, 15),
    // 2022
    (2022, 1, 26),
    (2022, 3, 16),
    (2022, 5, 4),
    (2022, 6, 15),
    (2022, 7, 27),
    (2022, 9, 21),
    (2022, 11, 2),
    (2022, 12, 14),
    // 2023
    (2023, 2, 1),
    (2023, 3, 22),
    (2023, 5, 3),
    (2023, 6, 14),
    (2023, 7, 26),
    (2023, 9, 20),
    (2023, 11, 1),
    (2023, 12, 13),
    // 2024
    (2024, 1, 31),
    (2024, 3, 20),
    (2024, 5, 1),
    (2024, 6, 12),
    (2024, 7, 31),
    (2024, 9, 18),
    (2024, 11, 7),
    (2024, 12, 18),
    // 2025
    (2025, 1, 29),
    (2025, 3, 19),
    (2025, 5, 7),
    (2025, 6, 18),
    (2025, 7, 30),
    (2025, 9, 17),
    (2025, 10, 29),
    (2025, 12, 10),
    // 2026 (Fed's announced schedule)
    (2026, 1, 28),
    (2026, 3, 18),
    (2026, 4, 29),
    (2026, 6, 17),
    (2026, 7, 29),
    (2026, 9, 16),
    (2026, 10, 28),
    (2026, 12, 9),
];

/// Sessions before an FOMC date that count as the "pre-FOMC" run-up.
const PRE_FOMC_SESSIONS: i64 = 2;

/// The event-overlay module.
#[derive(Debug, Clone, Copy, Default)]
pub struct EventOverlay;

impl EventOverlay {
    pub fn new() -> Self {
        EventOverlay
    }
}

/// The 3rd Friday of `(year, month)` — monthly equity-options expiration.
fn third_friday(year: i32, month: u32) -> NaiveDate {
    let first = NaiveDate::from_ymd_opt(year, month, 1).expect("valid first-of-month");
    // Days to the first Friday (Mon=0 .. Sun=6; Fri=4).
    let offset = (Weekday::Fri.num_days_from_monday() as i64
        - first.weekday().num_days_from_monday() as i64)
        .rem_euclid(7);
    first + Duration::days(offset + 14)
}

/// Is `d` a monthly options-expiration Friday (the 3rd Friday)?
fn is_opex(d: NaiveDate) -> bool {
    d == third_friday(d.year(), d.month())
}

/// Is `d` a quad-witch (the 3rd Friday of Mar/Jun/Sep/Dec)?
fn is_quad_witch(d: NaiveDate) -> bool {
    matches!(d.month(), 3 | 6 | 9 | 12) && is_opex(d)
}

/// Signed sessions from `d` to its month's OPEX (3rd Friday). Positive BEFORE
/// (sessions remaining), 0 on, negative after. Uses calendar weekdays as a
/// session proxy (markets are closed weekends; holidays are not modeled — coarse
/// by design).
fn days_to_opex(d: NaiveDate) -> i64 {
    let opex = third_friday(d.year(), d.month());
    weekday_diff(d, opex)
}

/// Trading-day (weekday) signed difference `to - from`, skipping weekends:
/// positive when `to` is after `from`, negative when before, 0 when equal.
fn weekday_diff(from: NaiveDate, to: NaiveDate) -> i64 {
    if from == to {
        return 0;
    }
    let (lo, hi, sign) = if from < to {
        (from, to, 1)
    } else {
        (to, from, -1)
    };
    let mut count = 0i64;
    let mut cur = lo;
    while cur < hi {
        cur += Duration::days(1);
        if !matches!(cur.weekday(), Weekday::Sat | Weekday::Sun) {
            count += 1;
        }
    }
    sign * count
}

/// Last weekday (Mon–Fri) of `(year, month)` — month-end session proxy.
fn last_weekday_of_month(year: i32, month: u32) -> NaiveDate {
    let (ny, nm) = if month == 12 {
        (year + 1, 1)
    } else {
        (year, month + 1)
    };
    let first_next = NaiveDate::from_ymd_opt(ny, nm, 1).expect("valid first-of-next-month");
    let mut d = first_next - Duration::days(1);
    while matches!(d.weekday(), Weekday::Sat | Weekday::Sun) {
        d -= Duration::days(1);
    }
    d
}

fn is_month_end(d: NaiveDate) -> bool {
    d == last_weekday_of_month(d.year(), d.month())
}

fn is_quarter_end(d: NaiveDate) -> bool {
    matches!(d.month(), 3 | 6 | 9 | 12) && is_month_end(d)
}

/// Minimum signed weekday distance from `d` to ANY scheduled FOMC date.
/// Returns `None` if the list has no usable date.
fn nearest_fomc_diff(d: NaiveDate) -> Option<i64> {
    FOMC_DATES
        .iter()
        .filter_map(|&(y, m, day)| NaiveDate::from_ymd_opt(y, m, day))
        .map(|fomc| weekday_diff(d, fomc))
        .min_by_key(|diff| diff.abs())
}

/// 1.0 when `d` is within the 1..=PRE_FOMC_SESSIONS sessions BEFORE an FOMC date.
fn is_pre_fomc(d: NaiveDate) -> bool {
    FOMC_DATES
        .iter()
        .filter_map(|&(y, m, day)| NaiveDate::from_ymd_opt(y, m, day))
        .any(|fomc| {
            // diff > 0 means d is BEFORE fomc (sessions remaining).
            let diff = weekday_diff(d, fomc);
            (1..=PRE_FOMC_SESSIONS).contains(&diff)
        })
}

#[async_trait]
impl FeatureModule for EventOverlay {
    fn layer(&self) -> Layer {
        Layer::Event
    }

    fn name(&self) -> &str {
        "event_overlay"
    }

    async fn compute(&self, ctx: &FeatureContext<'_>) -> Result<Vec<Feature>> {
        let decision_ts = ctx.pit.decision_ts();
        let as_of = AsOf::new(decision_ts.inner());
        let d = decision_ts.inner().date_naive();
        let mut out: Vec<Feature> = Vec::new();

        let push = |out: &mut Vec<Feature>, key: &str, value: f64| {
            if value.is_finite() {
                out.push(Feature::new(
                    key,
                    value,
                    Layer::Event,
                    as_of,
                    LeadTimeTag::EndOfDay,
                    "calendar",
                ));
            }
        };

        push(
            &mut out,
            "event.is_opex",
            if is_opex(d) { 1.0 } else { 0.0 },
        );
        push(&mut out, "event.days_to_opex", days_to_opex(d) as f64);
        push(
            &mut out,
            "event.is_quad_witch",
            if is_quad_witch(d) { 1.0 } else { 0.0 },
        );
        push(
            &mut out,
            "event.is_month_end",
            if is_month_end(d) { 1.0 } else { 0.0 },
        );
        push(
            &mut out,
            "event.is_quarter_end",
            if is_quarter_end(d) { 1.0 } else { 0.0 },
        );
        push(
            &mut out,
            "event.pre_fomc",
            if is_pre_fomc(d) { 1.0 } else { 0.0 },
        );
        if let Some(diff) = nearest_fomc_diff(d) {
            push(&mut out, "event.days_to_fomc", diff as f64);
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn third_friday_known_dates() {
        // March 2024's 3rd Friday is the 15th (a known quad-witch).
        assert_eq!(third_friday(2024, 3), date(2024, 3, 15));
        // June 2024's 3rd Friday is the 21st.
        assert_eq!(third_friday(2024, 6), date(2024, 6, 21));
        // January 2025's 3rd Friday is the 17th.
        assert_eq!(third_friday(2025, 1), date(2025, 1, 17));
    }

    #[test]
    fn opex_and_quad_witch_detection() {
        // 2024-03-15 is opex AND quad-witch (March).
        assert!(is_opex(date(2024, 3, 15)));
        assert!(is_quad_witch(date(2024, 3, 15)));
        // 2024-01-19 is opex but NOT quad-witch (January).
        assert!(is_opex(date(2024, 1, 19)));
        assert!(!is_quad_witch(date(2024, 1, 19)));
        // A non-Friday mid-month is neither.
        assert!(!is_opex(date(2024, 3, 12)));
        assert!(!is_quad_witch(date(2024, 3, 12)));
        // September quad-witch 2024 is the 20th.
        assert!(is_quad_witch(date(2024, 9, 20)));
    }

    #[test]
    fn days_to_opex_sign() {
        // Wednesday before the 2024-03-15 opex -> positive (two sessions remain).
        assert_eq!(days_to_opex(date(2024, 3, 13)), 2);
        // On opex -> 0.
        assert_eq!(days_to_opex(date(2024, 3, 15)), 0);
        // The Monday after -> negative (opex is in the past).
        assert_eq!(days_to_opex(date(2024, 3, 18)), -1);
    }

    #[test]
    fn month_and_quarter_end() {
        // 2024-03-29 (Fri) is the last weekday of March -> month AND quarter end
        // (March 30/31 are Sat/Sun).
        assert!(is_month_end(date(2024, 3, 29)));
        assert!(is_quarter_end(date(2024, 3, 29)));
        // 2024-04-30 (Tue) is month end but NOT quarter end.
        assert!(is_month_end(date(2024, 4, 30)));
        assert!(!is_quarter_end(date(2024, 4, 30)));
        // Mid-month is neither.
        assert!(!is_month_end(date(2024, 4, 15)));
    }

    #[test]
    fn pre_fomc_window() {
        // FOMC decision 2024-03-20 (Wed). The 1-2 sessions before: Mon 18, Tue 19.
        assert!(is_pre_fomc(date(2024, 3, 18)));
        assert!(is_pre_fomc(date(2024, 3, 19)));
        // ON the decision date -> not "pre".
        assert!(!is_pre_fomc(date(2024, 3, 20)));
        // Three sessions before (Fri 15) is outside the 2-session window.
        assert!(!is_pre_fomc(date(2024, 3, 15)));
        // A date far from any FOMC meeting.
        assert!(!is_pre_fomc(date(2024, 2, 5)));
    }

    #[test]
    fn weekday_diff_skips_weekends() {
        // Fri -> following Mon is one trading session.
        assert_eq!(weekday_diff(date(2024, 3, 15), date(2024, 3, 18)), 1);
        // Symmetric sign.
        assert_eq!(weekday_diff(date(2024, 3, 18), date(2024, 3, 15)), -1);
        assert_eq!(weekday_diff(date(2024, 3, 15), date(2024, 3, 15)), 0);
    }
}
