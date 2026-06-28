//! `se-orchestrator` — the nightly walk-forward loop as a declarative job graph.
//!
//! This crate defines the canonical ORDER and intent of the nightly loop (§0); the
//! executor is `se nightly` (in `se-cli`), which runs each [`Step`] against the live
//! store + ML worker. Production scheduling wraps `se nightly` with cron/apalis — the
//! loop body is the same whether fired by hand or by a scheduler.
//!
//! ```text
//! nightly:
//!   1. ingest the new session into the PIT store
//!   2. roll the walk-forward window forward, SEARCH/mutate candidates
//!   3. FIT in-sample, SCORE on the OOS slice (the scoreboard) — KEEP/MUTATE/KILL
//!   4. surface signals from promoted strategies; journal paper fills
//!   5. MONITOR live/paper vs expectation; ADAPT anything decaying
//!   6. write the weekly changelog of what decayed / retired / adapted
//! ```

use std::fmt;

/// One executable stage of the nightly loop. Several conceptual sub-steps (roll
/// window, fit, score, keep/kill, adapt) are encapsulated inside `Search`/`Monitor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Ingest the new session (bars + macro) into the PIT store.
    Ingest,
    /// Roll the walk-forward window; SEARCH/mutate; FIT in-sample; SCORE on OOS; KEEP/MUTATE/KILL.
    Search,
    /// Surface executable signals from promoted strategies.
    Signals,
    /// Journal paper fills (next-bar-open or worse) and realized R.
    Journal,
    /// Run the forward-adaptation monitor (detect → act → log → alert) and ADAPT.
    Monitor,
    /// Write the weekly changelog of what decayed / retired / adapted.
    Changelog,
}

impl Step {
    pub const fn label(self) -> &'static str {
        match self {
            Step::Ingest => "ingest",
            Step::Search => "search",
            Step::Signals => "signals",
            Step::Journal => "journal",
            Step::Monitor => "monitor",
            Step::Changelog => "changelog",
        }
    }

    pub const fn detail(self) -> &'static str {
        match self {
            Step::Ingest => "pull the new session's bars + macro into the PIT store",
            Step::Search => "roll window → search/mutate → fit(IS) → score(OOS) → keep/mutate/kill",
            Step::Signals => {
                "surface signals from promoted strategies (entry/stop/target/attribution)"
            }
            Step::Journal => "open/resolve paper trades with next-bar-open fills",
            Step::Monitor => "forward-adaptation detectors → automatic actions",
            Step::Changelog => "summarize what decayed / retired / adapted this week",
        }
    }
}

impl fmt::Display for Step {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// The canonical nightly loop, in execution order.
pub const NIGHTLY: [Step; 6] = [
    Step::Ingest,
    Step::Search,
    Step::Signals,
    Step::Journal,
    Step::Monitor,
    Step::Changelog,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_is_ordered_and_complete() {
        assert_eq!(NIGHTLY.first(), Some(&Step::Ingest));
        assert_eq!(NIGHTLY.last(), Some(&Step::Changelog));
        assert_eq!(NIGHTLY.len(), 6);
        for s in NIGHTLY {
            assert!(!s.label().is_empty() && !s.detail().is_empty());
        }
    }
}
