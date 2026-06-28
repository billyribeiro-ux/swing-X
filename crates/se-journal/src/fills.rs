//! Realistic fill modeling: **next-bar-open or worse**, plus the horizon's round-trip cost.
//!
//! A signal is decided at the close of bar `T`; the earliest executable price is the OPEN of
//! bar `T+1`. We fill there, then make the price *adverse* by half the round-trip cost fraction
//! on each side (entry and exit) so the modeled fill is never better than reality. This is the
//! journal half of the system's cost discipline (the validator applies a cost on the metrics
//! side; here it bites the actual paper fills).

use se_core::{CostModel, Side};

/// Apply adverse slippage to a raw fill price. `cost_frac` is the FULL round-trip fraction;
/// each leg (entry/exit) eats half. Direction makes it worse: a long pays UP on entry and
/// receives LESS on exit; a short is the mirror.
pub fn adverse_price(raw: f64, side: Side, leg: Leg, cost: &CostModel) -> f64 {
    let half = cost.round_trip_frac() / 2.0;
    let worsen = match (side, leg) {
        // Long entry: pay more. Long exit: get less.
        (Side::Long, Leg::Entry) => 1.0 + half,
        (Side::Long, Leg::Exit) => 1.0 - half,
        // Short entry: sell lower (receive less). Short exit (buy to cover): pay more.
        (Side::Short, Leg::Entry) => 1.0 - half,
        (Side::Short, Leg::Exit) => 1.0 + half,
    };
    raw * worsen
}

/// Which leg of the round trip a fill belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Leg {
    Entry,
    Exit,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_entry_is_paid_up_exit_is_received_down() {
        let cost = CostModel::etf_default();
        let raw = 100.0;
        let entry = adverse_price(raw, Side::Long, Leg::Entry, &cost);
        let exit = adverse_price(raw, Side::Long, Leg::Exit, &cost);
        assert!(entry > raw, "long entry should be worse (higher)");
        assert!(exit < raw, "long exit should be worse (lower)");
    }

    #[test]
    fn short_is_mirrored() {
        let cost = CostModel::etf_default();
        let raw = 100.0;
        let entry = adverse_price(raw, Side::Short, Leg::Entry, &cost);
        let exit = adverse_price(raw, Side::Short, Leg::Exit, &cost);
        assert!(entry < raw, "short entry receives less");
        assert!(exit > raw, "short cover pays more");
    }

    #[test]
    fn zero_cost_is_identity() {
        let cost = CostModel {
            commission_per_share: 0.0,
            slippage_bps: 0.0,
            spread_bps: 0.0,
        };
        assert_eq!(adverse_price(100.0, Side::Long, Leg::Entry, &cost), 100.0);
    }
}
