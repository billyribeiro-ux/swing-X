//! Horizon as a config axis. Every "swing constant" — barrier widths, time
//! barrier, sampling cadence, costs — lives in a [`HorizonProfile`], never
//! hardcoded. Re-weighting the SAME feature taxonomy under a different profile
//! is how SHORT_SWING / DAY / 0DTE / SCALP plug in (P8).
//!
//! Critical coupling (risk #4): the CPCV **purge length equals the label horizon**
//! (`max_hold_bars`). [`HorizonProfile::purge_bars`] is the single source of truth
//! for that, so no horizon can silently desync labeling from cross-validation.

use serde::{Deserialize, Serialize};
use std::str::FromStr;

use crate::error::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Horizon {
    /// 2–15 sessions (v1 default).
    Swing,
    ShortSwing,
    Day,
    ZeroDte,
    Scalp,
}

impl Horizon {
    pub const ALL: [Horizon; 5] = [
        Horizon::Swing,
        Horizon::ShortSwing,
        Horizon::Day,
        Horizon::ZeroDte,
        Horizon::Scalp,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Horizon::Swing => "swing",
            Horizon::ShortSwing => "short_swing",
            Horizon::Day => "day",
            Horizon::ZeroDte => "zero_dte",
            Horizon::Scalp => "scalp",
        }
    }
}

impl FromStr for Horizon {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let low = s.trim().to_ascii_lowercase().replace(['-', ' '], "_");
        Horizon::ALL
            .into_iter()
            .find(|h| h.as_str() == low)
            .ok_or_else(|| Error::Config(format!("unknown horizon: {s}")))
    }
}

/// Bar sampling cadence for a horizon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cadence {
    Daily,
    Hourly,
    Min30,
    Min15,
    Min5,
    Min1,
}

impl Cadence {
    /// Approximate minutes per bar (calendar minutes; sessions are handled upstream).
    pub const fn minutes(self) -> u32 {
        match self {
            Cadence::Daily => 390, // one RTH session
            Cadence::Hourly => 60,
            Cadence::Min30 => 30,
            Cadence::Min15 => 15,
            Cadence::Min5 => 5,
            Cadence::Min1 => 1,
        }
    }
}

/// Cost assumptions applied adversely at fill time (`se-journal` fills next-bar-open or worse).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CostModel {
    pub commission_per_share: f64,
    /// Adverse slippage in basis points of price.
    pub slippage_bps: f64,
    /// Half-spread crossed on entry/exit, in basis points.
    pub spread_bps: f64,
}

impl CostModel {
    pub const fn etf_default() -> Self {
        CostModel {
            commission_per_share: 0.0, // commission-free retail ETF assumption
            slippage_bps: 1.0,
            spread_bps: 1.0,
        }
    }

    /// Round-trip cost as a fraction of notional (entry + exit).
    pub fn round_trip_frac(&self) -> f64 {
        2.0 * (self.slippage_bps + self.spread_bps) / 10_000.0
    }
}

/// All horizon-dependent constants in one place.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HorizonProfile {
    pub horizon: Horizon,
    pub cadence: Cadence,
    /// Minimum bars a position is held before the time barrier can fire.
    pub min_hold_bars: u32,
    /// Time barrier in bars — this IS the label horizon and the CPCV purge length.
    pub max_hold_bars: u32,
    /// Profit-target barrier in ATR multiples.
    pub target_atr_mult: f64,
    /// Stop barrier in ATR multiples.
    pub stop_atr_mult: f64,
    /// Lookback (bars) for the ATR that sizes the barriers.
    pub atr_lookback: u32,
    /// Embargo (bars) applied after each CPCV test fold.
    pub embargo_bars: u32,
    pub cost: CostModel,
}

impl HorizonProfile {
    /// The CPCV purge length. By construction it equals the label horizon so
    /// labeling and cross-validation can never desync (see module docs, risk #4).
    pub const fn purge_bars(&self) -> u32 {
        self.max_hold_bars
    }

    pub fn for_horizon(h: Horizon) -> Self {
        match h {
            Horizon::Swing => Self::swing(),
            Horizon::ShortSwing => Self::short_swing(),
            Horizon::Day => Self::day(),
            Horizon::ZeroDte => Self::zero_dte(),
            Horizon::Scalp => Self::scalp(),
        }
    }

    /// 2–15 sessions, daily bars (v1 default).
    pub const fn swing() -> Self {
        HorizonProfile {
            horizon: Horizon::Swing,
            cadence: Cadence::Daily,
            min_hold_bars: 2,
            max_hold_bars: 15,
            target_atr_mult: 2.0,
            stop_atr_mult: 1.0,
            atr_lookback: 14,
            embargo_bars: 3,
            cost: CostModel::etf_default(),
        }
    }

    pub const fn short_swing() -> Self {
        HorizonProfile {
            horizon: Horizon::ShortSwing,
            cadence: Cadence::Hourly,
            min_hold_bars: 2,
            max_hold_bars: 8,
            target_atr_mult: 1.5,
            stop_atr_mult: 0.8,
            atr_lookback: 14,
            embargo_bars: 2,
            cost: CostModel::etf_default(),
        }
    }

    pub const fn day() -> Self {
        HorizonProfile {
            horizon: Horizon::Day,
            cadence: Cadence::Min15,
            min_hold_bars: 2,
            max_hold_bars: 13, // ~one session of 15m bars
            target_atr_mult: 1.2,
            stop_atr_mult: 0.7,
            atr_lookback: 14,
            embargo_bars: 2,
            cost: CostModel::etf_default(),
        }
    }

    pub const fn zero_dte() -> Self {
        HorizonProfile {
            horizon: Horizon::ZeroDte,
            cadence: Cadence::Min5,
            min_hold_bars: 1,
            max_hold_bars: 12,
            target_atr_mult: 1.0,
            stop_atr_mult: 0.6,
            atr_lookback: 14,
            embargo_bars: 1,
            cost: CostModel::etf_default(),
        }
    }

    pub const fn scalp() -> Self {
        HorizonProfile {
            horizon: Horizon::Scalp,
            cadence: Cadence::Min1,
            min_hold_bars: 1,
            max_hold_bars: 10,
            target_atr_mult: 0.8,
            stop_atr_mult: 0.5,
            atr_lookback: 14,
            embargo_bars: 1,
            cost: CostModel::etf_default(),
        }
    }
}

impl Default for HorizonProfile {
    fn default() -> Self {
        HorizonProfile::swing()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn purge_equals_label_horizon() {
        for h in Horizon::ALL {
            let p = HorizonProfile::for_horizon(h);
            assert_eq!(
                p.purge_bars(),
                p.max_hold_bars,
                "purge must equal label horizon for {h:?}"
            );
        }
    }

    #[test]
    fn horizon_parse_roundtrip() {
        for h in Horizon::ALL {
            assert_eq!(h.as_str().parse::<Horizon>().unwrap(), h);
        }
        assert_eq!("0dte".replace('0', "zero_"), "zero_dte");
        assert_eq!(
            "short-swing".parse::<Horizon>().unwrap(),
            Horizon::ShortSwing
        );
    }
}
