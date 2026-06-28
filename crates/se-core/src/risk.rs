//! Operator-configurable + self-learnable risk geometry (stop / target placement).
//!
//! Today's labeler placed barriers as ATR multiples baked into the [`HorizonProfile`]. That
//! made the geometry that *declares a loss* a fixed constant. This module lifts it into a
//! first-class, evolvable object: the operator sets a starting [`RiskModel`] (the ground rules),
//! and — unless locked — the search EXPLORES risk geometries and the OOS scoreboard keeps
//! whatever maximizes cost-aware out-of-sample expectancy. Tighter stops trade win-rate for
//! R:R; the machine finds the trade-off.
//!
//! The three primitives:
//!
//!   * [`StopSpec`] — the stop distance from entry, as `N` ATRs, a fixed dollar amount, or a
//!     percent of the entry price.
//!   * [`TargetSpec`] — the target distance, same three forms plus [`TargetSpec::RMultiple`]
//!     (`N ×` the risk/stop distance), the natural way to express R:R.
//!   * [`RiskModel`] — one stop + one or two targets, with helpers to resolve concrete prices
//!     for a given entry/ATR/side.
//!
//! All math is pure and exhaustively unit-tested. The enums serialize with serde's
//! internally-tagged representation (`{"kind":"atr","mult":1.0}`) so they round-trip cleanly in
//! the genome jsonb and read well by eye.

use serde::{Deserialize, Serialize};
use std::str::FromStr;

use crate::{Error, HorizonProfile, Side};

/// How far the stop sits from the entry. The distance is always reported as a non-negative
/// magnitude; the side decides whether it is above or below entry (see [`RiskModel::stop_price`]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StopSpec {
    /// `N` ATRs from entry.
    Atr { mult: f64 },
    /// A fixed dollar distance from entry (e.g. `$5.35`).
    Fixed { dollars: f64 },
    /// A percent of the entry price (e.g. `2.5` => 2.5%).
    Percent { pct: f64 },
}

impl StopSpec {
    /// Convenience constructor matching the design's `Atr(f64)` shorthand.
    pub const fn atr(mult: f64) -> Self {
        StopSpec::Atr { mult }
    }
    pub const fn fixed(dollars: f64) -> Self {
        StopSpec::Fixed { dollars }
    }
    pub const fn percent(pct: f64) -> Self {
        StopSpec::Percent { pct }
    }

    /// The stop distance from entry, in price units, always `>= 0`.
    pub fn distance(&self, entry: f64, atr: f64) -> f64 {
        let d = match self {
            StopSpec::Atr { mult } => mult * atr,
            StopSpec::Fixed { dollars } => *dollars,
            StopSpec::Percent { pct } => pct / 100.0 * entry,
        };
        d.abs()
    }

    /// Short human form, e.g. `1.0ATR`, `$5.35`, `2.5%`.
    pub fn describe(&self) -> String {
        match self {
            StopSpec::Atr { mult } => format!("{mult}ATR"),
            StopSpec::Fixed { dollars } => format!("${dollars}"),
            StopSpec::Percent { pct } => format!("{pct}%"),
        }
    }
}

impl FromStr for StopSpec {
    type Err = Error;

    /// Parse `"atr:1.0"`, `"fixed:5.35"`, or `"pct:2.5"` (case-insensitive; `percent`/`$`/`%`
    /// shorthands also accepted).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (kind, raw) = split_spec(s)?;
        let val = parse_num(&raw, s)?;
        match kind.as_str() {
            "atr" => Ok(StopSpec::Atr { mult: val }),
            "fixed" | "dollar" | "dollars" | "usd" | "$" => Ok(StopSpec::Fixed { dollars: val }),
            "pct" | "percent" | "%" => Ok(StopSpec::Percent { pct: val }),
            other => Err(Error::Config(format!(
                "unknown stop kind '{other}' in '{s}' (want atr:|fixed:|pct:)"
            ))),
        }
    }
}

/// How far the target sits from the entry. Adds [`TargetSpec::RMultiple`] over the stop forms:
/// a target at `N ×` the risk distance, the natural way to pin R:R independent of volatility.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TargetSpec {
    Atr {
        mult: f64,
    },
    Fixed {
        dollars: f64,
    },
    Percent {
        pct: f64,
    },
    /// `N ×` the risk (entry->stop) distance.
    RMultiple {
        r: f64,
    },
}

impl TargetSpec {
    pub const fn atr(mult: f64) -> Self {
        TargetSpec::Atr { mult }
    }
    pub const fn fixed(dollars: f64) -> Self {
        TargetSpec::Fixed { dollars }
    }
    pub const fn percent(pct: f64) -> Self {
        TargetSpec::Percent { pct }
    }
    pub const fn r_multiple(r: f64) -> Self {
        TargetSpec::RMultiple { r }
    }

    /// The target distance from entry, in price units, always `>= 0`. `risk_dist` is the stop
    /// distance (one R) — used only by [`TargetSpec::RMultiple`].
    pub fn distance(&self, entry: f64, atr: f64, risk_dist: f64) -> f64 {
        let d = match self {
            TargetSpec::Atr { mult } => mult * atr,
            TargetSpec::Fixed { dollars } => *dollars,
            TargetSpec::Percent { pct } => pct / 100.0 * entry,
            TargetSpec::RMultiple { r } => r * risk_dist,
        };
        d.abs()
    }

    /// Short human form, e.g. `2.0R`, `2.0ATR`, `$10.00`, `3.0%`.
    pub fn describe(&self) -> String {
        match self {
            TargetSpec::Atr { mult } => format!("{mult}ATR"),
            TargetSpec::Fixed { dollars } => format!("${dollars}"),
            TargetSpec::Percent { pct } => format!("{pct}%"),
            TargetSpec::RMultiple { r } => format!("{r}R"),
        }
    }
}

impl FromStr for TargetSpec {
    type Err = Error;

    /// Parse `"atr:2.0"`, `"fixed:10.0"`, `"pct:3.0"`, or `"r:2.0"` (RMultiple).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (kind, raw) = split_spec(s)?;
        let val = parse_num(&raw, s)?;
        match kind.as_str() {
            "atr" => Ok(TargetSpec::Atr { mult: val }),
            "fixed" | "dollar" | "dollars" | "usd" | "$" => Ok(TargetSpec::Fixed { dollars: val }),
            "pct" | "percent" | "%" => Ok(TargetSpec::Percent { pct: val }),
            "r" | "rmultiple" | "r_multiple" => Ok(TargetSpec::RMultiple { r: val }),
            other => Err(Error::Config(format!(
                "unknown target kind '{other}' in '{s}' (want atr:|fixed:|pct:|r:)"
            ))),
        }
    }
}

/// A full risk geometry: one stop, one primary target, optionally a second target. This is the
/// unit the operator configures and the search evolves.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RiskModel {
    pub stop: StopSpec,
    pub target1: TargetSpec,
    pub target2: Option<TargetSpec>,
}

impl RiskModel {
    pub const fn new(stop: StopSpec, target1: TargetSpec, target2: Option<TargetSpec>) -> Self {
        RiskModel {
            stop,
            target1,
            target2,
        }
    }

    /// The default risk model used when an old persisted genome lacks the field. A sensible,
    /// horizon-agnostic geometry: a 1-ATR stop with 2R/3R targets (see `Genome` serde default).
    pub const fn default_const() -> Self {
        RiskModel {
            stop: StopSpec::Atr { mult: 1.0 },
            target1: TargetSpec::RMultiple { r: 2.0 },
            target2: Some(TargetSpec::RMultiple { r: 3.0 }),
        }
    }

    /// Reproduce the legacy profile-driven geometry exactly: ATR-mult stop/target1 from the
    /// profile, and a second target at `1.6 ×` the first target's R-multiple (mirroring the old
    /// signal's `target2 = 1.6 × target1_dist`). This keeps current behavior as the default.
    pub fn from_profile(profile: &HorizonProfile) -> Self {
        let stop_mult = profile.stop_atr_mult;
        let target_mult = profile.target_atr_mult;
        let r1 = if stop_mult > 0.0 {
            target_mult / stop_mult
        } else {
            2.0
        };
        RiskModel {
            stop: StopSpec::Atr { mult: stop_mult },
            target1: TargetSpec::Atr { mult: target_mult },
            target2: Some(TargetSpec::RMultiple { r: r1 * 1.6 }),
        }
    }

    /// The stop distance (one R) for a given entry/ATR, always `>= 0`.
    pub fn risk_distance(&self, entry: f64, atr: f64) -> f64 {
        self.stop.distance(entry, atr)
    }

    /// The concrete stop price. Long stops sit below entry, short stops above.
    pub fn stop_price(&self, entry: f64, atr: f64, side: Side) -> f64 {
        let dist = self.stop.distance(entry, atr);
        entry - side.sign() * dist
    }

    /// The concrete target prices `(target1, target2)`. Long targets sit above entry, short
    /// below. R-multiple targets resolve against the stop distance computed for the same inputs.
    pub fn target_prices(&self, entry: f64, atr: f64, side: Side) -> (f64, Option<f64>) {
        let risk = self.risk_distance(entry, atr);
        let s = side.sign();
        let t1 = entry + s * self.target1.distance(entry, atr, risk);
        let t2 = self
            .target2
            .map(|t| entry + s * t.distance(entry, atr, risk));
        (t1, t2)
    }

    /// Parse a risk model from operator config strings.
    pub fn parse(stop: &str, target1: &str, target2: Option<&str>) -> Result<Self, Error> {
        Ok(RiskModel {
            stop: stop.parse()?,
            target1: target1.parse()?,
            target2: target2.map(str::parse).transpose()?,
        })
    }

    /// Short human form, e.g. `stop=1.0ATR target=2.0R/3.0R` or `stop=$5.35 target=2.0R`.
    pub fn describe(&self) -> String {
        match self.target2 {
            Some(t2) => format!(
                "stop={} target={}/{}",
                self.stop.describe(),
                self.target1.describe(),
                t2.describe()
            ),
            None => format!(
                "stop={} target={}",
                self.stop.describe(),
                self.target1.describe()
            ),
        }
    }
}

impl Default for RiskModel {
    fn default() -> Self {
        RiskModel::default_const()
    }
}

/// Split a `"kind:value"` spec into its lowercased kind and the raw value string.
fn split_spec(s: &str) -> Result<(String, String), Error> {
    let trimmed = s.trim();
    let (kind, raw) = trimmed.split_once(':').ok_or_else(|| {
        Error::Config(format!(
            "risk spec '{s}' must be 'kind:value' (e.g. atr:1.0)"
        ))
    })?;
    Ok((
        kind.trim().to_ascii_lowercase(),
        raw.trim().trim_start_matches('$').to_string(),
    ))
}

fn parse_num(raw: &str, full: &str) -> Result<f64, Error> {
    raw.trim().parse::<f64>().map_err(|_| {
        Error::Config(format!(
            "risk spec '{full}' has a non-numeric value '{raw}'"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn stop_distance_is_nonnegative_for_each_kind() {
        assert!((StopSpec::atr(1.5).distance(100.0, 2.0) - 3.0).abs() < EPS);
        assert!((StopSpec::fixed(5.35).distance(100.0, 2.0) - 5.35).abs() < EPS);
        assert!((StopSpec::percent(2.5).distance(200.0, 2.0) - 5.0).abs() < EPS);
        // Negative configs still produce a magnitude.
        assert!(StopSpec::atr(-1.0).distance(100.0, 2.0) >= 0.0);
        assert!(StopSpec::fixed(-3.0).distance(100.0, 2.0) >= 0.0);
    }

    #[test]
    fn target_distance_including_r_multiple() {
        // risk_dist = 4.0 (one R). 2R target => 8.0.
        assert!((TargetSpec::r_multiple(2.0).distance(100.0, 2.0, 4.0) - 8.0).abs() < EPS);
        assert!((TargetSpec::atr(2.0).distance(100.0, 2.0, 4.0) - 4.0).abs() < EPS);
        assert!((TargetSpec::fixed(10.0).distance(100.0, 2.0, 4.0) - 10.0).abs() < EPS);
        assert!((TargetSpec::percent(3.0).distance(200.0, 2.0, 4.0) - 6.0).abs() < EPS);
    }

    #[test]
    fn stop_price_directionality() {
        let rm = RiskModel::new(StopSpec::atr(1.0), TargetSpec::r_multiple(2.0), None);
        // Long: entry 100, atr 2 -> stop below at 98.
        assert!((rm.stop_price(100.0, 2.0, Side::Long) - 98.0).abs() < EPS);
        // Short: stop above at 102.
        assert!((rm.stop_price(100.0, 2.0, Side::Short) - 102.0).abs() < EPS);
    }

    #[test]
    fn target_prices_r_multiple_against_stop_distance() {
        // stop 1ATR (=2 risk), target1 2R (=4), target2 3R (=6).
        let rm = RiskModel::new(
            StopSpec::atr(1.0),
            TargetSpec::r_multiple(2.0),
            Some(TargetSpec::r_multiple(3.0)),
        );
        let (t1, t2) = rm.target_prices(100.0, 2.0, Side::Long);
        assert!((t1 - 104.0).abs() < EPS);
        assert!((t2.unwrap() - 106.0).abs() < EPS);
        // Short flips direction.
        let (t1s, t2s) = rm.target_prices(100.0, 2.0, Side::Short);
        assert!((t1s - 96.0).abs() < EPS);
        assert!((t2s.unwrap() - 94.0).abs() < EPS);
    }

    #[test]
    fn fixed_dollar_stop_resolves() {
        let rm = RiskModel::new(StopSpec::fixed(5.35), TargetSpec::r_multiple(2.0), None);
        assert!((rm.stop_price(100.0, 99.0, Side::Long) - 94.65).abs() < EPS);
        // R-multiple target uses the $5.35 risk -> 2R = $10.70 above entry.
        let (t1, _) = rm.target_prices(100.0, 99.0, Side::Long);
        assert!((t1 - 110.70).abs() < EPS);
    }

    #[test]
    fn from_profile_preserves_legacy_geometry() {
        let p = HorizonProfile::swing(); // stop 1.0, target 2.0
        let rm = RiskModel::from_profile(&p);
        assert_eq!(rm.stop, StopSpec::Atr { mult: 1.0 });
        assert_eq!(rm.target1, TargetSpec::Atr { mult: 2.0 });
        // target1 distance == old labeler's 2R target; target2 at 1.6x.
        let (t1, t2) = rm.target_prices(100.0, 1.0, Side::Long);
        assert!((t1 - 102.0).abs() < EPS, "t1={t1}");
        // r1 = 2.0/1.0 = 2.0; target2 = 1.6*2.0 = 3.2R = 3.2 price => 103.2.
        assert!((t2.unwrap() - 103.2).abs() < EPS, "t2={:?}", t2);
    }

    #[test]
    fn stop_spec_from_str() {
        assert_eq!("atr:1.0".parse::<StopSpec>().unwrap(), StopSpec::atr(1.0));
        assert_eq!(
            "fixed:5.35".parse::<StopSpec>().unwrap(),
            StopSpec::fixed(5.35)
        );
        assert_eq!(
            "pct:2.5".parse::<StopSpec>().unwrap(),
            StopSpec::percent(2.5)
        );
        // Case-insensitive + $ shorthand.
        assert_eq!("ATR:2".parse::<StopSpec>().unwrap(), StopSpec::atr(2.0));
        assert_eq!(
            "fixed:$5.00".parse::<StopSpec>().unwrap(),
            StopSpec::fixed(5.0)
        );
        assert!("bogus:1".parse::<StopSpec>().is_err());
        assert!("atr:notanumber".parse::<StopSpec>().is_err());
        assert!("noseparator".parse::<StopSpec>().is_err());
    }

    #[test]
    fn target_spec_from_str_including_r() {
        assert_eq!(
            "r:2.0".parse::<TargetSpec>().unwrap(),
            TargetSpec::r_multiple(2.0)
        );
        assert_eq!("atr:2".parse::<TargetSpec>().unwrap(), TargetSpec::atr(2.0));
        assert_eq!(
            "pct:3".parse::<TargetSpec>().unwrap(),
            TargetSpec::percent(3.0)
        );
        assert!("r:bad".parse::<TargetSpec>().is_err());
    }

    #[test]
    fn risk_model_parse() {
        let rm = RiskModel::parse("atr:1.0", "r:2.0", Some("r:3.0")).unwrap();
        assert_eq!(rm.stop, StopSpec::atr(1.0));
        assert_eq!(rm.target1, TargetSpec::r_multiple(2.0));
        assert_eq!(rm.target2, Some(TargetSpec::r_multiple(3.0)));
        let rm2 = RiskModel::parse("fixed:5.00", "r:2.0", None).unwrap();
        assert_eq!(rm2.stop, StopSpec::fixed(5.0));
        assert!(rm2.target2.is_none());
    }

    #[test]
    fn describe_forms() {
        let rm = RiskModel::new(
            StopSpec::atr(1.0),
            TargetSpec::r_multiple(2.0),
            Some(TargetSpec::r_multiple(3.0)),
        );
        assert_eq!(rm.describe(), "stop=1ATR target=2R/3R");
        let rm2 = RiskModel::new(StopSpec::fixed(5.35), TargetSpec::r_multiple(2.0), None);
        assert_eq!(rm2.describe(), "stop=$5.35 target=2R");
    }

    #[test]
    fn serde_round_trip_is_readable() {
        let rm = RiskModel::new(
            StopSpec::atr(1.5),
            TargetSpec::r_multiple(2.0),
            Some(TargetSpec::fixed(10.0)),
        );
        let json = serde_json::to_string(&rm).unwrap();
        // Internally tagged + snake_case kinds.
        assert!(json.contains("\"kind\":\"atr\""), "json={json}");
        assert!(json.contains("\"kind\":\"r_multiple\""), "json={json}");
        let back: RiskModel = serde_json::from_str(&json).unwrap();
        assert_eq!(rm, back);
    }

    #[test]
    fn default_const_matches_default_trait() {
        assert_eq!(RiskModel::default(), RiskModel::default_const());
    }
}
