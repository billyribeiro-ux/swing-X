//! The transparent, rule-based v1 regime classifier.
//!
//! Input is the Layer-1 regime feature vector (`BTreeMap<String, f64>`); output is
//! a [`RegimeAssessment`] — a label, a softmax-normalized probability map over the
//! candidate labels, a confidence (the top probability), and a count of observed
//! inputs. Every rule is interpretable so the assignment can be sanity-checked
//! against known historical events (COVID crash, 2022 bear, calm uptrends).
//!
//! ## Design
//! Each candidate [`RegimeLabel`] accumulates an *evidence score* from the signals
//! that are actually present (missing signals contribute nothing — no zero-fill).
//! Scores are turned into probabilities with a softmax. We deliberately keep the
//! label set the classifier can emit to the **observable** subset:
//! `VolExpansion`, `VolCompression`, `RiskOff`, `RiskOn`, `ShortGamma`, `LongGamma`,
//! `Transition`, and the first-class `OutOfDistribution`.
//!
//! ## Proxies (gaps made explicit)
//!
//! Without the proprietary GEX feed we cannot observe dealer gamma directly, so
//! `ShortGamma` / `LongGamma` are approximated from the vol regime + VVIX:
//! high realized-vol percentile + backwardation + high VVIX is short-gamma-like
//! (trending / unstable); low realized-vol + contango + low VVIX is long-gamma-like
//! (pinned / mean-reverting). These gamma labels carry LOWER weight than the
//! vol/risk labels and are only surfaced when the direct GEX sign is absent. When
//! `regime.gex_sign` IS present (proprietary feed wired) it dominates the gamma
//! decision.
//!
//! ## Out-of-distribution
//! We return [`RegimeLabel::OutOfDistribution`] when too few core signals are
//! observed (default `< 3`) OR a core signal sits wildly outside its historical
//! envelope (e.g. VIX term-structure ratio or rv-percentile far out of range),
//! so the system suppresses rather than guesses.

use std::collections::BTreeMap;

use se_core::RegimeLabel;
use serde::{Deserialize, Serialize};

/// Tunable thresholds for the rule-based classifier. Defaults are documented and
/// chosen to be interpretable rather than fitted (this is a v1, pre-ML baseline).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ClassifierConfig {
    /// Minimum number of CORE signals that must be observed to classify at all.
    pub min_core_inputs: usize,
    /// rv_percentile at/above this is "high" (vol expanding).
    pub rv_pctl_high: f64,
    /// rv_percentile at/below this is "low" (vol compressed).
    pub rv_pctl_low: f64,
    /// vix_vix3m at/above this is backwardation (front-month stress).
    pub backwardation_hi: f64,
    /// vix_vix3m at/below this is contango (calm).
    pub contango_lo: f64,
    /// HY OAS (percentage points) at/above this is "credit wide" (risk-off).
    pub hy_oas_wide: f64,
    /// HY OAS at/below this is "credit tight" (risk-on).
    pub hy_oas_tight: f64,
    /// VVIX at/above this is elevated vol-of-vol.
    pub vvix_high: f64,
    /// VVIX at/below this is calm vol-of-vol.
    pub vvix_low: f64,
    /// |copper/gold or DXY trend| below this is "flat" (no cross-asset signal).
    pub trend_flat: f64,
    /// Softmax temperature; lower = sharper (more confident) distribution.
    pub softmax_temp: f64,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        ClassifierConfig {
            min_core_inputs: 3,
            rv_pctl_high: 0.80,
            rv_pctl_low: 0.30,
            backwardation_hi: 1.00, // VIX >= VIX3M -> curve inverted/flat = stress
            contango_lo: 0.95,      // VIX comfortably below VIX3M = calm
            hy_oas_wide: 5.0,       // HY OAS >= ~500bps historically = stress
            hy_oas_tight: 3.5,      // HY OAS <= ~350bps = benign credit
            vvix_high: 110.0,
            vvix_low: 90.0,
            trend_flat: 0.005, // 0.5% over the trend window
            softmax_temp: 0.60,
        }
    }
}

/// The classifier's output for one decision bar.
#[derive(Debug, Clone, PartialEq)]
pub struct RegimeAssessment {
    pub label: RegimeLabel,
    /// Softmax-normalized probability over the candidate labels (sums to ~1).
    pub prob_map: BTreeMap<RegimeLabel, f64>,
    /// Top probability — the model's confidence in `label`.
    pub confidence: f64,
    /// How many regime signals were actually observed (drives OOD).
    pub observed_inputs: usize,
}

impl RegimeAssessment {
    /// `prob_map` as a JSON object keyed by the regime string form, for storage.
    pub fn prob_map_json(&self) -> serde_json::Value {
        let obj: serde_json::Map<String, serde_json::Value> = self
            .prob_map
            .iter()
            .map(|(k, v)| (k.as_str().to_string(), serde_json::json!(v)))
            .collect();
        serde_json::Value::Object(obj)
    }
}

/// The rule-based v1 classifier.
#[derive(Debug, Clone, Copy, Default)]
pub struct RegimeClassifier {
    pub cfg: ClassifierConfig,
}

/// CORE signals: their presence/absence drives the OOD gate. These are the
/// observable backbone of the regime read on this data tier.
const CORE_SIGNALS: &[&str] = &[
    "regime.rv_percentile",
    "regime.vix_vix3m",
    "regime.vvix",
    "regime.rv20",
    "regime.copper_gold",
    "regime.dxy_trend",
];

impl RegimeClassifier {
    pub fn new(cfg: ClassifierConfig) -> Self {
        RegimeClassifier { cfg }
    }

    /// Count of CORE signals present in the feature map.
    fn core_count(&self, f: &BTreeMap<String, f64>) -> usize {
        CORE_SIGNALS
            .iter()
            .filter(|k| f.get(**k).map(|v| v.is_finite()).unwrap_or(false))
            .count()
    }

    /// Detect a CORE signal that is wildly outside any plausible historical
    /// envelope -> treat as out-of-distribution rather than guessing.
    fn out_of_bounds(&self, f: &BTreeMap<String, f64>) -> Option<String> {
        let checks: &[(&str, f64, f64)] = &[
            // (key, lo, hi) generous historical envelopes.
            ("regime.vix_vix3m", 0.4, 3.0),
            ("regime.vix9d_vix", 0.4, 3.0),
            ("regime.rv_percentile", 0.0, 1.0),
            ("regime.rv20", 0.0, 5.0), // 500% annualized rv is absurd
            ("regime.vvix", 40.0, 280.0),
        ];
        for (key, lo, hi) in checks {
            if let Some(&v) = f.get(*key) {
                if !v.is_finite() || v < *lo || v > *hi {
                    return Some(format!("{key}={v} outside [{lo}, {hi}]"));
                }
            }
        }
        None
    }

    /// Classify a regime feature map into a [`RegimeAssessment`].
    pub fn classify(&self, f: &BTreeMap<String, f64>) -> RegimeAssessment {
        let cfg = &self.cfg;
        let observed = f.values().filter(|v| v.is_finite()).count();
        let core = self.core_count(f);

        // --- Out-of-distribution gate ------------------------------------
        if core < cfg.min_core_inputs {
            return ood(observed);
        }
        if self.out_of_bounds(f).is_some() {
            return ood(observed);
        }

        let get = |k: &str| f.get(k).copied().filter(|v| v.is_finite());

        // Evidence accumulator over candidate labels.
        let mut score: BTreeMap<RegimeLabel, f64> = BTreeMap::new();
        let mut add = |label: RegimeLabel, w: f64| {
            *score.entry(label).or_insert(0.0) += w;
        };

        // --- Volatility regime -------------------------------------------
        let rv_pctl = get("regime.rv_percentile");
        let vv3m = get("regime.vix_vix3m");
        let vvix = get("regime.vvix");

        let backwardation = vv3m.map(|x| x >= cfg.backwardation_hi).unwrap_or(false);
        let contango = vv3m.map(|x| x <= cfg.contango_lo).unwrap_or(false);
        let rv_high = rv_pctl.map(|x| x >= cfg.rv_pctl_high).unwrap_or(false);
        let rv_low = rv_pctl.map(|x| x <= cfg.rv_pctl_low).unwrap_or(false);
        let vvix_high = vvix.map(|x| x >= cfg.vvix_high).unwrap_or(false);
        let vvix_low = vvix.map(|x| x <= cfg.vvix_low).unwrap_or(false);

        // VolExpansion: high rv percentile AND/OR backwardation AND/OR high VVIX.
        if rv_high {
            add(RegimeLabel::VolExpansion, 1.0);
        }
        if backwardation {
            add(RegimeLabel::VolExpansion, 1.0);
        }
        if vvix_high {
            add(RegimeLabel::VolExpansion, 0.5);
        }
        // VolCompression: low rv AND contango (and calm vol-of-vol helps).
        if rv_low {
            add(RegimeLabel::VolCompression, 1.0);
        }
        if contango {
            add(RegimeLabel::VolCompression, 1.0);
        }
        if vvix_low {
            add(RegimeLabel::VolCompression, 0.5);
        }

        // --- Risk on/off -------------------------------------------------
        let hy = get("regime.hy_oas");
        let credit_wide = hy.map(|x| x >= cfg.hy_oas_wide).unwrap_or(false);
        let credit_tight = hy.map(|x| x <= cfg.hy_oas_tight).unwrap_or(false);

        let copper_gold = get("regime.copper_gold"); // >0 risk-on, <0 risk-off
        let dxy_trend = get("regime.dxy_trend"); // up = risk-off (flight to USD)

        let cg_up = copper_gold.map(|x| x > cfg.trend_flat).unwrap_or(false);
        let cg_down = copper_gold.map(|x| x < -cfg.trend_flat).unwrap_or(false);
        let dxy_up = dxy_trend.map(|x| x > cfg.trend_flat).unwrap_or(false);
        let dxy_down = dxy_trend.map(|x| x < -cfg.trend_flat).unwrap_or(false);

        // RiskOff: credit widening, backwardation, copper/gold falling, DXY up.
        if credit_wide {
            add(RegimeLabel::RiskOff, 1.5);
        }
        if backwardation {
            add(RegimeLabel::RiskOff, 0.75);
        }
        if cg_down {
            add(RegimeLabel::RiskOff, 0.75);
        }
        if dxy_up {
            add(RegimeLabel::RiskOff, 0.5);
        }
        if rv_high {
            add(RegimeLabel::RiskOff, 0.5);
        }
        // RiskOn: the opposite.
        if credit_tight {
            add(RegimeLabel::RiskOn, 1.5);
        }
        if contango {
            add(RegimeLabel::RiskOn, 0.75);
        }
        if cg_up {
            add(RegimeLabel::RiskOn, 0.75);
        }
        if dxy_down {
            add(RegimeLabel::RiskOn, 0.5);
        }
        if rv_low {
            add(RegimeLabel::RiskOn, 0.5);
        }

        // --- Gamma (PROXY unless GEX wired) ------------------------------
        if let Some(sign) = get("regime.gex_sign") {
            // Direct proprietary signal dominates when present.
            if sign >= 0.0 {
                add(RegimeLabel::LongGamma, 2.0);
            } else {
                add(RegimeLabel::ShortGamma, 2.0);
            }
        } else {
            // Proxy from vol regime + VVIX (documented approximation).
            if (rv_high || backwardation) && vvix_high {
                add(RegimeLabel::ShortGamma, 0.6);
            }
            if rv_low && contango && vvix_low {
                add(RegimeLabel::LongGamma, 0.6);
            }
        }

        // --- Softmax over candidates -------------------------------------
        // If nothing scored, the signals conflict/are neutral -> Transition.
        if score.values().all(|&v| v <= 0.0) {
            let mut prob_map = BTreeMap::new();
            prob_map.insert(RegimeLabel::Transition, 1.0);
            return RegimeAssessment {
                label: RegimeLabel::Transition,
                prob_map,
                confidence: 1.0,
                observed_inputs: observed,
            };
        }

        let prob_map = softmax(&score, cfg.softmax_temp);

        // Rank labels by probability (descending). Ties broken by the enum's Ord
        // for determinism.
        let mut ranked: Vec<(RegimeLabel, f64)> = prob_map.iter().map(|(k, v)| (*k, *v)).collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        let (mut label, mut confidence) = ranked
            .first()
            .copied()
            .unwrap_or((RegimeLabel::Transition, 1.0));

        // Conflict detection: if the top-2 labels are near-tied (within 8pp) and
        // point in opposite directions (e.g. VolExpansion vs VolCompression),
        // call it a Transition instead of forcing a coin-flip.
        if ranked.len() >= 2 {
            let (l1, p1) = ranked[0];
            let (l2, p2) = ranked[1];
            if (p1 - p2).abs() <= 0.08 && opposed(l1, l2) {
                label = RegimeLabel::Transition;
                confidence = prob_map.get(&label).copied().unwrap_or(p1);
            }
        }

        RegimeAssessment {
            label,
            prob_map,
            confidence,
            observed_inputs: observed,
        }
    }
}

fn ood(observed: usize) -> RegimeAssessment {
    let mut prob_map = BTreeMap::new();
    prob_map.insert(RegimeLabel::OutOfDistribution, 1.0);
    RegimeAssessment {
        label: RegimeLabel::OutOfDistribution,
        prob_map,
        confidence: 1.0,
        observed_inputs: observed,
    }
}

/// Softmax of an evidence-score map at temperature `temp`.
fn softmax(score: &BTreeMap<RegimeLabel, f64>, temp: f64) -> BTreeMap<RegimeLabel, f64> {
    let t = if temp <= 0.0 { 1.0 } else { temp };
    let max = score.values().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<(RegimeLabel, f64)> = score
        .iter()
        .map(|(k, v)| (*k, ((v - max) / t).exp()))
        .collect();
    let sum: f64 = exps.iter().map(|(_, e)| e).sum();
    exps.into_iter()
        .map(|(k, e)| (k, if sum > 0.0 { e / sum } else { 0.0 }))
        .collect()
}

/// Whether two labels are semantically opposite directions.
fn opposed(a: RegimeLabel, b: RegimeLabel) -> bool {
    use RegimeLabel::*;
    matches!(
        (a, b),
        (VolExpansion, VolCompression)
            | (VolCompression, VolExpansion)
            | (RiskOff, RiskOn)
            | (RiskOn, RiskOff)
            | (ShortGamma, LongGamma)
            | (LongGamma, ShortGamma)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn calm_contango_is_risk_on_or_vol_compression() {
        let f = map(&[
            ("regime.rv_percentile", 0.15),
            ("regime.vix_vix3m", 0.88),
            ("regime.vvix", 85.0),
            ("regime.rv20", 0.10),
            ("regime.hy_oas", 3.0),
            ("regime.copper_gold", 0.02),
            ("regime.dxy_trend", -0.01),
        ]);
        let a = RegimeClassifier::default().classify(&f);
        assert!(
            matches!(a.label, RegimeLabel::RiskOn | RegimeLabel::VolCompression),
            "calm/contango should be RiskOn or VolCompression, got {:?}",
            a.label
        );
        assert!(a.label.is_tradeable());
    }

    #[test]
    fn backwardation_credit_wide_is_risk_off_or_vol_expansion() {
        let f = map(&[
            ("regime.rv_percentile", 0.95),
            ("regime.vix_vix3m", 1.20),
            ("regime.vvix", 130.0),
            ("regime.rv20", 0.55),
            ("regime.hy_oas", 7.5),
            ("regime.copper_gold", -0.06),
            ("regime.dxy_trend", 0.03),
        ]);
        let a = RegimeClassifier::default().classify(&f);
        assert!(
            matches!(a.label, RegimeLabel::RiskOff | RegimeLabel::VolExpansion),
            "backwardation + wide credit should be RiskOff or VolExpansion, got {:?}",
            a.label
        );
    }

    #[test]
    fn sparse_inputs_are_out_of_distribution() {
        let f = map(&[("regime.rv20", 0.12), ("regime.vvix", 95.0)]);
        let a = RegimeClassifier::default().classify(&f);
        assert_eq!(a.label, RegimeLabel::OutOfDistribution);
        assert!(!a.label.is_tradeable());
    }

    #[test]
    fn empty_is_out_of_distribution() {
        let a = RegimeClassifier::default().classify(&BTreeMap::new());
        assert_eq!(a.label, RegimeLabel::OutOfDistribution);
        assert_eq!(a.observed_inputs, 0);
    }

    #[test]
    fn wildly_out_of_bounds_is_ood() {
        let f = map(&[
            ("regime.rv_percentile", 0.5),
            ("regime.vix_vix3m", 9.9), // absurd term-structure ratio
            ("regime.vvix", 95.0),
            ("regime.rv20", 0.2),
        ]);
        let a = RegimeClassifier::default().classify(&f);
        assert_eq!(a.label, RegimeLabel::OutOfDistribution);
    }

    #[test]
    fn prob_map_normalizes_and_confidence_is_top() {
        let f = map(&[
            ("regime.rv_percentile", 0.95),
            ("regime.vix_vix3m", 1.2),
            ("regime.vvix", 130.0),
            ("regime.rv20", 0.5),
            ("regime.hy_oas", 7.0),
            ("regime.copper_gold", -0.05),
            ("regime.dxy_trend", 0.02),
        ]);
        let a = RegimeClassifier::default().classify(&f);
        let sum: f64 = a.prob_map.values().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "prob_map must sum to 1, got {sum}"
        );
        let top = a.prob_map.values().cloned().fold(0.0, f64::max);
        assert!((a.confidence - top).abs() < 1e-9);
        // JSON round-trips the regime keys.
        let j = a.prob_map_json();
        assert!(j.is_object());
    }

    #[test]
    fn gex_sign_drives_gamma_when_present() {
        let mut f = map(&[
            ("regime.rv_percentile", 0.5),
            ("regime.vix_vix3m", 0.97),
            ("regime.vvix", 100.0),
            ("regime.rv20", 0.2),
            ("regime.gex_sign", -1.0),
        ]);
        let a = RegimeClassifier::default().classify(&f);
        // Short gamma should be in the running with a clear weight.
        assert!(a.prob_map.contains_key(&RegimeLabel::ShortGamma));
        f.insert("regime.gex_sign".into(), 1.0);
        let b = RegimeClassifier::default().classify(&f);
        assert!(b.prob_map.contains_key(&RegimeLabel::LongGamma));
    }
}
