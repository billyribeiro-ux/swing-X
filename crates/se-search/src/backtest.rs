//! Backtesting a genome over a materialized feature window.
//!
//! Walk the decision bars in order. At each bar, if the genome fires AND the bar's regime is
//! tradeable AND we are not inside a cooldown/min-hold window, open an entry, label it with the
//! triple-barrier labeler (ATR-sized), and record a [`LabeledEntry`] carrying the firing
//! feature map and the regime tag. The collected entries assemble into the `DatasetRow`s the
//! OOS scorer consumes.
//!
//! This is an IN-SAMPLE generator of labeled events ONLY — it produces the dataset. It makes
//! NO promotion decision; ranking is done exclusively on the OOS score (see [`crate::score`]).

use std::collections::BTreeMap;

use se_core::{Genome, HorizonProfile};
use se_labeler::{LabeledEntry, TripleBarrier};
use se_mlclient::DatasetRow;
use se_regime::RegimeClassifier;

use crate::feature_matrix::{dotted_to_column, FeatureWindow};

/// Outcome of backtesting one genome over one window.
#[derive(Debug, Clone, Default)]
pub struct BacktestResult {
    /// Labeled entries in chronological order (entry-time ascending).
    pub entries: Vec<LabeledEntry>,
    /// Bars where the genome fired but the regime was not tradeable (skipped).
    pub skipped_regime: usize,
    /// Bars where the genome fired but we lacked ATR / forward bars to label (skipped).
    pub skipped_unlabelable: usize,
}

/// Walk `window` and produce labeled entries for `genome` under `profile`.
///
/// Entry convention: the genome fires at the decision bar; the entry executes at that bar's
/// CLOSE (the labeler's first touchable bar is the NEXT bar, so there is no look-ahead). After
/// an entry we enforce `min_hold_bars` as a cooldown before the next entry, so overlapping
/// same-direction entries don't double-count one move.
pub fn backtest(
    genome: &Genome,
    window: &FeatureWindow,
    profile: &HorizonProfile,
) -> BacktestResult {
    let mut result = BacktestResult::default();
    if window.points.is_empty() || window.bars.is_empty() {
        return result;
    }

    let labeler = TripleBarrier::new(*profile);
    let classifier = RegimeClassifier::default();
    let cooldown = profile.min_hold_bars.max(1) as usize;
    let side = genome.side;

    // Bar index of the last entry, to enforce the cooldown.
    let mut last_entry_idx: Option<usize> = None;

    for point in &window.points {
        // Cooldown: skip if too close to the previous entry.
        if let Some(prev) = last_entry_idx {
            if point.idx <= prev + cooldown {
                continue;
            }
        }

        if !genome.fires(&point.features) {
            continue;
        }

        // Regime gate: classify from the regime features at this bar; suppress if not tradeable.
        let regime = classify_regime(&classifier, &point.features);
        let regime_label = regime.as_ref();
        if regime
            .as_ref()
            .map(|r| !r.is_tradeable_str())
            .unwrap_or(false)
        {
            result.skipped_regime += 1;
            continue;
        }

        // Need a positive ATR to size barriers and at least one forward bar to resolve.
        let Some(atr) = point.atr else {
            result.skipped_unlabelable += 1;
            continue;
        };
        if point.idx + 1 >= window.bars.len() {
            result.skipped_unlabelable += 1;
            continue;
        }

        match labeler.label_one(&window.bars, point.idx, side, atr) {
            Ok(event) => {
                let features = to_column_features(&point.features);
                result.entries.push(LabeledEntry {
                    event,
                    features,
                    regime: regime_label.map(|s| s.label.clone()),
                });
                last_entry_idx = Some(point.idx);
            }
            Err(_) => {
                result.skipped_unlabelable += 1;
            }
        }
    }

    result
}

/// A classified regime at one bar: the string label (for the dataset's `regime` column) plus
/// tradeability.
struct BarRegime {
    label: String,
    tradeable: bool,
}

impl BarRegime {
    fn is_tradeable_str(&self) -> bool {
        self.tradeable
    }
}

/// Classify the bar's regime from the regime-layer features. Returns `None` if there are too
/// few regime signals to classify at all (the classifier returns `OutOfDistribution`, which is
/// not tradeable — handled by the caller).
fn classify_regime(
    classifier: &RegimeClassifier,
    features: &BTreeMap<String, f64>,
) -> Option<BarRegime> {
    // The classifier reads `regime.*` keys directly from the full feature map.
    let assessment = classifier.classify(features);
    Some(BarRegime {
        label: assessment.label.as_str().to_string(),
        tradeable: assessment.label.is_tradeable(),
    })
}

/// Convert a dotted feature map to the `layer__feature` column names the Parquet writer needs.
fn to_column_features(features: &BTreeMap<String, f64>) -> BTreeMap<String, f64> {
    features
        .iter()
        .map(|(k, v)| (dotted_to_column(k), *v))
        .collect()
}

/// Assemble a genome's labeled entries into the OOS dataset rows, sorted ascending by entry
/// time (a precondition of the Parquet writer + CPCV purging).
pub fn assemble(result: &BacktestResult) -> Vec<DatasetRow> {
    let mut rows = se_labeler::assemble_dataset(&result.entries);
    rows.sort_by(|a, b| a.ts.cmp(&b.ts));
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_matrix::BarPoint;
    use chrono::{TimeZone, Utc};
    use se_core::{Bar, CmpOp, Horizon, Predicate, Side, Ticker};

    fn bar(i: i64, c: f64) -> Bar {
        Bar {
            ticker: Ticker::Spy,
            ts: Utc.timestamp_opt(1_600_000_000 + i * 86_400, 0).unwrap(),
            open: c,
            high: c + 2.0,
            low: c - 2.0,
            close: c,
            volume: 1_000.0,
        }
    }

    /// A window with a steady uptrend and regime features that classify as tradeable.
    fn synthetic_window(n: usize) -> FeatureWindow {
        let bars: Vec<Bar> = (0..n).map(|i| bar(i as i64, 100.0 + i as f64)).collect();
        let points: Vec<BarPoint> = (0..n)
            .map(|i| {
                let mut f = BTreeMap::new();
                // A trigger feature that the genome keys on.
                f.insert("trigger.momentum_roc".to_string(), (i % 5) as f64 - 2.0);
                // Calm/contango regime signals -> tradeable (RiskOn / VolCompression).
                f.insert("regime.rv_percentile".to_string(), 0.15);
                f.insert("regime.vix_vix3m".to_string(), 0.88);
                f.insert("regime.vvix".to_string(), 85.0);
                f.insert("regime.rv20".to_string(), 0.10);
                f.insert("regime.copper_gold".to_string(), 0.02);
                f.insert("regime.dxy_trend".to_string(), -0.01);
                BarPoint {
                    bar: bars[i],
                    idx: i,
                    features: f,
                    atr: Some(1.0),
                }
            })
            .collect();
        FeatureWindow {
            ticker: Ticker::Spy,
            bars,
            points,
        }
    }

    #[test]
    fn fires_and_labels_with_cooldown() {
        let w = synthetic_window(40);
        let genome = Genome::new(
            Side::Long,
            Horizon::Swing,
            vec![Predicate {
                layer: se_core::Layer::Trigger,
                feature_key: "trigger.momentum_roc".into(),
                op: CmpOp::Gt,
                threshold: 0.0, // fires when roc in {1,2}
            }],
        );
        let profile = HorizonProfile::swing();
        let res = backtest(&genome, &w, &profile);
        // Should produce several entries, all chronological, with cooldown respected.
        assert!(!res.entries.is_empty(), "expected entries");
        let rows = assemble(&res);
        for w2 in rows.windows(2) {
            assert!(w2[0].ts <= w2[1].ts, "rows must be ascending by ts");
        }
        // Feature columns are in __ form.
        assert!(rows[0].features.keys().any(|k| k.contains("__")));
        // Cooldown: entries are at least min_hold+1 bars apart.
        let min_gap = profile.min_hold_bars as i64 + 1;
        let idxs: Vec<i64> = res
            .entries
            .iter()
            .map(|e| e.event.entry_ts.timestamp())
            .collect();
        for w2 in idxs.windows(2) {
            let gap_bars = (w2[1] - w2[0]) / 86_400;
            assert!(
                gap_bars >= min_gap,
                "cooldown violated: {gap_bars} < {min_gap}"
            );
        }
    }

    #[test]
    fn non_firing_genome_yields_nothing() {
        let w = synthetic_window(40);
        let genome = Genome::new(
            Side::Long,
            Horizon::Swing,
            vec![Predicate {
                layer: se_core::Layer::Trigger,
                feature_key: "trigger.momentum_roc".into(),
                op: CmpOp::Gt,
                threshold: 1000.0, // never fires
            }],
        );
        let res = backtest(&genome, &w, &HorizonProfile::swing());
        assert!(res.entries.is_empty());
    }
}
