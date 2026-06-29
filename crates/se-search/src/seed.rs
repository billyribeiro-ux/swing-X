//! Seeding the initial population from the feature keys actually observed in the data.
//!
//! We never invent feature names or thresholds: candidate predicates are drawn from the keys
//! present across the materialized [`FeatureWindow`]s, and thresholds are sampled from the
//! OBSERVED empirical quantiles of each feature. That keeps genomes inside the support of the
//! data (a predicate that can never fire is useless) and makes the search honest. All draws
//! come from a [`crate::rng::Rng`] seeded per generation, so the seed set is reproducible.

use std::collections::BTreeMap;

use se_core::{CmpOp, Genome, Horizon, Layer, Predicate, Side};

use crate::feature_matrix::FeatureWindow;
use crate::rng::Rng;

/// A candidate feature available to the search: its dotted key, its layer, and the sorted
/// observed values (for quantile-based threshold draws).
#[derive(Debug, Clone)]
pub struct FeatureStat {
    pub key: String,
    pub layer: Layer,
    /// Observed values, ascending. Used to draw thresholds at empirical quantiles.
    pub sorted_values: Vec<f64>,
}

impl FeatureStat {
    /// The value at empirical quantile `q` in `[0, 1]` (nearest-rank). `None` if no samples.
    pub fn quantile(&self, q: f64) -> Option<f64> {
        if self.sorted_values.is_empty() {
            return None;
        }
        let q = q.clamp(0.0, 1.0);
        let n = self.sorted_values.len();
        let idx = ((q * (n as f64 - 1.0)).round() as usize).min(n - 1);
        Some(self.sorted_values[idx])
    }

    /// True if the feature varies enough to be worth a predicate (not a constant).
    pub fn is_informative(&self) -> bool {
        match (self.sorted_values.first(), self.sorted_values.last()) {
            (Some(lo), Some(hi)) => (hi - lo).abs() > f64::EPSILON,
            _ => false,
        }
    }
}

/// The catalog of features the search may use, derived from one or more windows.
#[derive(Debug, Clone, Default)]
pub struct FeatureCatalog {
    pub stats: Vec<FeatureStat>,
}

impl FeatureCatalog {
    /// Build the catalog from the per-bar feature maps of the given windows. A feature is
    /// included only if it appears with a minimum number of finite, varying observations.
    pub fn from_windows(windows: &[FeatureWindow], min_observations: usize) -> Self {
        let mut buckets: BTreeMap<String, (Layer, Vec<f64>)> = BTreeMap::new();
        for w in windows {
            for p in &w.points {
                for (k, v) in &p.features {
                    if !v.is_finite() {
                        continue;
                    }
                    let layer = layer_of_key(k);
                    buckets
                        .entry(k.clone())
                        .or_insert_with(|| (layer, Vec::new()))
                        .1
                        .push(*v);
                }
            }
        }

        let mut stats = Vec::new();
        for (key, (layer, mut vals)) in buckets {
            if vals.len() < min_observations {
                continue;
            }
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let stat = FeatureStat {
                key,
                layer,
                sorted_values: vals,
            };
            if stat.is_informative() {
                stats.push(stat);
            }
        }
        FeatureCatalog { stats }
    }

    pub fn is_empty(&self) -> bool {
        self.stats.is_empty()
    }

    pub fn len(&self) -> usize {
        self.stats.len()
    }
}

/// Infer a [`Layer`] from a dotted feature key prefix (`regime.adx` -> Regime). Falls back to
/// [`Layer::Trigger`] (a benign default) for unknown prefixes — only used for the genome's
/// attribution tag, never for leakage decisions.
pub fn layer_of_key(key: &str) -> Layer {
    match key.split_once('.').map(|(p, _)| p) {
        Some("tradeability") => Layer::Tradeability,
        Some("regime") => Layer::Regime,
        Some("location") => Layer::Location,
        Some("trigger") => Layer::Trigger,
        Some("event") => Layer::Event,
        _ => Layer::Trigger,
    }
}

/// Draw one predicate over a random catalog feature, with a threshold at a random observed
/// quantile and a random comparison direction.
fn draw_predicate(catalog: &FeatureCatalog, rng: &mut Rng) -> Option<Predicate> {
    let stat = rng.choice(&catalog.stats)?;
    // Avoid the extreme tails so the predicate has a chance to both fire and not-fire.
    let q = 0.2 + 0.6 * rng.uniform();
    let threshold = stat.quantile(q)?;
    let op = if rng.chance(0.5) {
        CmpOp::Gt
    } else {
        CmpOp::Lt
    };
    Some(Predicate {
        layer: stat.layer,
        feature_key: stat.key.clone(),
        op,
        threshold,
    })
}

/// Public entry to [`draw_predicate`] for the mutation operators, which also draw fresh
/// predicates over the same observed catalog.
pub fn draw_predicate_public(catalog: &FeatureCatalog, rng: &mut Rng) -> Option<Predicate> {
    draw_predicate(catalog, rng)
}

/// Build a single random genome: 1–3 predicates over distinct features, a random side, the
/// configured horizon.
pub fn random_genome(
    catalog: &FeatureCatalog,
    horizon: Horizon,
    rng: &mut Rng,
    max_predicates: usize,
) -> Option<Genome> {
    if catalog.is_empty() {
        return None;
    }
    let n_pred = 1 + rng.below(max_predicates.max(1));
    let mut predicates: Vec<Predicate> = Vec::new();
    let mut used: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // Bounded attempts so we don't loop forever on a tiny catalog.
    for _ in 0..(n_pred * 4) {
        if predicates.len() >= n_pred {
            break;
        }
        if let Some(p) = draw_predicate(catalog, rng) {
            if used.insert(p.feature_key.clone()) {
                predicates.push(p);
            }
        }
    }
    if predicates.is_empty() {
        return None;
    }
    let side = if rng.chance(0.5) {
        Side::Long
    } else {
        Side::Short
    };
    Some(Genome::new(side, horizon, predicates))
}

/// Seed an initial population of `size` distinct genomes for `horizon`, drawn deterministically
/// from `rng`. Both sides are represented (subject to the random draws). Duplicate genomes are
/// de-duplicated by their `describe()` string.
pub fn seed_population(
    catalog: &FeatureCatalog,
    horizon: Horizon,
    size: usize,
    rng: &mut Rng,
    max_predicates: usize,
) -> Vec<Genome> {
    let mut out: Vec<Genome> = Vec::new();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // Try generously; a small catalog may not yield `size` unique genomes.
    for _ in 0..(size * 8) {
        if out.len() >= size {
            break;
        }
        if let Some(g) = random_genome(catalog, horizon, rng, max_predicates) {
            if seen.insert(g.describe()) {
                out.push(g);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_matrix::BarPoint;
    use se_core::{Bar, Ticker};

    fn window_with(features: &[(&str, &[f64])]) -> FeatureWindow {
        // Build N synthetic bar points; feature i gets the i-th value of each series.
        let n = features.iter().map(|(_, v)| v.len()).max().unwrap_or(0);
        let mut points = Vec::new();
        for i in 0..n {
            let mut map = BTreeMap::new();
            for (k, vals) in features {
                if let Some(v) = vals.get(i) {
                    map.insert((*k).to_string(), *v);
                }
            }
            points.push(BarPoint {
                bar: Bar {
                    ticker: Ticker::SPY,
                    ts: chrono::Utc::now(),
                    open: 1.0,
                    high: 1.0,
                    low: 1.0,
                    close: 1.0,
                    volume: 1.0,
                },
                idx: i,
                features: map,
                atr: Some(1.0),
            });
        }
        FeatureWindow {
            ticker: Ticker::SPY,
            bars: Vec::new(),
            points,
        }
    }

    #[test]
    fn catalog_filters_constants_and_sparse() {
        let w = window_with(&[
            ("regime.adx", &[10.0, 20.0, 30.0, 40.0, 50.0]),
            ("regime.flat", &[5.0, 5.0, 5.0, 5.0, 5.0]), // constant -> dropped
            ("trigger.rsi14", &[1.0]),                   // too sparse -> dropped
        ]);
        let cat = FeatureCatalog::from_windows(&[w], 3);
        let keys: Vec<&str> = cat.stats.iter().map(|s| s.key.as_str()).collect();
        assert!(keys.contains(&"regime.adx"));
        assert!(!keys.contains(&"regime.flat"));
        assert!(!keys.contains(&"trigger.rsi14"));
    }

    #[test]
    fn quantiles_are_within_observed_range() {
        let stat = FeatureStat {
            key: "x".into(),
            layer: Layer::Regime,
            sorted_values: vec![0.0, 1.0, 2.0, 3.0, 4.0],
        };
        assert_eq!(stat.quantile(0.0), Some(0.0));
        assert_eq!(stat.quantile(1.0), Some(4.0));
        let mid = stat.quantile(0.5).unwrap();
        assert!((0.0..=4.0).contains(&mid));
    }

    #[test]
    fn seeded_population_is_deterministic_and_bounded() {
        let w = window_with(&[
            ("regime.adx", &(0..50).map(|i| i as f64).collect::<Vec<_>>()),
            (
                "trigger.rsi14",
                &(0..50).map(|i| (i as f64) * 2.0).collect::<Vec<_>>(),
            ),
            (
                "location.dist_50dma",
                &(0..50).map(|i| (i as f64) * 0.1 - 2.0).collect::<Vec<_>>(),
            ),
        ]);
        let cat = FeatureCatalog::from_windows(&[w], 3);
        assert!(!cat.is_empty());

        let pop_a = {
            let mut r = Rng::seeded(123, 0);
            seed_population(&cat, Horizon::Swing, 12, &mut r, 3)
        };
        let pop_b = {
            let mut r = Rng::seeded(123, 0);
            seed_population(&cat, Horizon::Swing, 12, &mut r, 3)
        };
        assert_eq!(pop_a.len(), pop_b.len());
        for (a, b) in pop_a.iter().zip(pop_b.iter()) {
            assert_eq!(a.describe(), b.describe());
        }
        // Every genome has 1..=3 predicates and a valid horizon.
        for g in &pop_a {
            assert!(!g.predicates.is_empty() && g.predicates.len() <= 3);
            assert_eq!(g.horizon, Horizon::Swing);
        }
    }
}
