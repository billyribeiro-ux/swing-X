//! Mutation and crossover over [`Genome`]s.
//!
//! Every operator stays inside the OBSERVED feature catalog and the observed value range, so a
//! mutated genome remains a plausible, data-supported hypothesis (a threshold outside the data
//! could never fire). Operators draw from a seeded [`Rng`] for reproducibility.

use std::collections::BTreeSet;

use se_core::{CmpOp, Genome, Predicate};

use crate::rng::Rng;
use crate::seed::{draw_predicate_public, FeatureCatalog, FeatureStat};

/// Maximum predicates a genome may carry after an `add` mutation.
const MAX_PREDICATES: usize = 4;

/// Find the catalog stat for a feature key, if present.
fn stat_for<'a>(catalog: &'a FeatureCatalog, key: &str) -> Option<&'a FeatureStat> {
    catalog.stats.iter().find(|s| s.key == key)
}

/// Perturb a predicate's threshold to a nearby observed quantile. Keeping it on the empirical
/// grid guarantees the new threshold is reachable by the data.
fn perturb_threshold(pred: &mut Predicate, stat: &FeatureStat, rng: &mut Rng) {
    // Nudge by up to +/- 15 percentiles around the current rank.
    let n = stat.sorted_values.len();
    if n < 2 {
        return;
    }
    // Locate the current threshold's approximate rank.
    let cur_rank =
        stat.sorted_values.partition_point(|&v| v < pred.threshold) as f64 / (n as f64 - 1.0);
    let delta = (rng.uniform() - 0.5) * 0.30; // +/- 0.15
    let q = (cur_rank + delta).clamp(0.0, 1.0);
    if let Some(t) = stat.quantile(q) {
        pred.threshold = t;
    }
}

/// Mutate `genome` in place-ish (returns a new genome). One of: perturb a threshold, flip an
/// operator, swap a predicate's feature, add a predicate, or remove a predicate. The choice is
/// random but the result is always a valid, non-empty genome over catalog features.
pub fn mutate(genome: &Genome, catalog: &FeatureCatalog, rng: &mut Rng) -> Genome {
    if catalog.is_empty() {
        return genome.clone();
    }
    let mut g = genome.clone();
    let roll = rng.below(5);

    match roll {
        // 0: perturb a threshold.
        0 => {
            if !g.predicates.is_empty() {
                let i = rng.below(g.predicates.len());
                let key = g.predicates[i].feature_key.clone();
                if let Some(stat) = stat_for(catalog, &key) {
                    perturb_threshold(&mut g.predicates[i], stat, rng);
                }
            }
        }
        // 1: flip a comparison operator (Gt<->Lt, Gte<->Lte).
        1 => {
            if !g.predicates.is_empty() {
                let i = rng.below(g.predicates.len());
                g.predicates[i].op = match g.predicates[i].op {
                    CmpOp::Gt => CmpOp::Lt,
                    CmpOp::Lt => CmpOp::Gt,
                    CmpOp::Gte => CmpOp::Lte,
                    CmpOp::Lte => CmpOp::Gte,
                };
            }
        }
        // 2: swap a predicate for a fresh one over a different feature.
        2 => {
            if !g.predicates.is_empty() {
                let i = rng.below(g.predicates.len());
                if let Some(p) = draw_predicate_public(catalog, rng) {
                    // Only swap if it doesn't duplicate another predicate's feature.
                    let used: BTreeSet<&str> = g
                        .predicates
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, pr)| pr.feature_key.as_str())
                        .collect();
                    if !used.contains(p.feature_key.as_str()) {
                        g.predicates[i] = p;
                    }
                }
            }
        }
        // 3: add a predicate over a new feature (if room).
        3 => {
            if g.predicates.len() < MAX_PREDICATES {
                if let Some(p) = draw_predicate_public(catalog, rng) {
                    let used: BTreeSet<&str> = g
                        .predicates
                        .iter()
                        .map(|pr| pr.feature_key.as_str())
                        .collect();
                    if !used.contains(p.feature_key.as_str()) {
                        g.predicates.push(p);
                    }
                }
            }
        }
        // 4: remove a predicate (never empty the genome).
        _ => {
            if g.predicates.len() > 1 {
                let i = rng.below(g.predicates.len());
                g.predicates.remove(i);
            }
        }
    }

    // Guard: a mutation must never leave a genome predicate-less.
    if g.predicates.is_empty() {
        return genome.clone();
    }
    g
}

/// One-point crossover: take a prefix of `a`'s predicates and a suffix of `b`'s, de-duplicating
/// by feature key, capped at [`MAX_PREDICATES`]. Side/horizon inherit from `a`. If the result
/// would be empty, fall back to `a`.
pub fn crossover(a: &Genome, b: &Genome, rng: &mut Rng) -> Genome {
    let mut predicates: Vec<Predicate> = Vec::new();
    let mut used: BTreeSet<String> = BTreeSet::new();

    let cut_a = if a.predicates.is_empty() {
        0
    } else {
        1 + rng.below(a.predicates.len())
    };
    for p in a.predicates.iter().take(cut_a) {
        if used.insert(p.feature_key.clone()) {
            predicates.push(p.clone());
        }
    }
    for p in &b.predicates {
        if predicates.len() >= MAX_PREDICATES {
            break;
        }
        if used.insert(p.feature_key.clone()) {
            predicates.push(p.clone());
        }
    }

    if predicates.is_empty() {
        return a.clone();
    }
    Genome::new(a.side, a.horizon, predicates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_matrix::{BarPoint, FeatureWindow};
    use crate::seed::random_genome;
    use se_core::{Bar, Horizon, Ticker};
    use std::collections::BTreeMap;

    fn catalog() -> FeatureCatalog {
        let mut points = Vec::new();
        for i in 0..60 {
            let mut m = BTreeMap::new();
            m.insert("regime.adx".to_string(), i as f64);
            m.insert("trigger.rsi14".to_string(), (i as f64) * 1.5);
            m.insert("location.dist_50dma".to_string(), (i as f64) * 0.1 - 3.0);
            m.insert(
                "trigger.obv_trend".to_string(),
                if i % 2 == 0 { 1.0 } else { -1.0 },
            );
            points.push(BarPoint {
                bar: Bar {
                    ticker: Ticker::Spy,
                    ts: chrono::Utc::now(),
                    open: 1.0,
                    high: 1.0,
                    low: 1.0,
                    close: 1.0,
                    volume: 1.0,
                },
                idx: i,
                features: m,
                atr: Some(1.0),
            });
        }
        let w = FeatureWindow {
            ticker: Ticker::Spy,
            bars: Vec::new(),
            points,
        };
        FeatureCatalog::from_windows(&[w], 3)
    }

    #[test]
    fn mutate_keeps_genome_valid_and_on_catalog() {
        let cat = catalog();
        let mut rng = Rng::seeded(5, 0);
        let g = random_genome(&cat, Horizon::Swing, &mut rng, 3).unwrap();
        for _ in 0..200 {
            let m = mutate(&g, &cat, &mut rng);
            assert!(!m.predicates.is_empty());
            assert!(m.predicates.len() <= MAX_PREDICATES);
            // Every predicate key is a catalog feature.
            for p in &m.predicates {
                assert!(
                    cat.stats.iter().any(|s| s.key == p.feature_key),
                    "mutated predicate {} not in catalog",
                    p.feature_key
                );
            }
            // No duplicate feature keys within a genome.
            let unique: BTreeSet<&str> = m
                .predicates
                .iter()
                .map(|p| p.feature_key.as_str())
                .collect();
            assert_eq!(unique.len(), m.predicates.len());
        }
    }

    #[test]
    fn crossover_dedups_and_bounds() {
        let cat = catalog();
        let mut rng = Rng::seeded(9, 0);
        let a = random_genome(&cat, Horizon::Swing, &mut rng, 4).unwrap();
        let b = random_genome(&cat, Horizon::Swing, &mut rng, 4).unwrap();
        let c = crossover(&a, &b, &mut rng);
        assert!(!c.predicates.is_empty());
        assert!(c.predicates.len() <= MAX_PREDICATES);
        let unique: BTreeSet<&str> = c
            .predicates
            .iter()
            .map(|p| p.feature_key.as_str())
            .collect();
        assert_eq!(unique.len(), c.predicates.len());
        assert_eq!(c.side, a.side);
    }
}
