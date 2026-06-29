//! Mutation and crossover over [`Genome`]s.
//!
//! Every operator stays inside the OBSERVED feature catalog and the observed value range, so a
//! mutated genome remains a plausible, data-supported hypothesis (a threshold outside the data
//! could never fire). Operators draw from a seeded [`Rng`] for reproducibility.

use std::collections::BTreeSet;

use se_core::{CmpOp, Genome, Layer, Predicate};

use crate::rng::Rng;
use crate::seed::{draw_predicate_in_layers, draw_predicate_public, FeatureCatalog, FeatureStat};

/// Maximum predicates a genome may carry after an `add` mutation.
const MAX_PREDICATES: usize = 4;

/// All catalog layers, in a fixed order, used to find which layer a genome is missing.
const ALL_LAYERS: [Layer; 5] = [
    Layer::Tradeability,
    Layer::Regime,
    Layer::Location,
    Layer::Trigger,
    Layer::Event,
];

/// Find the catalog stat for a feature key, if present.
fn stat_for<'a>(catalog: &'a FeatureCatalog, key: &str) -> Option<&'a FeatureStat> {
    catalog.stats.iter().find(|s| s.key == key)
}

/// Perturb a predicate's threshold to a nearby observed quantile. Keeping it on the empirical
/// grid guarantees the new threshold is reachable by the data. `span` is the full percentile width
/// of the jitter window (so the nudge is +/- span/2 around the current rank): a small span
/// fine-tunes, a large span takes a bolder step to escape a local optimum.
fn perturb_threshold_span(pred: &mut Predicate, stat: &FeatureStat, span: f64, rng: &mut Rng) {
    let n = stat.sorted_values.len();
    if n < 2 {
        return;
    }
    // Locate the current threshold's approximate rank.
    let cur_rank =
        stat.sorted_values.partition_point(|&v| v < pred.threshold) as f64 / (n as f64 - 1.0);
    let delta = (rng.uniform() - 0.5) * span;
    let q = (cur_rank + delta).clamp(0.0, 1.0);
    if let Some(t) = stat.quantile(q) {
        pred.threshold = t;
    }
}

/// Standard (fine) threshold nudge: +/- ~15 percentiles around the current rank.
fn perturb_threshold(pred: &mut Predicate, stat: &FeatureStat, rng: &mut Rng) {
    perturb_threshold_span(pred, stat, 0.30, rng);
}

/// Layers present in `genome`, as a set, for finding under-represented layers.
fn layers_present(genome: &Genome) -> BTreeSet<Layer> {
    genome.predicates.iter().map(|p| p.layer).collect()
}

/// Choose an under-represented layer for `genome` that the catalog can actually supply a feature
/// for: prefer a layer with NO predicate yet (and ≥1 observed feature). Falls back to any catalog
/// layer the genome lacks. `None` if the genome already spans every available layer.
fn underrepresented_layer(
    genome: &Genome,
    catalog: &FeatureCatalog,
    rng: &mut Rng,
) -> Option<Layer> {
    let present = layers_present(genome);
    let mut missing: Vec<Layer> = ALL_LAYERS
        .iter()
        .copied()
        .filter(|l| !present.contains(l) && !catalog.stats_in_layer(*l).is_empty())
        .collect();
    if missing.is_empty() {
        return None;
    }
    let i = rng.below(missing.len());
    Some(missing.swap_remove(i))
}

/// Mutate `genome` in place-ish (returns a new genome). One operator is chosen at random:
///
/// * `0` perturb a threshold by a small step (fine-tune);
/// * `1` flip a comparison operator (`Gt`<->`Lt`, `Gte`<->`Lte`);
/// * `2` swap a predicate for a fresh one over a different feature;
/// * `3` add a predicate over a new feature (if room);
/// * `4` remove a predicate (never empties the genome);
/// * `5` (exploration) ADD a predicate from an UNDER-REPRESENTED layer the genome lacks, so the
///   conjunction spans more of the regime×location×trigger taxonomy;
/// * `6` (exploration) SWAP a predicate's feature to a DIFFERENT layer's feature, jumping the
///   search to a new region of feature space;
/// * `7` (exploration) take a BOLD threshold step (a much wider percentile jump) to escape a
///   local optimum.
///
/// The choice is random but the result is always a valid, non-empty genome over catalog features
/// with no duplicate feature keys and at most [`MAX_PREDICATES`] predicates. Deterministic given
/// `rng`. The added exploration operators (5–7) widen the search without weakening any guard: they
/// only ever introduce OBSERVED catalog features at observed thresholds.
pub fn mutate(genome: &Genome, catalog: &FeatureCatalog, rng: &mut Rng) -> Genome {
    if catalog.is_empty() {
        return genome.clone();
    }
    let mut g = genome.clone();
    let roll = rng.below(8);

    match roll {
        // 0: perturb a threshold (fine step).
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
        4 => {
            if g.predicates.len() > 1 {
                let i = rng.below(g.predicates.len());
                g.predicates.remove(i);
            }
        }
        // 5: EXPLORE — add a predicate from an under-represented layer (one the genome lacks),
        //    so the conjunction spans more of the taxonomy. Falls back to nothing if full.
        5 => {
            if g.predicates.len() < MAX_PREDICATES {
                if let Some(layer) = underrepresented_layer(&g, catalog, rng) {
                    if let Some(p) = draw_predicate_in_layers(catalog, &[layer], rng) {
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
        }
        // 6: EXPLORE — swap a predicate's feature to a DIFFERENT layer's feature, jumping the
        //    search to a new region. Only swaps when a fresh non-duplicate feature is drawn.
        6 => {
            if !g.predicates.is_empty() {
                let i = rng.below(g.predicates.len());
                let cur_layer = g.predicates[i].layer;
                // Target layers: every layer except the current one that the catalog can supply.
                let targets: Vec<Layer> = ALL_LAYERS
                    .iter()
                    .copied()
                    .filter(|l| *l != cur_layer && !catalog.stats_in_layer(*l).is_empty())
                    .collect();
                if !targets.is_empty() {
                    let layer = targets[rng.below(targets.len())];
                    if let Some(p) = draw_predicate_in_layers(catalog, &[layer], rng) {
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
        }
        // 7: EXPLORE — take a BOLD threshold step (wide percentile jump) to escape a local optimum.
        _ => {
            if !g.predicates.is_empty() {
                let i = rng.below(g.predicates.len());
                let key = g.predicates[i].feature_key.clone();
                if let Some(stat) = stat_for(catalog, &key) {
                    // Span 0.8 => up to +/- 40 percentiles, a much larger move than operator 0.
                    perturb_threshold_span(&mut g.predicates[i], stat, 0.80, rng);
                }
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
    use se_core::{Bar, Horizon, Side, Ticker};
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
                    ticker: Ticker::SPY,
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
            ticker: Ticker::SPY,
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
    fn mutate_is_deterministic_given_seed() {
        let cat = catalog();
        let g = {
            let mut rng = Rng::seeded(17, 0);
            random_genome(&cat, Horizon::Swing, &mut rng, 3).unwrap()
        };
        let seq_a: Vec<String> = {
            let mut rng = Rng::seeded(21, 0);
            (0..100)
                .map(|_| mutate(&g, &cat, &mut rng).describe())
                .collect()
        };
        let seq_b: Vec<String> = {
            let mut rng = Rng::seeded(21, 0);
            (0..100)
                .map(|_| mutate(&g, &cat, &mut rng).describe())
                .collect()
        };
        assert_eq!(
            seq_a, seq_b,
            "mutation must be deterministic given a fixed seed"
        );
    }

    #[test]
    fn exploration_add_increases_layer_diversity() {
        let cat = catalog();
        // Start from a single-layer (trigger-only) genome, then drive the under-represented-layer
        // add operator (roll == 5) until it brings in a new layer.
        let g = Genome::new(
            Side::Long,
            Horizon::Swing,
            vec![Predicate {
                layer: Layer::Trigger,
                feature_key: "trigger.rsi14".into(),
                op: CmpOp::Gt,
                threshold: 30.0,
            }],
        );
        let mut rng = Rng::seeded(4, 0);
        let mut saw_multi_layer = false;
        for _ in 0..500 {
            let m = mutate(&g, &cat, &mut rng);
            let layers: BTreeSet<Layer> = m.predicates.iter().map(|p| p.layer).collect();
            if layers.len() > 1 {
                saw_multi_layer = true;
                break;
            }
        }
        assert!(
            saw_multi_layer,
            "exploration operators should reach a multi-layer genome"
        );
    }

    #[test]
    fn underrepresented_layer_picks_a_missing_catalog_layer() {
        let cat = catalog(); // has regime, trigger, location
        let g = Genome::new(
            Side::Long,
            Horizon::Swing,
            vec![Predicate {
                layer: Layer::Trigger,
                feature_key: "trigger.rsi14".into(),
                op: CmpOp::Gt,
                threshold: 30.0,
            }],
        );
        let mut rng = Rng::seeded(8, 0);
        for _ in 0..50 {
            let l = underrepresented_layer(&g, &cat, &mut rng).unwrap();
            assert_ne!(l, Layer::Trigger, "must pick a layer not already present");
            assert!(
                !cat.stats_in_layer(l).is_empty(),
                "picked layer must have catalog features"
            );
        }
    }

    #[test]
    fn mutation_operators_keep_genome_valid_over_many_rolls() {
        // Exercise the full operator set (roll 0..8) heavily and assert invariants every time.
        let cat = catalog();
        let mut rng = Rng::seeded(55, 0);
        let g = random_genome(&cat, Horizon::Swing, &mut rng, 4).unwrap();
        for _ in 0..1000 {
            let m = mutate(&g, &cat, &mut rng);
            assert!(!m.predicates.is_empty());
            assert!(m.predicates.len() <= MAX_PREDICATES);
            for p in &m.predicates {
                assert!(cat.stats.iter().any(|s| s.key == p.feature_key));
                // Threshold stays inside the observed range.
                let stat = cat.stats.iter().find(|s| s.key == p.feature_key).unwrap();
                let lo = *stat.sorted_values.first().unwrap();
                let hi = *stat.sorted_values.last().unwrap();
                assert!((lo..=hi).contains(&p.threshold));
            }
            let uniq: BTreeSet<&str> = m
                .predicates
                .iter()
                .map(|p| p.feature_key.as_str())
                .collect();
            assert_eq!(
                uniq.len(),
                m.predicates.len(),
                "no duplicate features allowed"
            );
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
