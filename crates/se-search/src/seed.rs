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

    /// Observed stats restricted to one layer (cheap; the catalog is small).
    pub fn stats_in_layer(&self, layer: Layer) -> Vec<&FeatureStat> {
        self.stats.iter().filter(|s| s.layer == layer).collect()
    }

    /// True if the catalog has at least one observed feature on an ACTIONABLE layer
    /// ([`Layer::Trigger`] or [`Layer::Location`]) — the entry-condition layers the promotion
    /// guardrail requires. Used to decide whether biased (promotable-shaped) seeding is even
    /// possible for this dataset.
    pub fn has_actionable_feature(&self) -> bool {
        self.stats
            .iter()
            .any(|s| matches!(s.layer, Layer::Trigger | Layer::Location))
    }

    /// Lookup a stat by exact dotted key (used by the hand-crafted archetypes to bind a threshold
    /// to the OBSERVED distribution of a known feature; archetypes whose key is absent are dropped).
    pub fn stat_for_key(&self, key: &str) -> Option<&FeatureStat> {
        self.stats.iter().find(|s| s.key == key)
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

/// Draw one predicate restricted to a set of layers (e.g. the actionable Trigger/Location layers,
/// or the contextual Regime/Tradeability layers). Same observed-quantile threshold draw as
/// [`draw_predicate`]; `None` if no observed feature exists in those layers. This is the building
/// block for guardrail-friendly seeding: pairing an ACTIONABLE predicate with a CONTEXT predicate
/// yields a genome that is promotable (carries a real entry trigger) rather than regime-only.
pub fn draw_predicate_in_layers(
    catalog: &FeatureCatalog,
    layers: &[Layer],
    rng: &mut Rng,
) -> Option<Predicate> {
    let pool: Vec<&FeatureStat> = catalog
        .stats
        .iter()
        .filter(|s| layers.contains(&s.layer))
        .collect();
    let stat = rng.choice(&pool)?;
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

/// The entry-condition (actionable) layers the promotion guardrail keys on.
const ACTIONABLE_LAYERS: [Layer; 2] = [Layer::Trigger, Layer::Location];
/// The conditioning (context) layers — pairing one of these with an actionable predicate makes a
/// seed regime-aware AND promotable (not regime-only).
const CONTEXT_LAYERS: [Layer; 3] = [Layer::Regime, Layer::Tradeability, Layer::Event];

/// Build a single random genome that is GUARANTEED promotable-shaped: at least one Trigger/Location
/// (actionable) predicate, plus — when the catalog has any context feature — a Regime/Tradeability/
/// Event predicate for regime context. Extra predicates are then drawn from the full catalog up to
/// `max_predicates`, over distinct features. Returns `None` only if no actionable feature exists.
fn random_genome_promotable(
    catalog: &FeatureCatalog,
    horizon: Horizon,
    rng: &mut Rng,
    max_predicates: usize,
) -> Option<Genome> {
    let mut predicates: Vec<Predicate> = Vec::new();
    let mut used: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    // 1) Mandatory actionable predicate (Trigger or Location).
    let actionable = draw_predicate_in_layers(catalog, &ACTIONABLE_LAYERS, rng)?;
    used.insert(actionable.feature_key.clone());
    predicates.push(actionable);

    // 2) A context predicate (Regime/Tradeability/Event) when available, so the seed is regime-aware
    //    and not a bare trigger.
    if !catalog.stats_in_layer(Layer::Regime).is_empty()
        || !catalog.stats_in_layer(Layer::Tradeability).is_empty()
        || !catalog.stats_in_layer(Layer::Event).is_empty()
    {
        if let Some(ctx) = draw_predicate_in_layers(catalog, &CONTEXT_LAYERS, rng) {
            if used.insert(ctx.feature_key.clone()) {
                predicates.push(ctx);
            }
        }
    }

    // 3) Optionally pad with more distinct predicates from anywhere in the catalog.
    let target = (2 + rng.below(max_predicates.max(1))).min(max_predicates.max(2));
    for _ in 0..(target * 4) {
        if predicates.len() >= target {
            break;
        }
        if let Some(p) = draw_predicate(catalog, rng) {
            if used.insert(p.feature_key.clone()) {
                predicates.push(p);
            }
        }
    }

    let side = if rng.chance(0.5) {
        Side::Long
    } else {
        Side::Short
    };
    Some(Genome::new(side, horizon, predicates))
}

/// A hand-crafted archetype template: a named trading hypothesis expressed as `(layer, key, op,
/// quantile)` predicate templates plus a side. The threshold for each predicate is bound at the
/// given OBSERVED empirical quantile of that feature, so archetypes never invent thresholds and an
/// archetype whose feature is absent from the catalog is dropped. Every archetype carries at least
/// one Trigger/Location predicate so it is promotable-shaped.
struct Archetype {
    name: &'static str,
    side: Side,
    /// `(layer, dotted_key, op, quantile_in_[0,1])`.
    preds: &'static [(Layer, &'static str, CmpOp, f64)],
}

/// The library of hand-crafted seeds. These span long/short and trend-follow/mean-revert, using
/// only real feature keys emitted by the layer modules (regime / location / trigger). Each pairs an
/// actionable Trigger/Location predicate with a Regime context predicate, so they are diverse AND
/// guardrail-friendly out of the gate. Thresholds are bound to observed quantiles at seed time.
const ARCHETYPES: &[Archetype] = &[
    // --- Trend-following longs --------------------------------------------------------------
    // Strong-trend continuation: high ADX regime + price extended above the 50DMA + momentum up.
    Archetype {
        name: "trend_follow_long",
        side: Side::Long,
        preds: &[
            (Layer::Regime, "regime.adx", CmpOp::Gt, 0.65),
            (Layer::Location, "location.dist_50dma", CmpOp::Gt, 0.60),
            (Layer::Trigger, "trigger.momentum_roc", CmpOp::Gt, 0.60),
        ],
    },
    // Relative-strength breakout long: leadership vs SPY near the top of the recent range.
    Archetype {
        name: "rs_breakout_long",
        side: Side::Long,
        preds: &[
            (Layer::Regime, "regime.adx", CmpOp::Gt, 0.55),
            (Layer::Trigger, "trigger.rs_vs_spy", CmpOp::Gt, 0.70),
            (
                Layer::Location,
                "location.pct_range_position",
                CmpOp::Gt,
                0.70,
            ),
        ],
    },
    // --- Mean-reversion longs ---------------------------------------------------------------
    // Pullback-in-uptrend: oversold RSI while still above the 200DMA (buy the dip).
    Archetype {
        name: "mean_revert_long",
        side: Side::Long,
        preds: &[
            (Layer::Trigger, "trigger.rsi14", CmpOp::Lt, 0.25),
            (Layer::Location, "location.dist_200dma", CmpOp::Gt, 0.45),
            (Layer::Regime, "regime.adx", CmpOp::Lt, 0.50),
        ],
    },
    // Reclaim of anchored VWAP from below in a low-trend regime.
    Archetype {
        name: "vwap_reclaim_long",
        side: Side::Long,
        preds: &[
            (
                Layer::Location,
                "location.anchored_vwap_dist",
                CmpOp::Lt,
                0.30,
            ),
            (Layer::Trigger, "trigger.rsi14", CmpOp::Lt, 0.35),
            (Layer::Regime, "regime.rv_percentile", CmpOp::Lt, 0.60),
        ],
    },
    // --- Trend-following / momentum shorts --------------------------------------------------
    // Downtrend continuation: high ADX, price extended below the 50DMA, momentum down.
    Archetype {
        name: "trend_follow_short",
        side: Side::Short,
        preds: &[
            (Layer::Regime, "regime.adx", CmpOp::Gt, 0.65),
            (Layer::Location, "location.dist_50dma", CmpOp::Lt, 0.40),
            (Layer::Trigger, "trigger.momentum_roc", CmpOp::Lt, 0.40),
        ],
    },
    // Relative-weakness breakdown short: laggard vs SPY near the bottom of the recent range.
    Archetype {
        name: "rs_breakdown_short",
        side: Side::Short,
        preds: &[
            (Layer::Regime, "regime.adx", CmpOp::Gt, 0.55),
            (Layer::Trigger, "trigger.rs_vs_spy", CmpOp::Lt, 0.30),
            (
                Layer::Location,
                "location.pct_range_position",
                CmpOp::Lt,
                0.30,
            ),
        ],
    },
    // --- Mean-reversion shorts --------------------------------------------------------------
    // Overbought fade: hot RSI well above the 200DMA in a non-trending tape.
    Archetype {
        name: "mean_revert_short",
        side: Side::Short,
        preds: &[
            (Layer::Trigger, "trigger.rsi14", CmpOp::Gt, 0.75),
            (Layer::Location, "location.dist_200dma", CmpOp::Gt, 0.70),
            (Layer::Regime, "regime.adx", CmpOp::Lt, 0.50),
        ],
    },
];

/// Materialize the archetypes that are EXPRESSIBLE on this catalog: every templated predicate's
/// feature must be observed, and its threshold is bound to that feature's observed quantile. An
/// archetype referencing any absent feature is skipped (we never fabricate a key/threshold). The
/// result is deterministic (no RNG) so the archetype block of a seed population is stable.
pub fn archetype_seeds(catalog: &FeatureCatalog, horizon: Horizon) -> Vec<Genome> {
    let mut out = Vec::new();
    for arch in ARCHETYPES {
        let mut predicates: Vec<Predicate> = Vec::new();
        let mut ok = true;
        let mut used: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for (layer, key, op, q) in arch.preds {
            // Drop duplicate-feature templates defensively (none currently, but keep it valid).
            if !used.insert(*key) {
                continue;
            }
            match catalog.stat_for_key(key).and_then(|s| s.quantile(*q)) {
                Some(threshold) => predicates.push(Predicate {
                    layer: *layer,
                    feature_key: (*key).to_string(),
                    op: *op,
                    threshold,
                }),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        // Keep only fully-expressible archetypes that retain an actionable predicate.
        let actionable = predicates
            .iter()
            .any(|p| matches!(p.layer, Layer::Trigger | Layer::Location));
        if ok && actionable && !predicates.is_empty() {
            tracing::debug!(archetype = arch.name, "seeded hand-crafted archetype");
            out.push(Genome::new(arch.side, horizon, predicates));
        }
    }
    out
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

/// Default fraction of the RANDOM portion of a seed population biased to be promotable-shaped
/// (>= one Trigger/Location predicate paired with regime context). The rest are drawn fully at
/// random for diversity. Chosen so a healthy majority of every generation can clear the
/// actionable-predicate guardrail while the search still explores regime-only / unusual shapes.
pub const DEFAULT_PROMOTABLE_BIAS: f64 = 0.6;

/// Seed an initial population of `size` distinct genomes for `horizon`, drawn deterministically
/// from `rng`. The population leads with the hand-crafted [`archetype_seeds`] expressible on this
/// catalog (diverse, promotable-shaped starting points), then fills with random genomes a healthy
/// fraction of which are biased to be promotable-shaped (see [`DEFAULT_PROMOTABLE_BIAS`]). Both
/// sides are represented (subject to the random draws). Duplicate genomes are de-duplicated by
/// their `describe()` string.
///
/// The public signature is unchanged; the diversity bias uses the default. See
/// [`seed_population_biased`] to control the bias explicitly.
pub fn seed_population(
    catalog: &FeatureCatalog,
    horizon: Horizon,
    size: usize,
    rng: &mut Rng,
    max_predicates: usize,
) -> Vec<Genome> {
    seed_population_biased(
        catalog,
        horizon,
        size,
        rng,
        max_predicates,
        DEFAULT_PROMOTABLE_BIAS,
    )
}

/// As [`seed_population`], but with an explicit `promotable_bias` in `[0,1]` controlling the
/// fraction of the random fill that is forced promotable-shaped. `0.0` reproduces the legacy
/// fully-random fill (still preceded by the archetype seeds). Deterministic given `rng`.
pub fn seed_population_biased(
    catalog: &FeatureCatalog,
    horizon: Horizon,
    size: usize,
    rng: &mut Rng,
    max_predicates: usize,
    promotable_bias: f64,
) -> Vec<Genome> {
    let bias = promotable_bias.clamp(0.0, 1.0);
    let mut out: Vec<Genome> = Vec::new();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    // Lead with the expressible hand-crafted archetypes (deterministic; no RNG draws), so every
    // seed population starts from a spread of long/short, trend/mean-revert hypotheses.
    for g in archetype_seeds(catalog, horizon) {
        if out.len() >= size {
            break;
        }
        if seen.insert(g.describe()) {
            out.push(g);
        }
    }

    // Fill the remainder with random genomes. A `bias` fraction are forced promotable-shaped
    // (>= one actionable predicate + regime context); the rest are fully random for exploration.
    let can_bias = catalog.has_actionable_feature();
    for _ in 0..(size * 8) {
        if out.len() >= size {
            break;
        }
        let want_promotable = can_bias && rng.chance(bias);
        let g = if want_promotable {
            random_genome_promotable(catalog, horizon, rng, max_predicates)
        } else {
            random_genome(catalog, horizon, rng, max_predicates)
        };
        if let Some(g) = g {
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

    /// A catalog rich enough to express several hand-crafted archetypes.
    fn rich_catalog() -> FeatureCatalog {
        let series = |scale: f64, off: f64| {
            (0..60)
                .map(|i| (i as f64) * scale + off)
                .collect::<Vec<_>>()
        };
        let w = window_with(&[
            ("regime.adx", &series(1.0, 5.0)),
            ("regime.rv_percentile", &series(0.01, 0.0)),
            ("trigger.rsi14", &series(1.5, 0.0)),
            ("trigger.momentum_roc", &series(0.2, -6.0)),
            ("trigger.rs_vs_spy", &series(0.05, -1.5)),
            ("location.dist_50dma", &series(0.1, -3.0)),
            ("location.dist_200dma", &series(0.08, -2.0)),
            ("location.pct_range_position", &series(0.016, 0.0)),
            ("location.anchored_vwap_dist", &series(0.07, -2.0)),
        ]);
        FeatureCatalog::from_windows(&[w], 3)
    }

    #[test]
    fn archetypes_use_only_observed_keys_and_are_actionable() {
        let cat = rich_catalog();
        let arch = archetype_seeds(&cat, Horizon::Swing);
        // At least a few archetypes must be expressible on this catalog.
        assert!(
            arch.len() >= 4,
            "expected several archetypes, got {}",
            arch.len()
        );
        let mut long = 0;
        let mut short = 0;
        for g in &arch {
            // Every predicate key is an observed catalog feature.
            for p in &g.predicates {
                assert!(
                    cat.stats.iter().any(|s| s.key == p.feature_key),
                    "archetype predicate {} not in catalog",
                    p.feature_key
                );
                // Threshold sits inside the observed range of its feature.
                let stat = cat.stat_for_key(&p.feature_key).unwrap();
                let lo = *stat.sorted_values.first().unwrap();
                let hi = *stat.sorted_values.last().unwrap();
                assert!((lo..=hi).contains(&p.threshold));
            }
            // Each archetype carries an actionable Trigger/Location predicate (promotable-shaped).
            assert!(
                crate::score::genome_has_actionable_predicate(g),
                "archetype {:?} is not actionable",
                g.describe()
            );
            match g.side {
                Side::Long => long += 1,
                Side::Short => short += 1,
            }
        }
        // The archetype library spans both sides.
        assert!(long > 0 && short > 0, "archetypes must span long and short");
    }

    #[test]
    fn archetypes_deterministic_and_drop_absent_features() {
        // A catalog with only adx + rsi14: most archetypes reference absent features -> dropped,
        // but the call is still deterministic and never fabricates a key.
        let w = window_with(&[
            ("regime.adx", &(0..40).map(|i| i as f64).collect::<Vec<_>>()),
            (
                "trigger.rsi14",
                &(0..40).map(|i| i as f64).collect::<Vec<_>>(),
            ),
        ]);
        let cat = FeatureCatalog::from_windows(&[w], 3);
        let a = archetype_seeds(&cat, Horizon::Swing);
        let b = archetype_seeds(&cat, Horizon::Swing);
        let da: Vec<String> = a.iter().map(|g| g.describe()).collect();
        let db: Vec<String> = b.iter().map(|g| g.describe()).collect();
        assert_eq!(da, db, "archetype materialization must be deterministic");
        for g in &a {
            for p in &g.predicates {
                assert!(matches!(
                    p.feature_key.as_str(),
                    "regime.adx" | "trigger.rsi14"
                ));
            }
        }
    }

    #[test]
    fn biased_seeding_lifts_promotable_fraction() {
        let cat = rich_catalog();
        // High bias: a clear majority of the population should be promotable-shaped.
        let pop = {
            let mut r = Rng::seeded(7, 0);
            seed_population_biased(&cat, Horizon::Swing, 40, &mut r, 4, 0.8)
        };
        let promotable = pop
            .iter()
            .filter(|g| crate::score::genome_has_actionable_predicate(g))
            .count();
        // Archetypes + the biased fill should put promotable genomes well over half.
        assert!(
            promotable * 2 > pop.len(),
            "expected a majority promotable-shaped, got {}/{}",
            promotable,
            pop.len()
        );

        // Zero bias still works and (preceded by archetypes) yields a valid population.
        let pop0 = {
            let mut r = Rng::seeded(7, 0);
            seed_population_biased(&cat, Horizon::Swing, 40, &mut r, 4, 0.0)
        };
        assert!(!pop0.is_empty());
        for g in &pop0 {
            assert!(!g.predicates.is_empty() && g.predicates.len() <= 4);
        }
    }

    #[test]
    fn promotable_seed_pairs_actionable_with_context() {
        let cat = rich_catalog();
        let mut r = Rng::seeded(3, 0);
        for _ in 0..50 {
            let g = random_genome_promotable(&cat, Horizon::Swing, &mut r, 4).unwrap();
            // Always actionable.
            assert!(crate::score::genome_has_actionable_predicate(&g));
            // And regime-aware: carries at least one context-layer predicate.
            assert!(g
                .predicates
                .iter()
                .any(|p| matches!(p.layer, Layer::Regime | Layer::Tradeability | Layer::Event)));
            // No duplicate features.
            let uniq: std::collections::BTreeSet<&str> = g
                .predicates
                .iter()
                .map(|p| p.feature_key.as_str())
                .collect();
            assert_eq!(uniq.len(), g.predicates.len());
            assert!(g.predicates.len() <= 4);
        }
    }

    #[test]
    fn seed_population_is_diverse() {
        let cat = rich_catalog();
        let mut r = Rng::seeded(99, 0);
        let pop = seed_population(&cat, Horizon::Swing, 30, &mut r, 4);
        // Distinct genomes (dedup by describe()).
        let uniq: std::collections::BTreeSet<String> = pop.iter().map(|g| g.describe()).collect();
        assert_eq!(
            uniq.len(),
            pop.len(),
            "seed population must be de-duplicated"
        );
        // Both sides represented.
        assert!(pop.iter().any(|g| g.side == Side::Long));
        assert!(pop.iter().any(|g| g.side == Side::Short));
        // A spread of feature keys is exercised (not collapsed onto one feature).
        let keys: std::collections::BTreeSet<&str> = pop
            .iter()
            .flat_map(|g| g.predicates.iter().map(|p| p.feature_key.as_str()))
            .collect();
        assert!(keys.len() >= 4, "expected feature diversity, got {keys:?}");
    }
}
