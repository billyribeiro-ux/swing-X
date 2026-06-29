//! [`PopulationManager`] — the search/mutation loop on the OOS scoreboard.
//!
//! One generation: SEARCH (the current population of genomes) -> FIT (in-sample, done
//! implicitly inside the worker's CPCV) -> SCORE (out-of-sample, the ONLY ranking input) ->
//! KEEP survivors (gate-pass OR positive cost-aware OOS expectancy with a clean overfit
//! signature) / MUTATE the promising ones / KILL the rest. Survivors and scores are persisted
//! every generation so a crash leaves a usable scoreboard.
//!
//! Tiny-dataset safety: a genome that produces too few labeled entries to validate is logged
//! and skipped — never a crash, never a fabricated score.

use std::collections::BTreeMap;

use futures::StreamExt;
use se_core::{
    Genome, HorizonProfile, Result, RiskModel, Scanner, Strategy, StrategyId, StrategyStatus,
    Ticker,
};
use se_store::Store;
use se_validation::ValidationHarness;

use crate::backtest::{assemble, backtest};
use crate::feature_matrix::{build_window, FeatureWindow};
use crate::risk_search::RiskSpace;
use crate::rng::Rng;
use crate::score::{
    genome_has_actionable_predicate, score_oos, OosScore, ScoreConfig, MIN_ACTED_TO_PROMOTE,
    MIN_ENTRIES_TO_VALIDATE,
};
use crate::seed::{seed_population, FeatureCatalog};
use crate::{genome_ops, persist};

/// One evaluated member of the population: its strategy identity, genome, and (if it had enough
/// data) its OOS score.
#[derive(Debug, Clone)]
pub struct Evaluated {
    pub strategy: Strategy,
    pub score: Option<OosScore>,
    /// Number of labeled entries the genome produced across the universe (cohort size).
    pub n_entries: usize,
}

impl Evaluated {
    /// Whether this member survives into the next generation.
    pub fn survives(&self) -> bool {
        self.score
            .as_ref()
            .map(|s| s.is_survivor())
            .unwrap_or(false)
    }

    /// Whether this member is promotable. The hard gate passing is necessary but NOT sufficient:
    /// the search guardrail also requires (a) the genome carries an actionable entry condition
    /// (a trigger/location predicate — not a regime/tradeability/event-only conjunction that fires
    /// trivially) and (b) the validation acted on at least [`MIN_ACTED_TO_PROMOTE`] OOS trades, so
    /// there is a real out-of-sample track record behind the promotion. A gate-passer that fails
    /// the guardrail is NOT promoted but is still kept as a survivor (see `is_survivor`), never
    /// retired for this reason.
    pub fn promotable(&self) -> bool {
        match &self.score {
            Some(s) => {
                s.passed_gate
                    && genome_has_actionable_predicate(&self.strategy.genome)
                    && s.n_acted_oos as usize >= MIN_ACTED_TO_PROMOTE
            }
            None => false,
        }
    }
}

/// Configuration for the search loop.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Active horizon profile (drives barriers, costs, cadence — the P8 axis).
    pub profile: HorizonProfile,
    /// Universe of tickers to backtest each genome across.
    pub universe: Vec<Ticker>,
    /// Window start/end (decision bars).
    pub from: chrono::DateTime<chrono::Utc>,
    pub to: chrono::DateTime<chrono::Utc>,
    /// Base RNG seed (mixed with generation). Deterministic search.
    pub base_seed: u64,
    /// Max predicates per seeded genome.
    pub max_predicates: usize,
    /// Minimum observations required for a feature to enter the catalog.
    pub min_feature_obs: usize,
    /// OOS scoring shape.
    pub score: ScoreConfig,
    /// The operator's ground-rule risk geometry (stop/target). Seeds/mutations explore around
    /// it, or — if [`SearchConfig::lock_risk`] — every genome is pinned to exactly this model.
    pub risk: RiskModel,
    /// When true, the operator's `risk` model is fixed: the search optimizes only the entry
    /// conditions. When false (default), risk geometry is explored and the OOS scoreboard keeps
    /// the best one.
    pub lock_risk: bool,
    /// Which scanner this population belongs to (ETF vs equity). Tags every persisted strategy so
    /// the two populations stay separate on the scoreboard, in the journal, and in promotion.
    pub scanner: Scanner,
}

/// Default deterministic search seed (never derived from the clock — see [`crate::rng`]).
pub const DEFAULT_SEARCH_SEED: u64 = 0x0005_EED0_F5EA_C401;

impl SearchConfig {
    pub fn new(profile: HorizonProfile, universe: Vec<Ticker>) -> Self {
        let to = chrono::Utc::now();
        let from = to - chrono::Duration::days(730);
        SearchConfig {
            profile,
            universe,
            from,
            to,
            base_seed: DEFAULT_SEARCH_SEED,
            max_predicates: 3,
            min_feature_obs: 30,
            score: ScoreConfig::default(),
            // Default ground rules: the legacy profile geometry (so a search with no `--stop`
            // flag behaves exactly as before, while still being explored).
            risk: RiskModel::from_profile(&profile),
            lock_risk: false,
            scanner: Scanner::Etf,
        }
    }
}

/// The population manager: owns the store, the validation harness, and the search config.
pub struct PopulationManager<'a> {
    store: &'a Store,
    harness: &'a ValidationHarness,
    cfg: SearchConfig,
    /// Materialized feature windows per ticker (computed once, reused every generation).
    windows: Vec<FeatureWindow>,
    catalog: FeatureCatalog,
    /// The risk-geometry sampling space (built around the operator's ground-rule model).
    risk_space: RiskSpace,
}

impl<'a> PopulationManager<'a> {
    /// Construct and materialize the feature windows for the universe. This is the expensive,
    /// one-time step (per-bar feature computation across the window).
    pub async fn new(
        store: &'a Store,
        harness: &'a ValidationHarness,
        cfg: SearchConfig,
    ) -> Result<Self> {
        // Build each ticker's window concurrently (each is independent and I/O-bound).
        let built: Vec<FeatureWindow> = futures::stream::iter(cfg.universe.clone())
            .map(|t| build_window(store, t, cfg.from, cfg.to, cfg.profile))
            .buffer_unordered(4)
            .collect::<Vec<Result<FeatureWindow>>>()
            .await
            .into_iter()
            .collect::<Result<Vec<FeatureWindow>>>()?;
        let mut windows = Vec::new();
        for w in built {
            if w.points.is_empty() {
                tracing::warn!(ticker = %w.ticker, "no decision bars in window; skipping");
                continue;
            }
            windows.push(w);
        }
        let catalog = FeatureCatalog::from_windows(&windows, cfg.min_feature_obs);
        let risk_space = RiskSpace::new(cfg.risk);
        Ok(PopulationManager {
            store,
            harness,
            cfg,
            windows,
            catalog,
            risk_space,
        })
    }

    /// Apply the risk geometry to a genome: pin to the operator's model when locked, otherwise
    /// sample a fresh geometry from the configurable space.
    fn assign_seed_risk(&self, mut g: Genome, rng: &mut Rng) -> Genome {
        g.risk = crate::risk_search::seed_risk(&self.risk_space, self.cfg.lock_risk, rng);
        g
    }

    /// The feature catalog the search draws from (observed keys + quantiles).
    pub fn catalog(&self) -> &FeatureCatalog {
        &self.catalog
    }

    /// The materialized windows (for diagnostics / signal generation reuse).
    pub fn windows(&self) -> &[FeatureWindow] {
        &self.windows
    }

    /// Backtest a genome across the whole universe and assemble one combined dataset (entries
    /// from all tickers, re-sorted ascending by entry time as the writer requires).
    fn build_dataset(&self, genome: &Genome) -> (Vec<se_mlclient::DatasetRow>, usize) {
        let mut all = Vec::new();
        for w in &self.windows {
            let res = backtest(genome, w, &self.cfg.profile);
            let rows = assemble(&res);
            all.extend(rows);
        }
        all.sort_by(|a, b| a.ts.cmp(&b.ts));
        let n = all.len();
        (all, n)
    }

    /// Evaluate one genome end-to-end: backtest -> assemble -> OOS score. Tiny datasets return
    /// `Ok(Evaluated{score: None})` (logged, skipped) rather than erroring.
    async fn evaluate_genome(&self, strategy: Strategy) -> Result<Evaluated> {
        let (rows, n_entries) = self.build_dataset(&strategy.genome);
        if n_entries < MIN_ENTRIES_TO_VALIDATE {
            tracing::info!(
                strategy = %strategy.id,
                n_entries,
                min = MIN_ENTRIES_TO_VALIDATE,
                "too few labeled entries to validate; skipping genome"
            );
            return Ok(Evaluated {
                strategy,
                score: None,
                n_entries,
            });
        }
        // A per-genome validation error (e.g. the worker's 422 "no OOS observations produced by
        // CPCV" for a thin/degenerate cohort) must NOT abort the whole search — log and skip that
        // genome. A genuinely-down worker is caught up front by the CLI's health probe.
        let score = match score_oos(
            self.harness,
            strategy.id,
            &rows,
            &self.cfg.profile,
            self.cfg.score,
        )
        .await
        {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(strategy = %strategy.id, n_entries, error = %e, "OOS scoring failed for genome; skipping");
                None
            }
        };
        Ok(Evaluated {
            strategy,
            score,
            n_entries,
        })
    }

    /// Persist a strategy and (if scored) its OOS score.
    async fn persist(&self, ev: &Evaluated) -> Result<()> {
        persist::upsert_strategy(self.store, &ev.strategy, self.cfg.scanner).await?;
        if let Some(score) = &ev.score {
            let fold_spec = serde_json::json!({
                "n_groups": self.cfg.score.n_groups,
                "k_test_groups": self.cfg.score.k_test_groups,
                "n_trials": self.cfg.score.n_trials,
                "embargo_bars": self.cfg.profile.embargo_bars,
                "purge": true,
                // Cohort size (entry count) so signal generation can report a real `cohort_n`
                // without re-backtesting.
                "n_entries": ev.n_entries,
                // The risk geometry this genome was scored on (provenance for the scoreboard).
                "risk": ev.strategy.genome.risk,
                "lock_risk": self.cfg.lock_risk,
            });
            persist::insert_oos_score(self.store, score, &fold_spec).await?;
        }
        Ok(())
    }

    /// Run the search for `generations` generations with `per_gen` genomes per generation.
    /// Returns the final evaluated population (sorted best-first by OOS), and the cumulative
    /// best scores seen. Persists strategies + scores every generation.
    pub async fn evolve(&self, generations: u32, per_gen: usize) -> Result<EvolveOutcome> {
        if self.catalog.is_empty() {
            return Err(se_core::Error::msg(
                "no features observed in the window — cannot seed a population (run `se scan` first)",
            ));
        }

        let mut all_evaluated: Vec<Evaluated> = Vec::new();
        let mut survivors: Vec<Genome> = Vec::new();

        for gen in 0..generations {
            let mut rng = Rng::seeded(self.cfg.base_seed, gen);

            // Build this generation's genomes: survivors + their mutations/crossovers, topped up
            // with fresh seeds to `per_gen`.
            let genomes = self.next_generation(&survivors, per_gen, &mut rng);

            tracing::info!(generation = gen, n = genomes.len(), "evaluating generation");

            let mut evaluated: Vec<Evaluated> = Vec::new();
            for genome in genomes {
                let mut strategy = Strategy::new(genome);
                strategy.generation = gen;
                let ev = self.evaluate_genome(strategy).await?;
                self.persist(&ev).await?;
                evaluated.push(ev);
            }

            // KEEP survivors, MUTATE-promising for next gen; KILL rest (status update).
            survivors = self.select_survivors(&evaluated).await?;
            all_evaluated.extend(evaluated);
        }

        // Final leaderboard: best OOS first. De-dup by strategy id (keep best score).
        all_evaluated.sort_by(|a, b| match (&a.score, &b.score) {
            (Some(sa), Some(sb)) => sa.cmp_best_first(sb),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        });

        Ok(EvolveOutcome {
            evaluated: all_evaluated,
            profile: self.cfg.profile,
        })
    }

    /// Assemble the next generation's genomes from survivors + fresh seeds.
    fn next_generation(&self, survivors: &[Genome], per_gen: usize, rng: &mut Rng) -> Vec<Genome> {
        let mut genomes: Vec<Genome> = Vec::new();
        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

        let push =
            |g: Genome, seen: &mut std::collections::BTreeSet<String>, out: &mut Vec<Genome>| {
                if seen.insert(g.describe()) {
                    out.push(g);
                }
            };

        // Carry survivors forward (elitism) + their mutations and pairwise crossovers. Survivors
        // keep their own (already-scored) risk geometry — that geometry is part of what won.
        for g in survivors {
            push(g.clone(), &mut seen, &mut genomes);
        }
        for g in survivors {
            if genomes.len() >= per_gen {
                break;
            }
            // Mutate predicates, then (unless locked) also perturb the risk geometry so risk and
            // conditions co-evolve.
            let mut m = genome_ops::mutate(g, &self.catalog, rng);
            if !self.cfg.lock_risk {
                m.risk = self.risk_space.mutate(&g.risk, rng);
            }
            push(m, &mut seen, &mut genomes);
        }
        if survivors.len() >= 2 {
            for pair in survivors.windows(2) {
                if genomes.len() >= per_gen {
                    break;
                }
                let mut c = genome_ops::crossover(&pair[0], &pair[1], rng);
                c.risk = if self.cfg.lock_risk {
                    self.cfg.risk
                } else {
                    self.risk_space.crossover(&pair[0].risk, &pair[1].risk, rng)
                };
                push(c, &mut seen, &mut genomes);
            }
        }

        // Top up with fresh seeds, each given a sampled (or locked) risk geometry.
        if genomes.len() < per_gen {
            let need = per_gen - genomes.len();
            let fresh = seed_population(
                &self.catalog,
                self.cfg.profile.horizon,
                need * 2,
                rng,
                self.cfg.max_predicates,
            );
            for g in fresh {
                if genomes.len() >= per_gen {
                    break;
                }
                push(self.assign_seed_risk(g, rng), &mut seen, &mut genomes);
            }
        }

        genomes.truncate(per_gen);
        genomes
    }

    /// Apply the survivor rule, update DB statuses, and return the surviving genomes for the
    /// next generation. Promotable strategies become `Promoted`; non-survivors become `Retired`.
    async fn select_survivors(&self, evaluated: &[Evaluated]) -> Result<Vec<Genome>> {
        let mut survivors = Vec::new();
        for ev in evaluated {
            if ev.promotable() {
                persist::update_status(self.store, ev.strategy.id, StrategyStatus::Promoted)
                    .await?;
                survivors.push(ev.strategy.genome.clone());
            } else if ev.survives() {
                // Keep as candidate (already persisted as candidate); carry forward to mutate.
                survivors.push(ev.strategy.genome.clone());
            } else if ev.score.is_some() {
                // Scored but failed the survivor rule -> retire.
                persist::update_status(self.store, ev.strategy.id, StrategyStatus::Retired).await?;
            }
            // Unscored (tiny dataset) strategies stay `candidate` but are not carried forward.
        }
        Ok(survivors)
    }
}

/// The outcome of an `evolve` run.
#[derive(Debug, Clone)]
pub struct EvolveOutcome {
    /// All evaluated members across all generations, sorted best-first by OOS score.
    pub evaluated: Vec<Evaluated>,
    /// The horizon profile the search ran under (for leaderboard labeling — P8).
    pub profile: HorizonProfile,
}

impl EvolveOutcome {
    /// The top `n` scored members (skips unscored/tiny-dataset members).
    pub fn leaderboard(&self, n: usize) -> Vec<&Evaluated> {
        self.evaluated
            .iter()
            .filter(|e| e.score.is_some())
            .take(n)
            .collect()
    }

    /// Count of promotable (gate-passing) members.
    pub fn n_promoted(&self) -> usize {
        self.evaluated.iter().filter(|e| e.promotable()).count()
    }

    /// Per-strategy id -> best score map (for downstream lookups).
    pub fn best_scores(&self) -> BTreeMap<StrategyId, &OosScore> {
        let mut m = BTreeMap::new();
        for e in &self.evaluated {
            if let Some(s) = &e.score {
                m.entry(e.strategy.id).or_insert(s);
            }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use se_core::{CmpOp, Genome, Horizon, Layer, Predicate, Side};
    use se_mlclient::ValidationResult;

    fn pred(layer: Layer) -> Predicate {
        Predicate {
            layer,
            feature_key: "k".into(),
            op: CmpOp::Gt,
            threshold: 0.0,
        }
    }

    // A gate-passing validation with a configurable acted-cohort size.
    fn passing_validation(n_acted: i64) -> ValidationResult {
        ValidationResult {
            dsr: 0.5,
            pbo: 0.1,
            oos_expectancy_cost_aware: 0.05,
            profit_factor: 1.4,
            cvar5: -0.4,
            mar: 0.5,
            regime_contrib: BTreeMap::new(),
            n_regimes_positive: 3,
            passed_gate: false, // gate is re-derived; metrics above pass independently
            precision_oos: 0.7,
            recall_oos: 0.5,
            act_threshold: 0.6,
            n_acted_oos: n_acted,
        }
    }

    fn evaluated(genome: Genome, validation: &ValidationResult) -> Evaluated {
        let strategy = Strategy::new(genome);
        let score = OosScore::from_validation(
            strategy.id,
            validation,
            se_validation::PromotionGate::evaluate(validation),
            80,
        );
        Evaluated {
            strategy,
            score: Some(score),
            n_entries: 80,
        }
    }

    #[test]
    fn regime_only_gate_passer_is_not_promotable() {
        // Regime-only conjunction: passes the hard gate but has no actionable entry trigger.
        let genome = Genome::new(
            Side::Long,
            Horizon::Swing,
            vec![pred(Layer::Regime), pred(Layer::Tradeability)],
        );
        let v = passing_validation(50); // plenty of acted trades, but no trigger/location
        let ev = evaluated(genome, &v);
        assert!(ev.score.as_ref().unwrap().passed_gate, "gate must pass");
        assert!(
            !ev.promotable(),
            "regime-only genome must NOT be promotable"
        );
        // It is still kept as a survivor (not retired) because the gate passed.
        assert!(ev.survives());
    }

    #[test]
    fn trigger_with_sufficient_acted_is_promotable() {
        let genome = Genome::new(
            Side::Long,
            Horizon::Swing,
            vec![pred(Layer::Regime), pred(Layer::Trigger)],
        );
        let v = passing_validation(MIN_ACTED_TO_PROMOTE as i64);
        let ev = evaluated(genome, &v);
        assert!(
            ev.promotable(),
            "trigger + sufficient n_acted must be promotable"
        );
    }

    #[test]
    fn trigger_with_too_few_acted_is_not_promotable() {
        // An actionable genome that nonetheless acted on too few OOS trades is held back.
        let genome = Genome::new(Side::Long, Horizon::Swing, vec![pred(Layer::Trigger)]);
        let v = passing_validation(MIN_ACTED_TO_PROMOTE as i64 - 1);
        let ev = evaluated(genome, &v);
        assert!(ev.score.as_ref().unwrap().passed_gate);
        assert!(
            !ev.promotable(),
            "too few acted OOS trades must block promotion"
        );
        assert!(ev.survives(), "but it is still kept as a survivor");
    }
}
