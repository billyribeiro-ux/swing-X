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
    genome_has_actionable_predicate, genome_signature, score_oos, wilson_lower_bound, OosScore,
    ScoreConfig, MIN_ACTED_TO_PROMOTE, MIN_ENTRIES_TO_VALIDATE, MIN_PROMOTE_PRECISION_LB,
    WILSON_Z_95,
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
                    // Optimizer's-curse guard: the (net-of-cost) precision's Wilson lower bound —
                    // not the point estimate — must clear the floor, so a small-n high-precision
                    // fluke cannot be promoted on sampling luck.
                    && wilson_lower_bound(s.precision_oos, s.n_acted_oos, WILSON_Z_95)
                        >= MIN_PROMOTE_PRECISION_LB
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
    /// Exploration breadth knob: how many DISTINCT candidate genomes to generate per generation
    /// before scoring, as a multiple of `per_gen`. The generation builds this larger, more diverse
    /// pool (extra survivor mutations + a bigger fresh-seed draw) and then keeps the first `per_gen`
    /// distinct ones — so widening this only changes which candidates are explored, never how many
    /// are scored (`per_gen` is unchanged) and never the survivor/ranking/gate logic.
    ///
    /// Defaults to [`DEFAULT_OFFSPRING_POOL_MULT`]. Backward-compatible: callers that construct via
    /// [`SearchConfig::new`] get the default automatically; values `<= 1.0` reproduce the legacy
    /// (per_gen-sized) pool.
    pub offspring_pool_mult: f64,
    /// A LOCKED out-of-time TEST ERA `[from, to]` (inclusive, session-close UTC) that is
    /// FIREWALLED out of this search's training dataset: any labeled entry whose decision bar
    /// (`ts`) OR whose label-window end (`t1`) falls inside it is purged before scoring
    /// (see [`PopulationManager::build_dataset`] / [`firewall_test_era`]). Purging by `t1` is what
    /// makes it leak-safe — a label entered BEFORE the era but resolving inside it would otherwise
    /// leak reserved-era outcome into training.
    ///
    /// `None` (the default via [`SearchConfig::new`]) => no reservation, behavior identical to
    /// before. This is REPORT-ONLY infrastructure: it only ever REMOVES rows from the dataset; it
    /// NEVER feeds rank_key/gate/survivor selection/nightly scoring or any promotion decision.
    pub test_era: Option<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>,
}

/// Default deterministic search seed (never derived from the clock — see [`crate::rng`]).
pub const DEFAULT_SEARCH_SEED: u64 = 0x0005_EED0_F5EA_C401;

/// Default exploration-pool multiple (see [`SearchConfig::offspring_pool_mult`]). Modestly larger
/// than 1 so each generation considers a richer, more diverse candidate pool before keeping the
/// best `per_gen` distinct ones — better exploration at no extra scoring cost.
pub const DEFAULT_OFFSPRING_POOL_MULT: f64 = 2.0;

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
            offspring_pool_mult: DEFAULT_OFFSPRING_POOL_MULT,
            // No out-of-time reservation by default: the firewall is a no-op until the operator
            // sets SE_TEST_FROM/SE_TEST_TO, so existing searches are unchanged.
            test_era: None,
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
    ///
    /// OUT-OF-TIME TEST-ERA FIREWALL: when `cfg.test_era` is set, every assembled row whose
    /// decision bar OR whose label end `t1` falls inside the reserved era is purged here
    /// (see [`firewall_test_era`]) so NO row that touches the reservation can reach OOS scoring,
    /// ranking, the gate, survivor selection, or nightly. When it is `None` this is a no-op and
    /// the dataset is exactly the legacy dataset.
    fn build_dataset(&self, genome: &Genome) -> (Vec<se_mlclient::DatasetRow>, usize) {
        let mut all = Vec::new();
        let mut purged = 0usize;
        for w in &self.windows {
            let res = backtest(genome, w, &self.cfg.profile);
            let rows = assemble(&res);
            let (kept, n_purged) = firewall_test_era(rows, self.cfg.test_era);
            purged += n_purged;
            all.extend(kept);
        }
        all.sort_by(|a, b| a.ts.cmp(&b.ts));
        let n = all.len();
        if purged > 0 {
            tracing::debug!(
                purged,
                kept = n,
                "test-era firewall purged labeled rows touching the reserved out-of-time era"
            );
        }
        (all, n)
    }

    /// Assemble the labeled dataset for one genome (rows + entry count) through the exact same
    /// machinery the search uses (PIT feature windows -> backtest -> assemble -> test-era
    /// firewall). Public so the once-only test-era scorer (`se test-era-score`) evaluates
    /// promoted genomes with identical data semantics.
    pub fn dataset_for(&self, genome: &Genome) -> (Vec<se_mlclient::DatasetRow>, usize) {
        self.build_dataset(genome)
    }

    /// Evaluate one genome end-to-end: backtest -> assemble -> OOS score. Tiny datasets return
    /// `Ok(Evaluated{score: None})` (logged, skipped) rather than erroring.
    ///
    /// `n_search_trials` is the run-cumulative count of DISTINCT genomes evaluated so far
    /// (including this one); it flows into the worker's DSR deflation so significance reflects
    /// the search's true multiple-comparisons burden.
    async fn evaluate_genome(&self, strategy: Strategy, n_search_trials: u32) -> Result<Evaluated> {
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
        let score_cfg = ScoreConfig {
            n_search_trials: n_search_trials.max(1),
            ..self.cfg.score
        };
        let score = match score_oos(
            self.harness,
            strategy.id,
            &rows,
            &self.cfg.profile,
            score_cfg,
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
        // Signatures of every DISTINCT genome we have already created/scored/persisted, across all
        // generations. Deduplicating candidates against this set (in `next_generation`) is what
        // stops the population from inflating with identical genomes — each of which would
        // otherwise become its own strategy row, its own OOS score, and its own promotion. Because
        // dedup happens at candidate-creation time, PROMOTION cannot produce two promoted
        // strategies with the same signature either: a genome equal to one already created in a
        // prior generation is never re-created, so it is never re-scored or re-promoted.
        let mut seen_signatures: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        // A STABLE strategy identity per canonical signature. Elite survivors are carried forward
        // every generation (that is deliberate — the elitism/re-scoring path), so without this a
        // single surviving genome would be re-wrapped in a fresh `Strategy` (new `StrategyId`) each
        // generation and `upsert_strategy` would INSERT a new row + a new promotion for the very
        // same genome — the exact inflation being fixed. Reusing the signature's first-assigned id
        // makes those re-persists UPDATE the one row (same id) instead. Fresh genomes get a new id.
        let mut id_for_signature: std::collections::BTreeMap<String, StrategyId> =
            std::collections::BTreeMap::new();

        for gen in 0..generations {
            let mut rng = Rng::seeded(self.cfg.base_seed, gen);

            // Build this generation's genomes: survivors + their mutations/crossovers, topped up
            // with fresh seeds to `per_gen`. Deduplicated by canonical signature within the
            // generation AND against every genome already created in a prior generation.
            let genomes = self.next_generation(&survivors, per_gen, &mut seen_signatures, &mut rng);

            tracing::info!(generation = gen, n = genomes.len(), "evaluating generation");

            let mut evaluated: Vec<Evaluated> = Vec::new();
            for genome in genomes {
                // Reuse the stable id already assigned to this genome's signature (survivors, or an
                // equal genome from a prior gen) so re-persist UPDATES that row; otherwise mint a
                // fresh id and remember it. Never create two strategy rows for one signature.
                let sig = genome_signature(&genome);
                // `StrategyId::new()` mints a fresh random UUID; it is NOT the type's default
                // (a nil UUID), so clippy's `or_default()` suggestion would collapse every unseen
                // signature onto the same nil id — exactly the duplication we are removing.
                #[allow(clippy::unwrap_or_default)]
                let id = *id_for_signature.entry(sig).or_insert_with(StrategyId::new);
                let mut strategy = Strategy::new(genome);
                strategy.id = id;
                strategy.generation = gen;
                // The signature map's size IS the run-cumulative distinct-genome count (this
                // genome included) — the search's true multiple-comparisons burden, fed into
                // the worker's DSR deflation.
                let n_search_trials = id_for_signature.len() as u32;
                let ev = self.evaluate_genome(strategy, n_search_trials).await?;
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
    ///
    /// Exploration breadth: candidates are drawn from a pool of up to `per_gen *
    /// offspring_pool_mult` distinct genomes (elite survivors first, then several mutations per
    /// survivor, then crossovers, then a larger fresh-seed draw) and the first `per_gen` distinct
    /// ones are kept. Widening `offspring_pool_mult` changes only which candidates fill the
    /// non-elite slots — the number scored stays `per_gen`, and survivors are always carried
    /// forward first (unchanged elitism). This never touches the survivor/ranking/gate logic.
    ///
    /// DEDUPLICATION: candidates are keyed by [`genome_signature`] — an order-independent canonical
    /// signature — so two genomes that would fire and manage risk identically are treated as one.
    /// A candidate is dropped if its signature was already produced (a) earlier in this same
    /// generation, or (b) in ANY prior generation (tracked in `seen`, which persists across the
    /// whole `evolve` run). This is what stops the population inflating with duplicate genomes that
    /// would each become a separate strategy/score/promotion. Carried-forward survivors are the one
    /// exception: they are always kept (elitism) even though their signature is already in `seen`
    /// from the generation that created them — dropping them would break elitism, and they are not
    /// re-persisted as new strategies here.
    fn next_generation(
        &self,
        survivors: &[Genome],
        per_gen: usize,
        global_seen: &mut std::collections::BTreeSet<String>,
        rng: &mut Rng,
    ) -> Vec<Genome> {
        let mut genomes: Vec<Genome> = Vec::new();
        let mut dropped: usize = 0;

        // Local dedup set for building this generation's pool: seeded with every signature seen in
        // a PRIOR generation, so a candidate equal to any already-created genome is dropped here
        // (never re-scored, never re-persisted, never re-promoted). We commit only the KEPT genomes'
        // signatures back to `global_seen` after truncation, so genomes that get truncated out of
        // the over-sized pool remain available to explore in a later generation.
        let mut seen = global_seen.clone();

        // The richer candidate pool we draw from before truncating to `per_gen`. `>= per_gen` so a
        // multiple of 1 (or less) reproduces the legacy per_gen-sized behavior.
        let pool_target =
            ((per_gen as f64) * self.cfg.offspring_pool_mult.max(1.0)).ceil() as usize;
        let pool_target = pool_target.max(per_gen);

        // Push a NEW candidate: keep it only if its canonical signature is unseen (within this
        // generation and across all prior ones). Later duplicates are dropped (counted for the
        // debug log). Keeps the first occurrence. Delegates the signature rule to `admit_unique`.
        let push = |g: Genome,
                    seen: &mut std::collections::BTreeSet<String>,
                    dropped: &mut usize,
                    out: &mut Vec<Genome>| {
            if !admit_unique(g, seen, out) {
                *dropped += 1;
            }
        };

        // Carry survivors forward (elitism) + their mutations and pairwise crossovers. Survivors
        // keep their own (already-scored) risk geometry — that geometry is part of what won. Their
        // signatures are already in `seen` (from the generation that created them), so we add them
        // unconditionally rather than through `push`, and (re)mark the signature so the rest of
        // this generation still dedups against them.
        for g in survivors {
            seen.insert(genome_signature(g));
            genomes.push(g.clone());
        }
        // One guaranteed mutation per survivor first (preserves prior behavior for the head of the
        // list), capped at `per_gen` so elitism + first mutations are unchanged from before.
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
            push(m, &mut seen, &mut dropped, &mut genomes);
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
                push(c, &mut seen, &mut dropped, &mut genomes);
            }
        }

        // EXPLORATION: extra mutations of survivors to enrich the pool beyond `per_gen` (only used
        // to widen diversity before truncation; does not displace the elite head above). We fill
        // the extra mutations up to the MIDPOINT between `per_gen` and `pool_target`, reserving the
        // rest of the pool for fresh seeds so both intensification (mutations) and diversification
        // (fresh seeds) stay represented.
        let mut_target = per_gen + (pool_target - per_gen) / 2;
        if !survivors.is_empty() && mut_target > genomes.len() {
            let mut guard = 0usize;
            let max_attempts = pool_target.saturating_mul(4).max(8);
            while genomes.len() < mut_target && guard < max_attempts {
                guard += 1;
                let parent = &survivors[rng.below(survivors.len())];
                let mut m = genome_ops::mutate(parent, &self.catalog, rng);
                if !self.cfg.lock_risk {
                    m.risk = self.risk_space.mutate(&parent.risk, rng);
                }
                push(m, &mut seen, &mut dropped, &mut genomes);
            }
        }

        // Top up with fresh seeds, each given a sampled (or locked) risk geometry. Draw enough to
        // fill the richer pool, then truncate to `per_gen` below.
        if genomes.len() < pool_target {
            let need = pool_target - genomes.len();
            let fresh = seed_population(
                &self.catalog,
                self.cfg.profile.horizon,
                need * 2,
                rng,
                self.cfg.max_predicates,
            );
            for g in fresh {
                if genomes.len() >= pool_target {
                    break;
                }
                push(
                    self.assign_seed_risk(g, rng),
                    &mut seen,
                    &mut dropped,
                    &mut genomes,
                );
            }
        }

        genomes.truncate(per_gen);

        // Commit the KEPT genomes' signatures to the run-wide `global_seen` so no later generation
        // re-creates (and thus re-scores/re-persists/re-promotes) an equal genome. Only the kept
        // ones are committed — truncated pool-tail candidates stay explorable later.
        for g in &genomes {
            global_seen.insert(genome_signature(g));
        }
        if dropped > 0 {
            tracing::debug!(
                dropped,
                kept = genomes.len(),
                "deduplicated candidate genomes by canonical signature"
            );
        }

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

/// Apply the OUT-OF-TIME TEST-ERA FIREWALL to a genome's assembled dataset rows.
///
/// When `era` is `Some((from, to))`, drop every row whose **decision/entry bar** (`ts`) OR whose
/// **label-window** `[ts, t1]` OVERLAPS the inclusive reserved window `[from, to]` — the interval
/// predicate `ts <= to AND t1 >= from ⇒ exclude`. Overlap (not just endpoint membership) is
/// essential: a label entered BEFORE `from` but resolving (`t1`) inside — or straddling — the era
/// would otherwise leak reserved-era outcome into training. When `era` is `None`, the rows pass
/// through untouched (legacy behavior).
///
/// Returns `(kept_rows, n_purged)`. This is a pure filter used only to build the training dataset;
/// it NEVER influences ranking, the gate, survivor selection, promotion, or nightly scoring — it
/// only removes rows so the reservation stays a never-touched selection-bias meter.
fn firewall_test_era(
    rows: Vec<se_mlclient::DatasetRow>,
    era: Option<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>,
) -> (Vec<se_mlclient::DatasetRow>, usize) {
    let Some((from, to)) = era else {
        return (rows, 0);
    };
    let mut kept = Vec::with_capacity(rows.len());
    let mut purged = 0usize;
    for row in rows {
        // Exclude iff the label's holding span [ts, t1] OVERLAPS the reserved era [from, to]:
        // `ts <= to && t1 >= from`. This interval-overlap test is stricter than endpoint
        // membership — it also catches a label that STRADDLES the whole era (entry before `from`,
        // resolution after `to`), which `decision_ts ∈ era OR t1 ∈ era` would miss. `ts` is the
        // decision/entry bar; `t1` is the barrier-end (label resolution).
        let overlaps_era = row.ts <= to && row.t1 >= from;
        if overlaps_era {
            purged += 1;
        } else {
            kept.push(row);
        }
    }
    (kept, purged)
}

/// Attempt to admit one candidate genome into a deduplicated set. Returns `true` (and pushes `g`
/// onto `out`) iff its canonical [`genome_signature`] was not already in `seen`; otherwise returns
/// `false` and leaves `out` untouched. `seen` accumulates every admitted (and pre-seeded)
/// signature. Keeps the FIRST occurrence of each signature; every later equal genome is rejected.
///
/// This is the single dedup primitive used when assembling a generation's candidate pool
/// (see [`PopulationManager::next_generation`]): pre-seed `seen` with prior-generation signatures
/// to also reject genomes equal to any already created earlier in the run.
fn admit_unique(
    g: Genome,
    seen: &mut std::collections::BTreeSet<String>,
    out: &mut Vec<Genome>,
) -> bool {
    if seen.insert(genome_signature(&g)) {
        out.push(g);
        true
    } else {
        false
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
            precision_forward: 0.6,
            expectancy_forward: 0.05,
            n_forward: n_acted,
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
    fn search_config_default_sets_offspring_pool_mult() {
        // Backward-compatible default: callers via `new()` get the exploration pool multiple
        // without having to set it, and it is > 1 so each generation explores a richer pool.
        let cfg = SearchConfig::new(se_core::HorizonProfile::swing(), vec![Ticker::SPY]);
        assert_eq!(cfg.offspring_pool_mult, DEFAULT_OFFSPRING_POOL_MULT);
        assert!(cfg.offspring_pool_mult >= 1.0);
        // Backward-compatible: no out-of-time reservation unless the operator sets one, so the
        // firewall is a no-op for callers that construct via `new()`.
        assert!(cfg.test_era.is_none());
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
        // A LARGE acted cohort so the 0.70 precision's Wilson lower bound (~0.58) clears the
        // MIN_PROMOTE_PRECISION_LB floor: a genuine, sampling-robust edge is promotable.
        let v = passing_validation(60);
        let ev = evaluated(genome, &v);
        assert!(
            ev.promotable(),
            "trigger + large acted cohort with a robust precision LB must be promotable"
        );
    }

    #[test]
    fn small_n_high_precision_fluke_is_not_promotable() {
        // The optimizer's-curse case: precision 0.70 but only n=8 acted. It clears the hard gate
        // and the MIN_ACTED floor, yet the Wilson lower bound (~0.37) is below the LB floor — a
        // small-sample fluke the search would otherwise promote. Must be held back (but survive).
        let genome = Genome::new(Side::Long, Horizon::Swing, vec![pred(Layer::Trigger)]);
        let v = passing_validation(MIN_ACTED_TO_PROMOTE as i64); // 8 acted, precision 0.70
        let ev = evaluated(genome, &v);
        assert!(ev.score.as_ref().unwrap().passed_gate);
        assert!(
            !ev.promotable(),
            "a high-precision but tiny-n cohort must fail the Wilson lower-bound gate"
        );
        assert!(
            ev.survives(),
            "but it is still kept as a survivor to mutate"
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

    // Two predicates over distinct features/layers, used to build order-permuted duplicates.
    fn two_pred_genome(order_swapped: bool) -> Genome {
        let p_trig = Predicate {
            layer: Layer::Trigger,
            feature_key: "trigger.rsi14".into(),
            op: CmpOp::Gt,
            threshold: 55.0,
        };
        let p_reg = Predicate {
            layer: Layer::Regime,
            feature_key: "regime.adx".into(),
            op: CmpOp::Lt,
            threshold: 20.0,
        };
        let preds = if order_swapped {
            vec![p_reg, p_trig]
        } else {
            vec![p_trig, p_reg]
        };
        Genome::new(Side::Long, Horizon::Swing, preds)
    }

    #[test]
    fn dedup_drops_signature_duplicates_keeping_first() {
        // A candidate pool built to CONTAIN duplicates: the same conjunction in two predicate
        // orders, plus a sub-grid-noise threshold twin, plus a genuinely different genome.
        let mut noisy = two_pred_genome(false);
        // Nudge a threshold below the 4-decimal signature grid: still the same firing genome.
        noisy.predicates[0].threshold = 55.000_001;

        let genuinely_different =
            Genome::new(Side::Short, Horizon::Swing, vec![pred(Layer::Trigger)]);

        let candidates = vec![
            two_pred_genome(false),      // first occurrence — kept
            two_pred_genome(true),       // same set, swapped order — dropped
            noisy,                       // same set, sub-grid threshold noise — dropped
            genuinely_different.clone(), // distinct — kept
            genuinely_different,         // exact repeat — dropped
        ];

        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut kept: Vec<Genome> = Vec::new();
        let mut dropped = 0usize;
        for g in candidates {
            if !admit_unique(g, &mut seen, &mut kept) {
                dropped += 1;
            }
        }

        assert_eq!(kept.len(), 2, "only the two distinct genomes survive dedup");
        assert_eq!(dropped, 3, "the three duplicates are dropped");

        // The invariant the fix guarantees: no two KEPT genomes share a signature.
        let sigs: std::collections::BTreeSet<String> = kept.iter().map(genome_signature).collect();
        assert_eq!(
            sigs.len(),
            kept.len(),
            "deduped candidate set must contain no two equal signatures"
        );
    }

    #[test]
    fn dedup_rejects_candidate_equal_to_a_prior_generation_genome() {
        // Simulate cross-generation dedup: `seen` is pre-seeded with a genome created in an
        // earlier generation. A later candidate equal to it (even with permuted predicate order)
        // must be rejected so it is never re-created / re-scored / re-promoted.
        let prior = two_pred_genome(false);
        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        seen.insert(genome_signature(&prior));

        let mut kept: Vec<Genome> = Vec::new();
        // Equal genome, predicates in swapped order — must be rejected against the prior gen.
        let admitted = admit_unique(two_pred_genome(true), &mut seen, &mut kept);
        assert!(
            !admitted && kept.is_empty(),
            "a genome equal to a prior-generation one must be rejected at candidate time"
        );

        // A genuinely new genome is still admitted.
        let fresh = Genome::new(Side::Short, Horizon::Day, vec![pred(Layer::Location)]);
        assert!(admit_unique(fresh, &mut seen, &mut kept));
        assert_eq!(kept.len(), 1);
    }

    // ---- out-of-time TEST-ERA firewall ---------------------------------------

    use chrono::{DateTime, TimeZone, Utc};
    use se_mlclient::DatasetRow;

    /// A dataset row with an explicit decision bar (`ts`) and label end (`t1`), days offset from
    /// a reference instant.
    fn era_row(entry: DateTime<Utc>, t1: DateTime<Utc>) -> DatasetRow {
        DatasetRow {
            ts: entry,
            t1,
            label: 0.0,
            regime: None,
            features: std::collections::BTreeMap::new(),
        }
    }

    fn day(n: i64) -> chrono::Duration {
        chrono::Duration::days(n)
    }

    #[test]
    fn test_era_firewall_purges_decision_and_t1_membership() {
        // Reserved out-of-time era [from, to].
        let from = Utc.with_ymd_and_hms(2025, 1, 1, 21, 0, 0).unwrap();
        let to = Utc.with_ymd_and_hms(2025, 3, 31, 21, 0, 0).unwrap();
        let era = Some((from, to));

        // (a) fully BEFORE the era (both decision and t1 before `from`) -> KEPT.
        let clean_before = era_row(from - day(100), from - day(90));
        // (b) PLANTED future-membership: the decision bar is BEFORE `from`, but the label RESOLVES
        //     (t1) inside the era. This is precisely the leak the t1-purge exists to catch and it
        //     MUST be excluded (a plain entry-ts firewall would wrongly keep it).
        let resolves_in_era = era_row(from - day(2), from + day(5));
        // (c) decision bar INSIDE the era -> PURGED (entry-membership).
        let decision_in_era = era_row(from + day(10), to + day(2));
        // (d) fully AFTER the era -> KEPT.
        let clean_after = era_row(to + day(10), to + day(15));

        let rows = vec![
            clean_before.clone(),
            resolves_in_era,
            decision_in_era,
            clean_after.clone(),
        ];
        let (kept, purged) = firewall_test_era(rows, era);

        assert_eq!(
            purged, 2,
            "the t1-in-era (planted) row and the decision-in-era row must both be purged"
        );
        assert_eq!(
            kept.len(),
            2,
            "only the fully-before and fully-after rows survive"
        );
        assert!(
            kept.iter().any(|r| r.ts == clean_before.ts),
            "the clean pre-era row is retained"
        );
        assert!(
            kept.iter().any(|r| r.ts == clean_after.ts),
            "the clean post-era row is retained"
        );
        // The invariant: no surviving row's decision bar OR label end lies inside the reservation.
        for r in &kept {
            assert!(
                !(r.ts >= from && r.ts <= to),
                "no kept row's decision bar may fall inside the reserved era"
            );
            assert!(
                !(r.t1 >= from && r.t1 <= to),
                "no kept row's label end t1 may fall inside the reserved era"
            );
        }
    }

    #[test]
    fn test_era_firewall_is_noop_when_unset() {
        // With NO reservation, the firewall must pass every row through unchanged — the
        // backward-compatible default that keeps existing searches identical.
        let base = Utc.with_ymd_and_hms(2025, 1, 1, 21, 0, 0).unwrap();
        let rows = vec![
            era_row(base, base + day(5)),
            era_row(base + day(10), base + day(15)),
        ];
        let n = rows.len();
        let (kept, purged) = firewall_test_era(rows, None);
        assert_eq!(purged, 0, "no reservation => nothing is ever purged");
        assert_eq!(kept.len(), n, "all rows pass through untouched");
    }
}
