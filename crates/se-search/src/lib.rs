//! `se-search` (P5) — genome search / mutation on the **OOS scoreboard**.
//!
//! The pipeline, end to end:
//!
//! 1. [`feature_matrix::build_window`] — materialize per-bar PIT-safe feature maps + bars + ATR
//!    for a ticker over a window (features computed on the fly through the leakage-safe feature
//!    modules, so the backtest never depends on what happens to be persisted).
//! 2. [`seed`] — build the catalog of OBSERVED feature keys + empirical quantiles, then seed an
//!    initial population of [`se_core::Genome`]s with a deterministic, per-generation RNG.
//! 3. [`backtest::backtest`] — walk bars; where a genome fires and the regime is tradeable, take
//!    an entry (min-hold cooldown), label it with the triple-barrier labeler, collect
//!    `LabeledEntry`s, and [`backtest::assemble`] them into a `DatasetRow` dataset.
//! 4. [`score::score_oos`] — run the dataset through [`se_validation::ValidationHarness`] (purged
//!    and embargoed CPCV on the worker) and wrap the result in an [`score::OosScore`]. That score
//!    is the only thing the search ranks on — no in-sample number enters the ranking key.
//! 5. [`genome_ops`] — mutate / crossover survivors.
//! 6. [`population::PopulationManager::evolve`] — SEARCH -> FIT(implicit) -> SCORE(OOS) -> KEEP /
//!    MUTATE / KILL, persisting strategies + OOS scores to the DB every generation.
//!
//! Horizon generalization (P8): every step takes a [`se_core::HorizonProfile`]; no swing
//! constant is hardcoded, so the same loop runs under any profile (swing, day, ...).

pub mod backtest;
pub mod feature_matrix;
pub mod genome_ops;
pub mod persist;
pub mod population;
pub mod risk_search;
pub mod rng;
pub mod score;
pub mod seed;

pub use backtest::{assemble, backtest, BacktestResult};
pub use feature_matrix::{build_window, dotted_to_column, BarPoint, FeatureWindow};
pub use genome_ops::{crossover, mutate};
pub use persist::{
    insert_oos_score, latest_oos_score, load_promoted, load_strategy, update_status,
    upsert_strategy, StoredOosScore,
};
pub use population::{
    Evaluated, EvolveOutcome, PopulationManager, SearchConfig, DEFAULT_SEARCH_SEED,
};
pub use risk_search::RiskSpace;
pub use rng::Rng;
pub use score::{score_oos, OosScore, ScoreConfig, MIN_ENTRIES_TO_VALIDATE};
pub use seed::{layer_of_key, random_genome, seed_population, FeatureCatalog, FeatureStat};
