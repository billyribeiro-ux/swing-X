-- Forward-in-time holdout durability metrics for the OOS scoreboard. Unlike the CPCV precision
-- (which shuffles folds across the whole timeline and can be flattered by a favorable regime),
-- these are measured on a STRICT time-ordered forward holdout: the primary model + acting
-- threshold are fit on the earliest 70% of rows and precision/expectancy are measured on the
-- latest 30% (never shuffled). This is the metric that separates a durable forward edge from
-- bull-window regime-fitting. Nullable (old rows stay NULL); no backfill. win_rate stays banned.

ALTER TABLE oos_scores ADD COLUMN IF NOT EXISTS precision_forward  DOUBLE PRECISION;  -- P(profit|acted) on the forward holdout
ALTER TABLE oos_scores ADD COLUMN IF NOT EXISTS expectancy_forward DOUBLE PRECISION;  -- cost-aware expectancy (R) on the forward holdout
ALTER TABLE oos_scores ADD COLUMN IF NOT EXISTS n_forward          INT;               -- acted-trade count in the forward holdout
