-- Precision / meta-labeling acting-threshold metrics on the OOS scoreboard.
--
-- The north-star OOS metric is now PRECISION: the fraction of OOS trades the meta-label
-- classifier ACTS ON (prob >= τ*) that turn out profitable. These columns surface that
-- precision, its paired recall, the acting threshold τ* (which the signal layer later reads),
-- and the acted-cohort size. They are NOT a ranking key — ranking stays gate-pass then
-- cost-aware OOS expectancy (already precision-conditioned upstream) — but they are persisted
-- for attribution and for the search's "enough acted OOS trades to promote" guardrail.
--
-- All columns are nullable with no backfill: rows scored before this migration legitimately
-- predate the metrics and stay NULL. win_rate remains banned and absent (see 0001_init.sql).

ALTER TABLE oos_scores ADD COLUMN IF NOT EXISTS precision_oos DOUBLE PRECISION;  -- OOS precision at τ*
ALTER TABLE oos_scores ADD COLUMN IF NOT EXISTS recall_oos    DOUBLE PRECISION;  -- OOS recall at τ*
ALTER TABLE oos_scores ADD COLUMN IF NOT EXISTS act_threshold DOUBLE PRECISION;  -- τ*, acting threshold in [0,1]
ALTER TABLE oos_scores ADD COLUMN IF NOT EXISTS n_acted       INT;               -- OOS trades acted on at τ*
