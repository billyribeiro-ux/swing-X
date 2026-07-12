-- The once-only TEST-ERA scoreboard: the selection-bias meter's persistence.
--
-- A locked out-of-time era (SE_TEST_FROM/SE_TEST_TO) is firewalled out of ALL search/nightly
-- training (see se-search's firewall_test_era). `se test-era-score` evaluates each PROMOTED
-- strategy EXACTLY ONCE on that era (fit strictly pre-era, measure in-era) and records the
-- result here. The PRIMARY KEY (strategy_id, test_from, test_to) is the once-only guarantee:
-- re-scoring the same strategy on the same era is refused, so the era can never be iterated
-- against. REPORT-ONLY by design — nothing reads this table for ranking, the promotion gate,
-- survivor selection, or nightly; its sole purpose is to measure the optimism gap
-- (precision_oos − precision_test). win_rate remains banned.

CREATE TABLE IF NOT EXISTS test_era_scores (
    strategy_id     UUID             NOT NULL REFERENCES strategies(strategy_id) ON DELETE CASCADE,
    test_from       DATE             NOT NULL,
    test_to         DATE             NOT NULL,
    precision_test  DOUBLE PRECISION,                 -- P(net profit | acted) inside the era
    expectancy_test DOUBLE PRECISION,                 -- cost-aware expectancy (R) inside the era
    n_test          INT,                              -- acted-trade count inside the era
    scored_at       TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (strategy_id, test_from, test_to)
);
