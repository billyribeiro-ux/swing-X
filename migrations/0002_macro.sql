-- swing-X macro store: market-wide (NOT per-ticker) point-in-time series.
--
-- The regime layer conditions on the vol complex, rates, cross-asset, credit and
-- liquidity. These are properties of the *market*, not of any single ticker, so
-- they live in their own table keyed by `(series, ts, as_of)` rather than carrying
-- a ticker column.
--
-- Leakage rule (same as features_pit): a value read for a decision at bar T must
-- satisfy BOTH `ts <= T` (the observation is for a date on or before T) AND
-- `as_of <= T` (the value was actually knowable by T). FRED publishes with a lag,
-- so `as_of` is frequently later than `ts`; the bitemporal key preserves every
-- vintage so a backtest at T never sees a value released after T.

CREATE TABLE IF NOT EXISTS macro_series_pit (
    series    TEXT             NOT NULL,
    ts        TIMESTAMPTZ      NOT NULL,          -- reference date (event time)
    as_of     TIMESTAMPTZ      NOT NULL,          -- when it became knowable (knowledge time)
    value     DOUBLE PRECISION NOT NULL,
    lead_time TEXT             NOT NULL,
    source    TEXT             NOT NULL,
    PRIMARY KEY (series, ts, as_of)
);
-- The index that makes PIT reads fast: per series, latest knowable observation.
CREATE INDEX IF NOT EXISTS macro_series_pit_read_idx
    ON macro_series_pit (series, ts DESC, as_of DESC);

-- ---------------------------------------------------------------------------
-- Promote to a Timescale hypertable on `ts` when the extension is present
-- (mirrors the guarded pattern in 0001_init.sql so the schema also runs on
-- plain PostgreSQL 16).
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('macro_series_pit', by_range('ts'), if_not_exists => TRUE, migrate_data => TRUE);
    END IF;
END
$$;
