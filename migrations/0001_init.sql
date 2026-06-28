-- swing-X initial schema: bitemporal, provenance-tagged point-in-time store.
--
-- Leakage prevention lives in the data model: every feature row carries both
--   * decision_ts  — the bar the value is "for" (event time), and
--   * as_of        — when the value became knowable (knowledge time).
-- A read "as of decision bar T" must filter `as_of <= T`. The se-store PitQuery
-- layer bakes that predicate into every query so it cannot be bypassed.
--
-- Hypertables (bars, features_pit, trades_journal, monitor_events) are created
-- only when TimescaleDB is present; the schema also runs on plain PostgreSQL 16.

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ---------------------------------------------------------------------------
-- Price bars
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bars (
    ticker   TEXT             NOT NULL,
    cadence  TEXT             NOT NULL DEFAULT 'daily',
    ts       TIMESTAMPTZ      NOT NULL,
    open     DOUBLE PRECISION NOT NULL,
    high     DOUBLE PRECISION NOT NULL,
    low      DOUBLE PRECISION NOT NULL,
    close    DOUBLE PRECISION NOT NULL,
    volume   DOUBLE PRECISION NOT NULL,
    source   TEXT             NOT NULL,
    as_of    TIMESTAMPTZ      NOT NULL,
    PRIMARY KEY (ticker, cadence, ts)
);
CREATE INDEX IF NOT EXISTS bars_ticker_ts_idx ON bars (ticker, ts DESC);

-- ---------------------------------------------------------------------------
-- Point-in-time feature store — the spine of the system.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS features_pit (
    ticker      TEXT             NOT NULL,
    feature_key TEXT             NOT NULL,
    layer       TEXT             NOT NULL,
    decision_ts TIMESTAMPTZ      NOT NULL,          -- event time (the bar it's "for")
    as_of       TIMESTAMPTZ      NOT NULL,          -- knowledge time (when it was knowable)
    value       DOUBLE PRECISION NOT NULL,
    lead_time   TEXT             NOT NULL,
    source      TEXT             NOT NULL,
    valid_from  TIMESTAMPTZ      NOT NULL DEFAULT now(),
    valid_to    TIMESTAMPTZ      NOT NULL DEFAULT 'infinity',
    PRIMARY KEY (ticker, feature_key, decision_ts, as_of)
);
-- The index that makes PIT reads fast: per (ticker, feature) latest knowable version.
CREATE INDEX IF NOT EXISTS features_pit_read_idx
    ON features_pit (ticker, feature_key, decision_ts DESC, as_of DESC);
CREATE INDEX IF NOT EXISTS features_pit_layer_idx
    ON features_pit (ticker, layer, decision_ts DESC);

-- ---------------------------------------------------------------------------
-- Regime assignments (calibrated)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS regimes (
    ticker       TEXT        NOT NULL,
    decision_ts  TIMESTAMPTZ NOT NULL,
    as_of        TIMESTAMPTZ NOT NULL,
    regime_label TEXT        NOT NULL,
    prob_map     JSONB       NOT NULL DEFAULT '{}',
    model_id     UUID,
    PRIMARY KEY (ticker, decision_ts, as_of)
);

-- ---------------------------------------------------------------------------
-- Triple-barrier labels
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS labels (
    label_id     UUID             PRIMARY KEY,
    ticker       TEXT             NOT NULL,
    horizon      TEXT             NOT NULL,
    side         TEXT             NOT NULL CHECK (side IN ('long','short')),
    entry_ts     TIMESTAMPTZ      NOT NULL,
    t_barrier_ts TIMESTAMPTZ      NOT NULL,
    entry_px     DOUBLE PRECISION NOT NULL,
    target_px    DOUBLE PRECISION NOT NULL,
    stop_px      DOUBLE PRECISION NOT NULL,
    outcome      TEXT             CHECK (outcome IN ('target','stop','time','open')),
    exit_ts      TIMESTAMPTZ,
    exit_px      DOUBLE PRECISION,
    ret_r        DOUBLE PRECISION,                       -- realized return in R units
    meta_label   DOUBLE PRECISION,                       -- meta-labeling bet size [0,1]
    created_at   TIMESTAMPTZ      NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS labels_ticker_entry_idx ON labels (ticker, entry_ts);

-- ---------------------------------------------------------------------------
-- Strategy population
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS strategies (
    strategy_id UUID        PRIMARY KEY,
    horizon     TEXT        NOT NULL,
    genome      JSONB       NOT NULL,
    parent_id   UUID,
    status      TEXT        NOT NULL DEFAULT 'candidate'
                CHECK (status IN ('candidate','promoted','quarantined','demoted','retired')),
    generation  INT         NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS strategies_status_idx ON strategies (status);

-- ---------------------------------------------------------------------------
-- OOS scoreboard. NOTE: there is intentionally NO win_rate column — win_rate is
-- banned as a selection metric and must never appear in a ranking key.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS oos_scores (
    id                        BIGSERIAL PRIMARY KEY,
    strategy_id               UUID NOT NULL REFERENCES strategies(strategy_id) ON DELETE CASCADE,
    fold_spec                 JSONB NOT NULL,
    dsr                       DOUBLE PRECISION,         -- deflated Sharpe ratio
    pbo                       DOUBLE PRECISION,         -- probability of backtest overfit
    oos_expectancy_cost_aware DOUBLE PRECISION,
    profit_factor             DOUBLE PRECISION,
    cvar5                     DOUBLE PRECISION,
    mar                       DOUBLE PRECISION,         -- MAR / Calmar
    regime_contrib            JSONB NOT NULL DEFAULT '{}',
    n_regimes_positive        INT   NOT NULL DEFAULT 0,
    passed_gate               BOOLEAN NOT NULL DEFAULT FALSE,
    evaluated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS oos_scores_strategy_idx ON oos_scores (strategy_id, evaluated_at DESC);

-- ---------------------------------------------------------------------------
-- Surfaced signals
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS signals (
    signal_id         UUID             PRIMARY KEY,
    strategy_id       UUID             NOT NULL REFERENCES strategies(strategy_id),
    ticker            TEXT             NOT NULL,
    side              TEXT             NOT NULL CHECK (side IN ('long','short')),
    decision_ts       TIMESTAMPTZ      NOT NULL,
    horizon           TEXT             NOT NULL,
    entry             DOUBLE PRECISION NOT NULL,
    stop              DOUBLE PRECISION NOT NULL,
    target1           DOUBLE PRECISION NOT NULL,
    target2           DOUBLE PRECISION,
    rr1               DOUBLE PRECISION,
    rr2               DOUBLE PRECISION,
    conviction        DOUBLE PRECISION NOT NULL,        -- calibrated probability
    cohort_n          INT              NOT NULL,
    regime_desc       TEXT             NOT NULL,
    why               JSONB            NOT NULL DEFAULT '[]',
    invalidation      TEXT             NOT NULL,
    cohort_expectancy DOUBLE PRECISION,
    cvar5             DOUBLE PRECISION,
    lead_time         TEXT,
    payload_json      JSONB            NOT NULL,
    payload_human     TEXT             NOT NULL,
    created_at        TIMESTAMPTZ      NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS signals_ticker_ts_idx ON signals (ticker, decision_ts DESC);

-- ---------------------------------------------------------------------------
-- Paper/live trade journal
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trades_journal (
    trade_id    UUID             NOT NULL,
    signal_id   UUID,
    strategy_id UUID,
    ticker      TEXT             NOT NULL,
    side        TEXT             NOT NULL CHECK (side IN ('long','short')),
    mode        TEXT             NOT NULL DEFAULT 'paper' CHECK (mode IN ('paper','live')),
    entry_ts    TIMESTAMPTZ      NOT NULL,
    fill_px     DOUBLE PRECISION NOT NULL,
    fill_ts     TIMESTAMPTZ      NOT NULL,
    exit_ts     TIMESTAMPTZ,
    exit_px     DOUBLE PRECISION,
    pnl_r       DOUBLE PRECISION,
    cost_frac   DOUBLE PRECISION,
    attribution JSONB            NOT NULL DEFAULT '{}',
    PRIMARY KEY (trade_id, entry_ts)
);
CREATE INDEX IF NOT EXISTS trades_journal_strategy_idx ON trades_journal (strategy_id, entry_ts DESC);

-- ---------------------------------------------------------------------------
-- Forward-adaptation monitor events
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS monitor_events (
    id           BIGSERIAL,
    ts           TIMESTAMPTZ      NOT NULL,
    strategy_id  UUID,
    ticker       TEXT,
    detector     TEXT             NOT NULL,
    metric_value DOUBLE PRECISION,
    threshold    DOUBLE PRECISION,
    action_taken TEXT             NOT NULL,
    detail       JSONB            NOT NULL DEFAULT '{}',
    PRIMARY KEY (id, ts)
);
CREATE INDEX IF NOT EXISTS monitor_events_strategy_idx ON monitor_events (strategy_id, ts DESC);

-- ---------------------------------------------------------------------------
-- Model registry
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_registry (
    model_id     UUID        PRIMARY KEY,
    kind         TEXT        NOT NULL,
    version      TEXT        NOT NULL,
    params       JSONB       NOT NULL DEFAULT '{}',
    artifact_uri TEXT,
    train_window JSONB,
    metrics      JSONB       NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Weekly self-changelog (what decayed / retired / adapted)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS changelog (
    week       DATE        PRIMARY KEY,
    decayed    JSONB       NOT NULL DEFAULT '[]',
    retired    JSONB       NOT NULL DEFAULT '[]',
    adapted    JSONB       NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Promote selected tables to Timescale hypertables when the extension exists.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('bars',           by_range('ts'),          if_not_exists => TRUE, migrate_data => TRUE);
        PERFORM create_hypertable('features_pit',   by_range('decision_ts'), if_not_exists => TRUE, migrate_data => TRUE);
        PERFORM create_hypertable('trades_journal', by_range('entry_ts'),    if_not_exists => TRUE, migrate_data => TRUE);
        PERFORM create_hypertable('monitor_events', by_range('ts'),          if_not_exists => TRUE, migrate_data => TRUE);
    END IF;
END
$$;
