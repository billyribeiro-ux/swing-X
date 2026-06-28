-- Scanner dimension: the engine now runs two parallel populations — the ETF universe and an
-- individual-equity universe. Tag strategies/signals/trades with their scanner so the two never
-- mix on the scoreboard, in the journal, or in promotion. Existing rows default to 'etf', so the
-- ETF scanner is unchanged.

ALTER TABLE strategies
    ADD COLUMN IF NOT EXISTS scanner TEXT NOT NULL DEFAULT 'etf'
        CHECK (scanner IN ('etf', 'equity'));

ALTER TABLE signals
    ADD COLUMN IF NOT EXISTS scanner TEXT NOT NULL DEFAULT 'etf'
        CHECK (scanner IN ('etf', 'equity'));

ALTER TABLE trades_journal
    ADD COLUMN IF NOT EXISTS scanner TEXT NOT NULL DEFAULT 'etf'
        CHECK (scanner IN ('etf', 'equity'));

CREATE INDEX IF NOT EXISTS strategies_scanner_status_idx ON strategies (scanner, status);
CREATE INDEX IF NOT EXISTS signals_scanner_ts_idx ON signals (scanner, decision_ts DESC);
CREATE INDEX IF NOT EXISTS trades_journal_scanner_idx ON trades_journal (scanner, entry_ts DESC);
