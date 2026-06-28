-- Allow `backtest` as a trade mode so a promoted strategy's per-trade backtest
-- outcomes (winners/losers) can be journaled distinctly from live `paper`/`live`
-- trades. Keeping them separate matters: a backtest result is NOT a live result,
-- and the system must never conflate the two.
ALTER TABLE trades_journal DROP CONSTRAINT IF EXISTS trades_journal_mode_check;
ALTER TABLE trades_journal
    ADD CONSTRAINT trades_journal_mode_check CHECK (mode IN ('paper', 'live', 'backtest'));
