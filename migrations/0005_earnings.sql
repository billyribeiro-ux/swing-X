-- Earnings calendar for the equity scanner's blackout guard: never open a new position into a
-- release inside the holding window (single-name gap risk the stop can't honor). ETFs have no
-- rows here, so the guard is a no-op for the ETF scanner.

CREATE TABLE IF NOT EXISTS earnings (
    ticker TEXT NOT NULL,
    date   DATE NOT NULL,
    source TEXT NOT NULL DEFAULT 'fmp',
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS earnings_ticker_date_idx ON earnings (ticker, date);
