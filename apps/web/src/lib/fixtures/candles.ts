import type { Ticker } from '@swing-x/shared-types';

/** OHLC candle in the shape lightweight-charts expects (UNIX seconds time). */
export interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

/** A point on the VWAP overlay line. */
export interface LinePoint {
  time: number;
  value: number;
}

/** Per-ticker starting price for the synthetic series. Roughly matches fixtures. */
const ANCHOR: Record<Ticker, number> = {
  SPY: 548,
  QQQ: 492,
  IWM: 211,
  DIA: 402,
  XLF: 49,
  XLK: 238,
  XLE: 96,
  SMH: 268,
  XLV: 142,
  XLU: 78
};

/**
 * Small deterministic LCG so the chart is stable across reloads and SSR/CSR
 * without pulling in a PRNG dependency.
 */
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

function seedFor(ticker: Ticker): number {
  let h = 2166136261;
  for (let i = 0; i < ticker.length; i++) {
    h ^= ticker.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/**
 * Build a deterministic synthetic candle series for a ticker, ending just before
 * `endTs`. Produces `bars` 30-minute candles. Pure stand-in for real OHLCV.
 */
export function buildCandles(ticker: Ticker, endTs: string, bars = 120): Candle[] {
  const rng = makeRng(seedFor(ticker));
  const anchor = ANCHOR[ticker] ?? 100;
  const stepSec = 30 * 60;
  const endSec = Math.floor(new Date(endTs).getTime() / 1000);
  const startSec = endSec - (bars - 1) * stepSec;

  const candles: Candle[] = [];
  let price = anchor * (0.97 + rng() * 0.02);
  for (let i = 0; i < bars; i++) {
    const drift = (rng() - 0.48) * anchor * 0.004;
    const open = price;
    const close = Math.max(0.01, open + drift);
    const wick = anchor * 0.0025 * (0.5 + rng());
    const high = Math.max(open, close) + wick * rng();
    const low = Math.min(open, close) - wick * rng();
    candles.push({
      time: startSec + i * stepSec,
      open: round2(open),
      high: round2(high),
      low: round2(low),
      close: round2(close)
    });
    price = close;
  }
  return candles;
}

/** Rolling VWAP-like overlay derived from the candle series (typical price). */
export function buildVwap(candles: Candle[]): LinePoint[] {
  let cumPv = 0;
  let cumV = 0;
  return candles.map((c, i) => {
    const typical = (c.high + c.low + c.close) / 3;
    // Synthetic volume weight that varies but stays positive.
    const vol = 1 + ((i * 37) % 11);
    cumPv += typical * vol;
    cumV += vol;
    return { time: c.time, value: round2(cumPv / cumV) };
  });
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}
