import { z } from 'zod';
import {
  DRIVER_LAYERS,
  HORIZONS,
  MONITOR_ACTIONS,
  REGIMES,
  SIDES,
  STRATEGY_STATUSES,
  TRADE_MODES
} from '@swing-x/shared-types';

/**
 * Zod v4 schemas mirroring the DTOs in @swing-x/shared-types. These are used to
 * validate payloads at the trust boundary when the real Rust API is wired in
 * (see ./client.ts). The static TS types remain the source of truth; the
 * `satisfies` guards below keep these schemas structurally aligned.
 */

// Open-universe ticker: the ETF scanner emits the fixed TICKERS list, but the
// equity scanner emits arbitrary stock symbols (TSLA, AAPL, ...). Validating
// against the closed ETF enum would fail every equity payload and silently fall
// back to fixtures, so accept any non-empty string at the trust boundary.
export const tickerSchema = z.string().min(1);
export const sideSchema = z.enum(SIDES);
export const horizonSchema = z.enum(HORIZONS);
export const regimeSchema = z.enum(REGIMES);
export const strategyStatusSchema = z.enum(STRATEGY_STATUSES);
export const driverLayerSchema = z.enum(DRIVER_LAYERS);
export const monitorActionSchema = z.enum(MONITOR_ACTIONS);
export const tradeModeSchema = z.enum(TRADE_MODES);

export const driverSchema = z.object({
  layer: driverLayerSchema,
  key: z.string(),
  contribution: z.number(),
  detail: z.string()
});

export const signalSchema = z.object({
  signalId: z.string(),
  strategyId: z.string(),
  ticker: tickerSchema,
  side: sideSchema,
  decisionTs: z.string(),
  horizon: horizonSchema,
  entry: z.number(),
  stop: z.number(),
  target1: z.number(),
  target2: z.number().optional(),
  rr1: z.number().optional(),
  rr2: z.number().optional(),
  conviction: z.number(),
  cohortN: z.number(),
  regimeDesc: z.string(),
  why: z.array(driverSchema),
  invalidation: z.string(),
  cohortExpectancy: z.number().optional(),
  cvar5: z.number().optional(),
  leadTime: z.number().optional(),
  payloadJson: z.unknown()
});

export const oosScoreSchema = z.object({
  dsr: z.number(),
  pbo: z.number(),
  oosExpectancyCostAware: z.number(),
  profitFactor: z.number(),
  cvar5: z.number(),
  mar: z.number(),
  nRegimesPositive: z.number(),
  passedGate: z.boolean(),
  evaluatedAt: z.string()
});

export const strategySchema = z.object({
  strategyId: z.string(),
  horizon: horizonSchema,
  status: strategyStatusSchema,
  generation: z.number(),
  genomeSummary: z.string(),
  latestScore: oosScoreSchema.optional()
});

export const monitorEventSchema = z.object({
  ts: z.string(),
  detector: z.string(),
  ticker: tickerSchema.optional(),
  strategyId: z.string().optional(),
  metricValue: z.number().optional(),
  threshold: z.number().optional(),
  actionTaken: monitorActionSchema,
  detail: z.string()
});

export const tradeSchema = z.object({
  tradeId: z.string(),
  signalId: z.string().optional(),
  strategyId: z.string().optional(),
  ticker: tickerSchema,
  side: sideSchema,
  mode: tradeModeSchema,
  entryTs: z.string(),
  fillPx: z.number(),
  fillTs: z.string(),
  exitTs: z.string().optional(),
  exitPx: z.number().optional(),
  pnlR: z.number().optional(),
  costFrac: z.number().optional(),
  attribution: z.array(driverSchema)
});

export const changelogWeekSchema = z.object({
  week: z.string(),
  decayed: z.array(z.string()),
  retired: z.array(z.string()),
  adapted: z.array(z.string())
});

export const signalsSchema = z.array(signalSchema);
export const populationSchema = z.array(strategySchema);
export const monitorEventsSchema = z.array(monitorEventSchema);
export const journalSchema = z.array(tradeSchema);
export const changelogSchema = z.array(changelogWeekSchema);
