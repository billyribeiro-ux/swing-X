/**
 * @swing-x/shared-types
 *
 * DTO interfaces mirroring the swing-X backend API contract. The Rust orchestrator
 * (axum API) and the Python ML worker align their serialized payloads to these
 * shapes; the SvelteKit dashboard consumes them. Field names are camelCase here and
 * are expected to be camelCase on the wire.
 *
 * This is intentionally dependency-free — pure TypeScript type declarations plus a
 * couple of small const arrays used for enumeration at runtime (e.g. table filters).
 */

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

/** Fixed 10-ticker US equity-index ETF universe. */
export const TICKERS = [
  'SPY',
  'QQQ',
  'IWM',
  'DIA',
  'XLF',
  'XLK',
  'XLE',
  'SMH',
  'XLV',
  'XLU',
] as const;
export type Ticker = (typeof TICKERS)[number];

export const SIDES = ['long', 'short'] as const;
export type Side = (typeof SIDES)[number];

export const HORIZONS = [
  'swing',
  'short_swing',
  'day',
  'zero_dte',
  'scalp',
] as const;
export type Horizon = (typeof HORIZONS)[number];

export const REGIMES = [
  'short_gamma',
  'long_gamma',
  'vol_expansion',
  'vol_compression',
  'risk_off',
  'risk_on',
  'transition',
  'out_of_distribution',
] as const;
export type Regime = (typeof REGIMES)[number];

/** Lifecycle status of a strategy genome in the population. */
export const STRATEGY_STATUSES = [
  'candidate',
  'promoted',
  'quarantined',
  'demoted',
  'retired',
] as const;
export type StrategyStatus = (typeof STRATEGY_STATUSES)[number];

/** The four feature-engine layers plus event context that produce driver attribution. */
export const DRIVER_LAYERS = [
  'tradeability',
  'regime',
  'location',
  'trigger',
  'event',
] as const;
export type DriverLayer = (typeof DRIVER_LAYERS)[number];

/** Actions the forward-adaptation monitor can take in response to a detector firing. */
export const MONITOR_ACTIONS = [
  'shrink',
  'quarantine',
  'refit',
  'recalibrate',
  'suppress',
  'disable',
  'alert',
] as const;
export type MonitorAction = (typeof MONITOR_ACTIONS)[number];

export const TRADE_MODES = ['paper', 'live'] as const;
export type TradeMode = (typeof TRADE_MODES)[number];

// ---------------------------------------------------------------------------
// DTOs
// ---------------------------------------------------------------------------

/**
 * A single driver attribution entry: how much one signal contributed within a
 * given feature-engine layer.
 */
export interface Driver {
  layer: DriverLayer;
  /** Stable machine key, e.g. "gex_wall_proximity". */
  key: string;
  /** Signed contribution toward conviction (model-space units). */
  contribution: number;
  /** Human-readable explanation of the driver's current state. */
  detail: string;
}

/**
 * A surfaced, executable trading signal with full attribution and cohort stats.
 */
export interface Signal {
  signalId: string;
  strategyId: string;
  ticker: Ticker;
  side: Side;
  /** ISO-8601 decision timestamp. */
  decisionTs: string;
  horizon: Horizon;
  entry: number;
  stop: number;
  target1: number;
  target2?: number;
  /** Reward:risk to target1. */
  rr1?: number;
  /** Reward:risk to target2. */
  rr2?: number;
  /** Calibrated conviction in [0, 1]. */
  conviction: number;
  /** Number of analog members in the historical cohort. */
  cohortN: number;
  /** Human-readable regime description for this decision. */
  regimeDesc: string;
  /** Per-layer driver attribution. */
  why: Driver[];
  /** The hard invalidation condition that voids the thesis. */
  invalidation: string;
  /** Cohort expectancy in R-multiples. */
  cohortExpectancy?: number;
  /** Conditional value-at-risk at the 5% tail, in R-multiples (negative = loss). */
  cvar5?: number;
  /** Lead-time edge in minutes ahead of naive entry. */
  leadTime?: number;
  /** Full raw decision payload for the JSON inspector. */
  payloadJson: unknown;
}

/**
 * Out-of-sample evaluation scorecard for a strategy. Selection is driven by these
 * cost-aware, overfit-penalized metrics — never by raw win rate.
 */
export interface OosScore {
  /** Deflated Sharpe Ratio. Healthy when > 0. */
  dsr: number;
  /** Probability of Backtest Overfit in [0, 1]. Healthy when < 0.5. */
  pbo: number;
  /** Cost-aware OOS expectancy in R-multiples. */
  oosExpectancyCostAware: number;
  profitFactor: number;
  /** CVaR at the 5% tail, in R-multiples (negative = loss). */
  cvar5: number;
  /** MAR ratio (CAGR / max drawdown). */
  mar: number;
  /** Number of distinct regimes in which the strategy is positive-expectancy. */
  nRegimesPositive: number;
  /** Whether the strategy cleared the promotion gate. */
  passedGate: boolean;
  /** ISO-8601 timestamp of the evaluation. */
  evaluatedAt: string;
}

/**
 * A strategy genome in the evolving population.
 */
export interface Strategy {
  strategyId: string;
  horizon: Horizon;
  status: StrategyStatus;
  /** Search-loop generation that produced this genome. */
  generation: number;
  /** Compact human-readable summary of the genome. */
  genomeSummary: string;
  latestScore?: OosScore;
}

/**
 * An adaptation-monitor event: a detector fired and an action was taken.
 */
export interface MonitorEvent {
  /** ISO-8601 timestamp. */
  ts: string;
  /** Detector name, e.g. "psi_feature_drift". */
  detector: string;
  ticker?: Ticker;
  strategyId?: string;
  /** Observed metric value that triggered the detector. */
  metricValue?: number;
  /** Threshold the metric breached. */
  threshold?: number;
  actionTaken: MonitorAction;
  detail: string;
}

/**
 * A paper (or live) trade in the journal, linked back to its originating
 * signal/strategy with an attribution snapshot taken at decision time.
 */
export interface Trade {
  tradeId: string;
  signalId?: string;
  strategyId?: string;
  ticker: Ticker;
  side: Side;
  mode: TradeMode;
  /** ISO-8601 entry decision timestamp. */
  entryTs: string;
  fillPx: number;
  /** ISO-8601 fill timestamp. */
  fillTs: string;
  /** ISO-8601 exit timestamp, if closed. */
  exitTs?: string;
  exitPx?: number;
  /** Realized PnL in R-multiples, if closed. */
  pnlR?: number;
  /** Round-trip cost as a fraction of risk. */
  costFrac?: number;
  /** Driver attribution snapshot captured at decision time. */
  attribution: Driver[];
}

/**
 * One week of the engine's self-changelog: what decayed, retired, and adapted.
 */
export interface ChangelogWeek {
  /** ISO week label, e.g. "2026-W26". */
  week: string;
  decayed: string[];
  retired: string[];
  adapted: string[];
}
