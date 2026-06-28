import type { Strategy } from '@swing-x/shared-types';

/**
 * Strategy population fixture with a mix of passed/failed promotion gates and
 * varied lifecycle statuses. Note: win rate is deliberately absent — it is a
 * banned selection metric.
 */
export const populationFixtures: Strategy[] = [
  {
    strategyId: 'str_gex_revert_v7',
    horizon: 'swing',
    status: 'promoted',
    generation: 7,
    genomeSummary: 'GEX mean-revert | put-wall location | VWAP reclaim trigger | meta-GBM',
    latestScore: {
      dsr: 0.82,
      pbo: 0.18,
      oosExpectancyCostAware: 0.42,
      profitFactor: 1.74,
      cvar5: -1.6,
      mar: 1.31,
      nRegimesPositive: 4,
      passedGate: true,
      evaluatedAt: '2026-06-27T06:00:00Z'
    }
  },
  {
    strategyId: 'str_trend_pullback_v4',
    horizon: 'short_swing',
    status: 'promoted',
    generation: 4,
    genomeSummary: 'Long-gamma drift | 20EMA pullback | hammer reversal | calibrated GBM',
    latestScore: {
      dsr: 0.61,
      pbo: 0.27,
      oosExpectancyCostAware: 0.46,
      profitFactor: 1.66,
      cvar5: -1.3,
      mar: 1.12,
      nRegimesPositive: 3,
      passedGate: true,
      evaluatedAt: '2026-06-27T06:00:00Z'
    }
  },
  {
    strategyId: 'str_breakout_vol_exp_v3',
    horizon: 'short_swing',
    status: 'candidate',
    generation: 3,
    genomeSummary: 'Vol-expansion | POC breakout | range-expansion trigger | trend-GBM',
    latestScore: {
      dsr: 0.34,
      pbo: 0.41,
      oosExpectancyCostAware: 0.51,
      profitFactor: 1.49,
      cvar5: -1.9,
      mar: 0.88,
      nRegimesPositive: 2,
      passedGate: false,
      evaluatedAt: '2026-06-27T06:00:00Z'
    }
  },
  {
    strategyId: 'str_compression_squeeze_v5',
    horizon: 'swing',
    status: 'candidate',
    generation: 5,
    genomeSummary: 'Vol-compression | BB-squeeze | MACD inflection | squeeze-GBM',
    latestScore: {
      dsr: 0.12,
      pbo: 0.48,
      oosExpectancyCostAware: 0.21,
      profitFactor: 1.22,
      cvar5: -1.4,
      mar: 0.61,
      nRegimesPositive: 2,
      passedGate: false,
      evaluatedAt: '2026-06-27T06:00:00Z'
    }
  },
  {
    strategyId: 'str_riskoff_fade_v2',
    horizon: 'day',
    status: 'quarantined',
    generation: 2,
    genomeSummary: 'Risk-off fade | failed-breakout | VWAP-loss trigger | fade-GBM',
    latestScore: {
      dsr: -0.21,
      pbo: 0.58,
      oosExpectancyCostAware: 0.33,
      profitFactor: 1.18,
      cvar5: -2.1,
      mar: 0.44,
      nRegimesPositive: 1,
      passedGate: false,
      evaluatedAt: '2026-06-27T06:00:00Z'
    }
  },
  {
    strategyId: 'str_zerodte_gamma_v9',
    horizon: 'zero_dte',
    status: 'demoted',
    generation: 9,
    genomeSummary: '0DTE charm-fade | call-wall pin | RSI-divergence | odte-GBM',
    latestScore: {
      dsr: -0.44,
      pbo: 0.63,
      oosExpectancyCostAware: 0.12,
      profitFactor: 1.06,
      cvar5: -2.4,
      mar: 0.19,
      nRegimesPositive: 1,
      passedGate: false,
      evaluatedAt: '2026-06-27T06:00:00Z'
    }
  },
  {
    strategyId: 'str_meanrev_overnight_v1',
    horizon: 'swing',
    status: 'retired',
    generation: 1,
    genomeSummary: 'Overnight gap mean-revert | gap-fill target | session-open trigger',
    latestScore: {
      dsr: -0.78,
      pbo: 0.71,
      oosExpectancyCostAware: -0.08,
      profitFactor: 0.94,
      cvar5: -2.9,
      mar: -0.12,
      nRegimesPositive: 0,
      passedGate: false,
      evaluatedAt: '2026-06-20T06:00:00Z'
    }
  },
  {
    strategyId: 'str_ood_guard_v1',
    horizon: 'scalp',
    status: 'candidate',
    generation: 1,
    genomeSummary: 'OOD-guarded micro-scalp | support test | size-capped under drift',
    latestScore: {
      dsr: 0.05,
      pbo: 0.49,
      oosExpectancyCostAware: 0.05,
      profitFactor: 1.09,
      cvar5: -1.1,
      mar: 0.33,
      nRegimesPositive: 1,
      passedGate: false,
      evaluatedAt: '2026-06-27T06:00:00Z'
    }
  },
  {
    strategyId: 'str_breadth_thrust_v6',
    horizon: 'swing',
    status: 'promoted',
    generation: 6,
    genomeSummary: 'Breadth-thrust momentum | sector confirm | acceptance trigger | ensemble',
    latestScore: {
      dsr: 0.74,
      pbo: 0.22,
      oosExpectancyCostAware: 0.39,
      profitFactor: 1.81,
      cvar5: -1.5,
      mar: 1.44,
      nRegimesPositive: 4,
      passedGate: true,
      evaluatedAt: '2026-06-27T06:00:00Z'
    }
  }
];
