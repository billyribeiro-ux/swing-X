import type { Signal } from '@swing-x/shared-types';

/**
 * Fixture signals spanning several regimes, sides, and horizons. Numbers are
 * internally consistent (entry/stop/target relationships, R:R, conviction in
 * [0,1]). Stand-in data until the Rust API is wired in.
 */
export const signalFixtures: Signal[] = [
  {
    signalId: 'sig_2026-06-28_SPY_001',
    strategyId: 'str_gex_revert_v7',
    ticker: 'SPY',
    side: 'long',
    decisionTs: '2026-06-28T13:32:00Z',
    horizon: 'swing',
    entry: 548.2,
    stop: 543.1,
    target1: 556.0,
    target2: 562.4,
    rr1: 1.53,
    rr2: 2.78,
    conviction: 0.71,
    cohortN: 184,
    regimeDesc: 'Short-gamma pin below put wall; dealers buy dips into 545 support',
    why: [
      {
        layer: 'tradeability',
        key: 'spread_bps',
        contribution: 0.08,
        detail: 'Effective spread 1.2bps; ADV ample for 2R size'
      },
      {
        layer: 'regime',
        key: 'dealer_gamma_sign',
        contribution: 0.31,
        detail: 'Net dealer gamma negative (-$0.9bn/1%): mean-reverting tape'
      },
      {
        layer: 'location',
        key: 'put_wall_distance',
        contribution: 0.22,
        detail: 'Price 0.4% above 545 put wall; high-probability bounce zone'
      },
      {
        layer: 'trigger',
        key: 'vwap_reclaim',
        contribution: 0.18,
        detail: 'Reclaimed session VWAP on rising delta after sweep'
      },
      {
        layer: 'event',
        key: 'macro_quiet',
        contribution: 0.04,
        detail: 'No tier-1 prints until Thursday claims'
      }
    ],
    invalidation: '15m close below 543.0 (put wall fails) voids long thesis',
    cohortExpectancy: 0.42,
    cvar5: -1.6,
    leadTime: 34,
    payloadJson: {
      model: 'meta_label_gbm_v7',
      calibratedProb: 0.71,
      rawProb: 0.66,
      features: { gex_z: -1.84, put_wall: 545, vwap_dev: -0.0011, dix: 0.402 },
      cohort: { n: 184, lookbackDays: 540, purgeBars: 12 }
    }
  },
  {
    signalId: 'sig_2026-06-28_QQQ_004',
    strategyId: 'str_breakout_vol_exp_v3',
    ticker: 'QQQ',
    side: 'long',
    decisionTs: '2026-06-28T14:05:00Z',
    horizon: 'short_swing',
    entry: 492.6,
    stop: 488.9,
    target1: 499.2,
    target2: 505.0,
    rr1: 1.78,
    rr2: 3.35,
    conviction: 0.64,
    cohortN: 96,
    regimeDesc: 'Vol-expansion breakout above prior balance high on widening realized vol',
    why: [
      {
        layer: 'tradeability',
        key: 'liquidity',
        contribution: 0.06,
        detail: 'Top-of-book depth healthy; slippage budget 0.3R'
      },
      {
        layer: 'regime',
        key: 'realized_vol_slope',
        contribution: 0.27,
        detail: 'RV 5d/20d ratio 1.34, expanding; trend-follow regime'
      },
      {
        layer: 'location',
        key: 'poc_breakout',
        contribution: 0.2,
        detail: 'Cleared 491.8 composite POC with acceptance'
      },
      {
        layer: 'trigger',
        key: 'range_expansion',
        contribution: 0.16,
        detail: 'NR7 -> expansion bar with delta confirmation'
      },
      {
        layer: 'event',
        key: 'sector_breadth',
        contribution: 0.05,
        detail: 'SMH/XLK breadth thrust supportive'
      }
    ],
    invalidation: 'Loss of 488.5 (back inside balance) invalidates breakout',
    cohortExpectancy: 0.51,
    cvar5: -1.9,
    leadTime: 22,
    payloadJson: {
      model: 'trend_gbm_v3',
      calibratedProb: 0.64,
      rawProb: 0.61,
      features: { rv_ratio: 1.34, poc: 491.8, breadth_z: 1.21 },
      cohort: { n: 96, lookbackDays: 380, purgeBars: 10 }
    }
  },
  {
    signalId: 'sig_2026-06-28_IWM_009',
    strategyId: 'str_riskoff_fade_v2',
    ticker: 'IWM',
    side: 'short',
    decisionTs: '2026-06-28T15:18:00Z',
    horizon: 'day',
    entry: 211.4,
    stop: 213.2,
    target1: 207.8,
    target2: 205.1,
    rr1: 2.0,
    rr2: 3.5,
    conviction: 0.58,
    cohortN: 71,
    regimeDesc: 'Risk-off rotation; small caps lead lower as credit spreads widen',
    why: [
      {
        layer: 'tradeability',
        key: 'borrow',
        contribution: 0.03,
        detail: 'Easy-to-borrow; no locate friction'
      },
      {
        layer: 'regime',
        key: 'credit_spread_delta',
        contribution: 0.29,
        detail: 'HY OAS +9bps on day; risk-off confirmed'
      },
      {
        layer: 'location',
        key: 'failed_breakout',
        contribution: 0.19,
        detail: 'Rejected at prior-day high; trapped longs above'
      },
      {
        layer: 'trigger',
        key: 'vwap_loss',
        contribution: 0.13,
        detail: 'Lost VWAP with expanding sell delta'
      },
      {
        layer: 'event',
        key: 'dxy_bid',
        contribution: 0.04,
        detail: 'DXY bid into risk-off; headwind for beta'
      }
    ],
    invalidation: 'Reclaim of 213.4 (back above failed high) invalidates short',
    cohortExpectancy: 0.33,
    cvar5: -2.1,
    leadTime: 18,
    payloadJson: {
      model: 'fade_gbm_v2',
      calibratedProb: 0.58,
      rawProb: 0.55,
      features: { hy_oas_d: 9, dxy_z: 0.84, vwap_dev: 0.0008 },
      cohort: { n: 71, lookbackDays: 300, purgeBars: 8 }
    }
  },
  {
    signalId: 'sig_2026-06-28_XLE_012',
    strategyId: 'str_compression_squeeze_v5',
    ticker: 'XLE',
    side: 'long',
    decisionTs: '2026-06-28T14:48:00Z',
    horizon: 'swing',
    entry: 96.85,
    stop: 95.2,
    target1: 99.6,
    target2: 102.3,
    rr1: 1.67,
    rr2: 3.3,
    conviction: 0.49,
    cohortN: 142,
    regimeDesc: 'Vol-compression squeeze; long-gamma coil resolving with energy bid',
    why: [
      {
        layer: 'tradeability',
        key: 'spread_bps',
        contribution: 0.05,
        detail: 'Spread 2.1bps; acceptable for swing hold'
      },
      {
        layer: 'regime',
        key: 'bollinger_squeeze',
        contribution: 0.18,
        detail: 'BB width at 8th percentile; energy expansion pending'
      },
      {
        layer: 'location',
        key: 'value_low',
        contribution: 0.12,
        detail: 'Sitting at 30d value-area low; favorable RR'
      },
      {
        layer: 'trigger',
        key: 'momentum_turn',
        contribution: 0.09,
        detail: 'Daily MACD histogram inflecting up'
      },
      {
        layer: 'event',
        key: 'crude_term',
        contribution: 0.03,
        detail: 'Crude backwardation steepening; tailwind'
      }
    ],
    invalidation: 'Daily close below 95.0 negates compression-long setup',
    cohortExpectancy: 0.21,
    cvar5: -1.4,
    leadTime: 51,
    payloadJson: {
      model: 'squeeze_gbm_v5',
      calibratedProb: 0.49,
      rawProb: 0.52,
      features: { bb_width_pct: 0.08, va_low: 96.4, macd_hist: 0.04 },
      cohort: { n: 142, lookbackDays: 600, purgeBars: 14 }
    }
  },
  {
    signalId: 'sig_2026-06-28_SMH_015',
    strategyId: 'str_zerodte_gamma_v9',
    ticker: 'SMH',
    side: 'short',
    decisionTs: '2026-06-28T17:42:00Z',
    horizon: 'zero_dte',
    entry: 268.3,
    stop: 270.1,
    target1: 265.2,
    rr1: 1.72,
    conviction: 0.45,
    cohortN: 53,
    regimeDesc: 'Transition regime; late-day charm flows fade the morning rip',
    why: [
      {
        layer: 'tradeability',
        key: 'gamma_liquidity',
        contribution: 0.04,
        detail: '0DTE chain liquid; tight option markets'
      },
      {
        layer: 'regime',
        key: 'charm_flow',
        contribution: 0.21,
        detail: 'Negative charm into close pressures spot lower'
      },
      {
        layer: 'location',
        key: 'call_wall',
        contribution: 0.16,
        detail: 'Pinned beneath 270 call wall; capped upside'
      },
      {
        layer: 'trigger',
        key: 'momentum_exhaustion',
        contribution: 0.08,
        detail: 'RSI divergence on 5m into resistance'
      },
      {
        layer: 'event',
        key: 'eod_imbalance',
        contribution: -0.02,
        detail: 'Mild buy imbalance indicated — minor headwind'
      }
    ],
    invalidation: 'Break and hold above 270.2 (call wall breach) voids fade',
    cohortExpectancy: 0.12,
    cvar5: -2.4,
    leadTime: 9,
    payloadJson: {
      model: 'odte_gbm_v9',
      calibratedProb: 0.45,
      rawProb: 0.5,
      features: { charm: -0.31, call_wall: 270, rsi_div: true },
      cohort: { n: 53, lookbackDays: 210, purgeBars: 4 }
    }
  },
  {
    signalId: 'sig_2026-06-28_XLF_021',
    strategyId: 'str_trend_pullback_v4',
    ticker: 'XLF',
    side: 'long',
    decisionTs: '2026-06-28T13:55:00Z',
    horizon: 'short_swing',
    entry: 49.1,
    stop: 48.35,
    target1: 50.4,
    target2: 51.2,
    rr1: 1.73,
    rr2: 2.8,
    conviction: 0.67,
    cohortN: 211,
    regimeDesc: 'Risk-on; orderly pullback to rising 20EMA in long-gamma drift',
    why: [
      {
        layer: 'tradeability',
        key: 'spread_bps',
        contribution: 0.07,
        detail: 'Penny-wide; ideal execution'
      },
      {
        layer: 'regime',
        key: 'dealer_gamma_sign',
        contribution: 0.24,
        detail: 'Positive dealer gamma: low-vol grind regime'
      },
      {
        layer: 'location',
        key: 'ema_pullback',
        contribution: 0.21,
        detail: 'Tagged 20EMA at value; trend intact'
      },
      {
        layer: 'trigger',
        key: 'hammer_reversal',
        contribution: 0.13,
        detail: 'Bullish reversal candle on declining volume'
      },
      {
        layer: 'event',
        key: 'rates_stable',
        contribution: 0.02,
        detail: '2y yield range-bound; financials supported'
      }
    ],
    invalidation: 'Close below 48.3 (EMA + structure break) invalidates pullback long',
    cohortExpectancy: 0.46,
    cvar5: -1.3,
    leadTime: 41,
    payloadJson: {
      model: 'pullback_gbm_v4',
      calibratedProb: 0.67,
      rawProb: 0.63,
      features: { gex_z: 1.42, ema20: 48.9, vol_decline: true },
      cohort: { n: 211, lookbackDays: 720, purgeBars: 12 }
    }
  },
  {
    signalId: 'sig_2026-06-28_XLU_028',
    strategyId: 'str_ood_guard_v1',
    ticker: 'XLU',
    side: 'long',
    decisionTs: '2026-06-28T16:10:00Z',
    horizon: 'scalp',
    entry: 78.4,
    stop: 77.9,
    target1: 79.3,
    rr1: 1.8,
    conviction: 0.32,
    cohortN: 19,
    regimeDesc: 'Out-of-distribution: features outside trained manifold — size cut, monitor only',
    why: [
      {
        layer: 'tradeability',
        key: 'spread_bps',
        contribution: 0.04,
        detail: 'Spread acceptable but thin tape'
      },
      {
        layer: 'regime',
        key: 'ood_flag',
        contribution: -0.14,
        detail: 'Mahalanobis distance 4.7σ from training centroid'
      },
      {
        layer: 'location',
        key: 'support_test',
        contribution: 0.09,
        detail: 'Testing weekly support; thin confluence'
      },
      {
        layer: 'trigger',
        key: 'micro_bounce',
        contribution: 0.06,
        detail: 'Tape bounce on small buy delta'
      },
      {
        layer: 'event',
        key: 'low_conviction',
        contribution: -0.03,
        detail: 'Cohort too small for reliable expectancy'
      }
    ],
    invalidation: 'Any 5m close below 77.8 ends scalp; OOD auto-suppress on widening distance',
    cohortExpectancy: 0.05,
    cvar5: -1.1,
    leadTime: 6,
    payloadJson: {
      model: 'guard_gbm_v1',
      calibratedProb: 0.32,
      rawProb: 0.41,
      features: { mahalanobis: 4.7, ood: true },
      cohort: { n: 19, lookbackDays: 90, purgeBars: 2 }
    }
  }
];
