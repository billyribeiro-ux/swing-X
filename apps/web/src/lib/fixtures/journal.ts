import type { Trade } from '@swing-x/shared-types';

/**
 * Paper-trade journal fixture. A mix of closed (with realized R) and one open
 * position. Each trade carries an attribution snapshot from decision time.
 */
export const journalFixtures: Trade[] = [
  {
    tradeId: 'trd_00451',
    signalId: 'sig_2026-06-26_SPY_088',
    strategyId: 'str_gex_revert_v7',
    ticker: 'SPY',
    side: 'long',
    mode: 'paper',
    entryTs: '2026-06-26T13:40:00Z',
    fillPx: 545.1,
    fillTs: '2026-06-26T13:40:06Z',
    exitTs: '2026-06-26T19:58:00Z',
    exitPx: 552.7,
    pnlR: 1.49,
    costFrac: 0.04,
    attribution: [
      {
        layer: 'regime',
        key: 'dealer_gamma_sign',
        contribution: 0.3,
        detail: 'Negative gamma mean-revert'
      },
      {
        layer: 'location',
        key: 'put_wall_distance',
        contribution: 0.21,
        detail: 'Bounce off put wall'
      },
      {
        layer: 'trigger',
        key: 'vwap_reclaim',
        contribution: 0.17,
        detail: 'VWAP reclaim on rising delta'
      }
    ]
  },
  {
    tradeId: 'trd_00452',
    signalId: 'sig_2026-06-26_QQQ_091',
    strategyId: 'str_breakout_vol_exp_v3',
    ticker: 'QQQ',
    side: 'long',
    mode: 'paper',
    entryTs: '2026-06-26T14:12:00Z',
    fillPx: 489.4,
    fillTs: '2026-06-26T14:12:09Z',
    exitTs: '2026-06-26T18:22:00Z',
    exitPx: 486.1,
    pnlR: -0.92,
    costFrac: 0.05,
    attribution: [
      { layer: 'regime', key: 'realized_vol_slope', contribution: 0.26, detail: 'RV expanding' },
      {
        layer: 'location',
        key: 'poc_breakout',
        contribution: 0.19,
        detail: 'POC breakout failed to hold'
      }
    ]
  },
  {
    tradeId: 'trd_00453',
    signalId: 'sig_2026-06-27_XLF_103',
    strategyId: 'str_trend_pullback_v4',
    ticker: 'XLF',
    side: 'long',
    mode: 'paper',
    entryTs: '2026-06-27T13:58:00Z',
    fillPx: 48.7,
    fillTs: '2026-06-27T13:58:04Z',
    exitTs: '2026-06-27T20:00:00Z',
    exitPx: 49.9,
    pnlR: 1.6,
    costFrac: 0.03,
    attribution: [
      {
        layer: 'regime',
        key: 'dealer_gamma_sign',
        contribution: 0.24,
        detail: 'Positive gamma grind'
      },
      {
        layer: 'location',
        key: 'ema_pullback',
        contribution: 0.21,
        detail: '20EMA pullback at value'
      },
      {
        layer: 'trigger',
        key: 'hammer_reversal',
        contribution: 0.13,
        detail: 'Bullish reversal candle'
      }
    ]
  },
  {
    tradeId: 'trd_00454',
    signalId: 'sig_2026-06-27_IWM_110',
    strategyId: 'str_riskoff_fade_v2',
    ticker: 'IWM',
    side: 'short',
    mode: 'paper',
    entryTs: '2026-06-27T15:22:00Z',
    fillPx: 212.8,
    fillTs: '2026-06-27T15:22:07Z',
    exitTs: '2026-06-27T19:10:00Z',
    exitPx: 213.4,
    pnlR: -0.34,
    costFrac: 0.06,
    attribution: [
      {
        layer: 'regime',
        key: 'credit_spread_delta',
        contribution: 0.29,
        detail: 'HY OAS widening'
      },
      {
        layer: 'location',
        key: 'failed_breakout',
        contribution: 0.18,
        detail: 'Rejected at prior high'
      }
    ]
  },
  {
    tradeId: 'trd_00455',
    signalId: 'sig_2026-06-28_SPY_001',
    strategyId: 'str_gex_revert_v7',
    ticker: 'SPY',
    side: 'long',
    mode: 'paper',
    entryTs: '2026-06-28T13:32:00Z',
    fillPx: 548.25,
    fillTs: '2026-06-28T13:32:05Z',
    // Open position — no exit yet.
    costFrac: 0.04,
    attribution: [
      {
        layer: 'regime',
        key: 'dealer_gamma_sign',
        contribution: 0.31,
        detail: 'Negative gamma mean-revert'
      },
      { layer: 'location', key: 'put_wall_distance', contribution: 0.22, detail: 'Above put wall' },
      { layer: 'trigger', key: 'vwap_reclaim', contribution: 0.18, detail: 'VWAP reclaim' }
    ]
  },
  {
    tradeId: 'trd_00456',
    signalId: 'sig_2026-06-25_XLE_077',
    strategyId: 'str_compression_squeeze_v5',
    ticker: 'XLE',
    side: 'long',
    mode: 'paper',
    entryTs: '2026-06-25T14:50:00Z',
    fillPx: 95.9,
    fillTs: '2026-06-25T14:50:08Z',
    exitTs: '2026-06-27T20:00:00Z',
    exitPx: 98.2,
    pnlR: 1.39,
    costFrac: 0.05,
    attribution: [
      {
        layer: 'regime',
        key: 'bollinger_squeeze',
        contribution: 0.18,
        detail: 'BB squeeze resolving up'
      },
      { layer: 'trigger', key: 'momentum_turn', contribution: 0.09, detail: 'MACD inflection' }
    ]
  }
];
