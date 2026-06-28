import type { MonitorEvent } from '@swing-x/shared-types';

/**
 * Adaptation-monitor event feed fixture. Each event pairs a detector firing with
 * the action the engine took. Ordered newest-first.
 */
export const monitorFixtures: MonitorEvent[] = [
  {
    ts: '2026-06-28T17:48:00Z',
    detector: 'mahalanobis_ood',
    ticker: 'XLU',
    strategyId: 'str_ood_guard_v1',
    metricValue: 4.7,
    threshold: 3.5,
    actionTaken: 'suppress',
    detail:
      'Feature vector 4.7σ from training centroid — signals suppressed until back in-distribution'
  },
  {
    ts: '2026-06-28T16:20:00Z',
    detector: 'live_calibration_drift',
    strategyId: 'str_zerodte_gamma_v9',
    metricValue: 0.21,
    threshold: 0.15,
    actionTaken: 'recalibrate',
    detail:
      'Reliability gap (predicted-vs-realized) 0.21 over trailing 60 trades — isotonic recalibration queued'
  },
  {
    ts: '2026-06-28T15:05:00Z',
    detector: 'rolling_expectancy_decay',
    strategyId: 'str_riskoff_fade_v2',
    metricValue: -0.05,
    threshold: 0.0,
    actionTaken: 'quarantine',
    detail: '20-trade rolling OOS expectancy turned negative — strategy quarantined pending review'
  },
  {
    ts: '2026-06-28T14:30:00Z',
    detector: 'psi_feature_drift',
    ticker: 'IWM',
    metricValue: 0.27,
    threshold: 0.2,
    actionTaken: 'refit',
    detail: 'Population Stability Index 0.27 on credit-spread feature — nightly refit flagged'
  },
  {
    ts: '2026-06-28T13:10:00Z',
    detector: 'realized_vol_regime_shift',
    ticker: 'QQQ',
    metricValue: 1.34,
    threshold: 1.25,
    actionTaken: 'alert',
    detail:
      'RV 5d/20d ratio crossed 1.25 — vol-expansion regime engaged, trend strategies prioritized'
  },
  {
    ts: '2026-06-28T11:55:00Z',
    detector: 'drawdown_guard',
    strategyId: 'str_compression_squeeze_v5',
    metricValue: 0.62,
    threshold: 0.7,
    actionTaken: 'shrink',
    detail: 'Equity at 62% of peak — position sizing shrunk to 0.5x while inside soft-drawdown band'
  },
  {
    ts: '2026-06-27T23:40:00Z',
    detector: 'pbo_gate_breach',
    strategyId: 'str_meanrev_overnight_v1',
    metricValue: 0.71,
    threshold: 0.5,
    actionTaken: 'disable',
    detail: 'Probability of Backtest Overfit 0.71 on re-evaluation — strategy disabled and retired'
  },
  {
    ts: '2026-06-27T06:05:00Z',
    detector: 'cost_model_divergence',
    ticker: 'SMH',
    metricValue: 0.41,
    threshold: 0.3,
    actionTaken: 'recalibrate',
    detail:
      'Realized slippage exceeded modeled cost by 41% — cost model recalibrated from live fills'
  }
];
