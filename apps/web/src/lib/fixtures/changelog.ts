import type { ChangelogWeek } from '@swing-x/shared-types';

/**
 * Weekly self-changelog fixture — what the engine decayed, retired, and adapted.
 * Ordered newest-first.
 */
export const changelogFixtures: ChangelogWeek[] = [
  {
    week: '2026-W26',
    decayed: [
      'str_riskoff_fade_v2 — 20-trade rolling expectancy went negative as credit-spread signal lost edge',
      'str_zerodte_gamma_v9 — calibration reliability gap widened to 0.21; predicted conviction overstated',
      'XLU feature manifold drifted out-of-distribution (Mahalanobis 4.7σ); signals auto-suppressed'
    ],
    retired: [
      'str_meanrev_overnight_v1 — PBO re-evaluated at 0.71 (> 0.5 gate); disabled and retired',
      'Overnight-gap feature group deprecated: profit factor fell below 1.0 across all regimes'
    ],
    adapted: [
      'str_gex_revert_v7 promoted: DSR 0.82, PBO 0.18, positive across 4 regimes',
      'str_breadth_thrust_v6 promoted after clearing cost-aware OOS gate (MAR 1.44)',
      'Cost model recalibrated from live SMH fills after 41% slippage divergence',
      'str_zerodte_gamma_v9 queued for isotonic recalibration on next nightly cycle',
      'Position sizing shrunk to 0.5x for str_compression_squeeze_v5 inside soft-drawdown band'
    ]
  },
  {
    week: '2026-W25',
    decayed: [
      'str_compression_squeeze_v5 — Sharpe deflation accelerated as vol-compression regime persisted',
      'IWM credit-spread feature PSI rose to 0.27, signaling distribution shift'
    ],
    retired: ['str_gap_scalp_v2 — failed promotion gate three consecutive evaluations'],
    adapted: [
      'str_trend_pullback_v4 promoted on improved cost-aware expectancy (0.46R)',
      'Embargo window widened from 10 to 12 bars on swing-horizon CV folds',
      'Regime classifier retrained with updated dealer-gamma proxy'
    ]
  }
];
