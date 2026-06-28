import type { MonitorAction, Regime, Side, StrategyStatus } from '@swing-x/shared-types';

/**
 * Mappings from domain enums to Tailwind utility class fragments. Centralized so
 * color semantics stay consistent across the scoreboard, badges, and feeds.
 */

/** Text color for a long/short side. */
export function sideTextClass(side: Side): string {
  return side === 'long' ? 'text-up' : 'text-down';
}

/** Badge classes (bg + text + border) for a long/short side. */
export function sideBadgeClass(side: Side): string {
  return side === 'long' ? 'bg-up/10 text-up border-up/30' : 'bg-down/10 text-down border-down/30';
}

/** Sign-aware text color for an R-multiple / signed metric. */
export function signTextClass(n: number | undefined | null): string {
  if (n == null || Number.isNaN(n)) return 'text-base-400';
  if (n > 0) return 'text-up';
  if (n < 0) return 'text-down';
  return 'text-base-300';
}

/** Badge classes for a strategy lifecycle status. */
export function statusBadgeClass(status: StrategyStatus): string {
  switch (status) {
    case 'promoted':
      return 'bg-up/10 text-up border-up/30';
    case 'candidate':
      return 'bg-accent/10 text-accent border-accent/30';
    case 'quarantined':
      return 'bg-warn/10 text-warn border-warn/30';
    case 'demoted':
      return 'bg-caution/10 text-caution border-caution/30';
    case 'retired':
      return 'bg-base-700/40 text-base-400 border-base-600/40';
  }
}

/** Badge classes for a monitor action, colored by severity. */
export function actionBadgeClass(action: MonitorAction): string {
  switch (action) {
    case 'disable':
    case 'quarantine':
      return 'bg-down/10 text-down border-down/30';
    case 'suppress':
    case 'shrink':
      return 'bg-caution/10 text-caution border-caution/30';
    case 'refit':
    case 'recalibrate':
      return 'bg-accent/10 text-accent border-accent/30';
    case 'alert':
      return 'bg-warn/10 text-warn border-warn/30';
  }
}

/** Severity rank for a monitor action — drives the left rail color of a row. */
export function actionSeverity(action: MonitorAction): 'high' | 'medium' | 'low' {
  switch (action) {
    case 'disable':
    case 'quarantine':
      return 'high';
    case 'suppress':
    case 'shrink':
    case 'recalibrate':
      return 'medium';
    case 'refit':
    case 'alert':
      return 'low';
  }
}

/** Left-border rail class for a severity level. */
export function severityRailClass(sev: 'high' | 'medium' | 'low'): string {
  switch (sev) {
    case 'high':
      return 'border-l-down';
    case 'medium':
      return 'border-l-caution';
    case 'low':
      return 'border-l-accent';
  }
}

/** Subtle text color for a regime label. */
export function regimeTextClass(regime: Regime): string {
  switch (regime) {
    case 'risk_on':
    case 'long_gamma':
    case 'vol_compression':
      return 'text-up-dim';
    case 'risk_off':
    case 'short_gamma':
    case 'vol_expansion':
      return 'text-down-dim';
    case 'out_of_distribution':
      return 'text-warn';
    case 'transition':
      return 'text-base-300';
  }
}

/** Driver-layer accent color used in the attribution panel. */
export function layerColor(
  layer: 'tradeability' | 'regime' | 'location' | 'trigger' | 'event'
): string {
  switch (layer) {
    case 'tradeability':
      return 'text-base-300';
    case 'regime':
      return 'text-accent';
    case 'location':
      return 'text-info';
    case 'trigger':
      return 'text-up';
    case 'event':
      return 'text-warn';
  }
}
