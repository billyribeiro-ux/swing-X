import { describe, expect, it } from 'vitest';
import {
  fmtInt,
  fmtLeadTime,
  fmtPct,
  fmtPrice,
  fmtR,
  fmtRatio,
  fmtSigned,
  fmtTs,
  fmtUnit,
  humanize
} from './format';

describe('format helpers', () => {
  it('formats prices with two decimals and a dash fallback', () => {
    expect(fmtPrice(548.2)).toBe('548.20');
    expect(fmtPrice(undefined)).toBe('—');
    expect(fmtPrice(null)).toBe('—');
    expect(fmtPrice(Number.NaN)).toBe('—');
  });

  it('formats signed R-multiples', () => {
    expect(fmtR(0.42)).toBe('+0.42R');
    expect(fmtR(-1.6)).toBe('-1.60R');
    expect(fmtR(0)).toBe('0.00R');
    expect(fmtR(undefined)).toBe('—');
  });

  it('formats ratios with a multiplication suffix', () => {
    expect(fmtRatio(1.53)).toBe('1.53×');
    expect(fmtRatio(undefined)).toBe('—');
  });

  it('formats probabilities as integer percents', () => {
    expect(fmtPct(0.71)).toBe('71%');
    expect(fmtPct(0.045)).toBe('5%');
    expect(fmtPct(1)).toBe('100%');
    expect(fmtPct(undefined)).toBe('—');
  });

  it('formats unit-interval values and signed numbers', () => {
    expect(fmtUnit(0.18)).toBe('0.18');
    expect(fmtSigned(0.82)).toBe('+0.82');
    expect(fmtSigned(-0.44)).toBe('-0.44');
  });

  it('formats lead time and integers', () => {
    expect(fmtLeadTime(34)).toBe('34m');
    expect(fmtInt(1234)).toBe('1,234');
    expect(fmtInt(undefined)).toBe('—');
  });

  it('formats ISO timestamps as compact UTC', () => {
    expect(fmtTs('2026-06-28T13:32:00Z')).toBe('06-28 13:32Z');
    expect(fmtTs(undefined)).toBe('—');
    expect(fmtTs('not-a-date')).toBe('—');
  });

  it('humanizes snake_case enums', () => {
    expect(humanize('out_of_distribution')).toBe('Out Of Distribution');
    expect(humanize('short_gamma')).toBe('Short Gamma');
    expect(humanize('psi_feature_drift')).toBe('Psi Feature Drift');
  });
});
