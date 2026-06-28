import { describe, expect, it } from 'vitest';
import {
  filterByRange,
  isWithinRange,
  presetToRange,
  rangeFromSearchParams,
  rangeToPreset,
  toDateParam
} from './date-range';

const NOW = new Date('2026-06-28T12:00:00Z');

describe('date-range helpers', () => {
  it('formats a UTC date param', () => {
    expect(toDateParam(new Date('2026-01-05T23:30:00Z'))).toBe('2026-01-05');
  });

  it('maps presets to inclusive windows ending today', () => {
    expect(presetToRange('All', NOW)).toEqual({});
    expect(presetToRange('1M', NOW)).toEqual({ from: '2026-05-28', to: '2026-06-28' });
    expect(presetToRange('3M', NOW)).toEqual({ from: '2026-03-28', to: '2026-06-28' });
    expect(presetToRange('1Y', NOW)).toEqual({ from: '2025-06-28', to: '2026-06-28' });
    expect(presetToRange('YTD', NOW)).toEqual({ from: '2026-01-01', to: '2026-06-28' });
  });

  it('round-trips a preset back from a range', () => {
    expect(rangeToPreset({}, NOW)).toBe('All');
    expect(rangeToPreset(presetToRange('6M', NOW), NOW)).toBe('6M');
    // a custom window that matches no preset
    expect(rangeToPreset({ from: '2026-02-14', to: '2026-02-15' }, NOW)).toBeNull();
  });

  it('reads a range from URL search params, omitting empties', () => {
    expect(rangeFromSearchParams(new URLSearchParams(''))).toEqual({});
    expect(rangeFromSearchParams(new URLSearchParams('from=2026-06-01&to=2026-06-30'))).toEqual({
      from: '2026-06-01',
      to: '2026-06-30'
    });
    expect(rangeFromSearchParams(new URLSearchParams('from=2026-06-01'))).toEqual({
      from: '2026-06-01'
    });
  });

  it('treats both bounds as inclusive on whole days', () => {
    const range = { from: '2026-06-01', to: '2026-06-30' };
    expect(isWithinRange('2026-06-01T00:00:00Z', range)).toBe(true);
    expect(isWithinRange('2026-06-30T23:59:59Z', range)).toBe(true); // same-day end is included
    expect(isWithinRange('2026-05-31T23:59:59Z', range)).toBe(false);
    expect(isWithinRange('2026-07-01T00:00:00Z', range)).toBe(false);
  });

  it('filterByRange keeps the list intact for an empty range', () => {
    const rows = [{ ts: '2020-01-01T00:00:00Z' }, { ts: '2030-01-01T00:00:00Z' }];
    expect(filterByRange(rows, {}, (r) => r.ts)).toHaveLength(2);
    expect(filterByRange(rows, { from: '2025-01-01' }, (r) => r.ts)).toEqual([
      { ts: '2030-01-01T00:00:00Z' }
    ]);
  });
});
