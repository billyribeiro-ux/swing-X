import { describe, expect, it } from 'vitest';
import { signalSchema, populationSchema } from './schemas';
import {
  getChangelog,
  getJournal,
  getMonitorEvents,
  getPopulation,
  getSignal,
  getSignals
} from './client';

describe('fixture-backed API client', () => {
  it('returns signals that satisfy the signal schema', async () => {
    const signals = await getSignals();
    expect(signals.length).toBeGreaterThan(0);
    for (const s of signals) {
      expect(() => signalSchema.parse(s)).not.toThrow();
    }
  });

  it('resolves a single signal by id and null for misses', async () => {
    const all = await getSignals();
    const first = all[0];
    const found = await getSignal(first.signalId);
    expect(found?.signalId).toBe(first.signalId);
    expect(await getSignal('does-not-exist')).toBeNull();
  });

  it('returns a population that satisfies the schema and omits win_rate', async () => {
    const pop = await getPopulation();
    expect(() => populationSchema.parse(pop)).not.toThrow();
    // win rate is a banned selection metric — it must not leak into the DTO.
    for (const strat of pop) {
      expect(strat).not.toHaveProperty('winRate');
      expect(strat.latestScore ?? {}).not.toHaveProperty('winRate');
    }
  });

  it('clones fixtures so callers cannot mutate shared state', async () => {
    const a = await getSignals();
    a[0].entry = -999;
    const b = await getSignals();
    expect(b[0].entry).not.toBe(-999);
  });

  it('provides monitor events, journal trades, and changelog weeks', async () => {
    expect((await getMonitorEvents()).length).toBeGreaterThan(0);
    expect((await getJournal()).length).toBeGreaterThan(0);
    expect((await getChangelog()).length).toBeGreaterThan(0);
  });

  it('filters fixtures by an inclusive date range in fixture mode', async () => {
    const all = await getSignals();
    // A window in the distant past matches nothing; an unbounded range matches all.
    const none = await getSignals(undefined, { from: '2000-01-01', to: '2000-12-31' });
    expect(none).toHaveLength(0);
    const same = await getSignals(undefined, {});
    expect(same).toHaveLength(all.length);
    // A window covering the fixtures keeps only the in-range rows.
    const ranged = await getSignals(undefined, { from: '2026-06-01', to: '2026-06-30' });
    expect(ranged.length).toBeGreaterThan(0);
    expect(ranged.length).toBeLessThanOrEqual(all.length);
    for (const s of ranged) {
      expect(s.decisionTs.slice(0, 10) >= '2026-06-01').toBe(true);
      expect(s.decisionTs.slice(0, 10) <= '2026-06-30').toBe(true);
    }
  });

  it('filters journal and monitor fixtures by range too', async () => {
    const trades = await getJournal(undefined, { from: '2000-01-01', to: '2000-12-31' });
    expect(trades).toHaveLength(0);
    const events = await getMonitorEvents(undefined, { from: '2000-01-01', to: '2000-12-31' });
    expect(events).toHaveLength(0);
  });
});
