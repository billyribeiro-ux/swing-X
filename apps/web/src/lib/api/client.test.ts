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
});
