import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

/**
 * Live-API code-path tests. These mock `$env/dynamic/public` so `PUBLIC_API_BASE`
 * is set BEFORE the client module is imported (the base URL is captured at module
 * load), then stub `globalThis.fetch`. The default fixture-mode behaviour is
 * covered separately in ./client.test.ts.
 */

const API_BASE = 'http://api.test';

// Mock the SvelteKit dynamic-env module to enable live mode.
vi.mock('$env/dynamic/public', () => ({
  env: { PUBLIC_API_BASE: API_BASE }
}));

function jsonResponse(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    statusText: ok ? 'OK' : 'Error',
    json: async () => body
  } as Response;
}

describe('live-API client', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('fetches and zod-parses signals from the backend', async () => {
    const live = [
      {
        signalId: 'sig_live_1',
        strategyId: 'str_live',
        ticker: 'SPY',
        side: 'long',
        decisionTs: '2026-06-28T13:32:00Z',
        horizon: 'swing',
        entry: 548.2,
        stop: 543.1,
        target1: 556.0,
        conviction: 0.71,
        cohortN: 184,
        regimeDesc: 'short gamma',
        why: [{ layer: 'regime', key: 'gex', contribution: 0.3, detail: 'd' }],
        invalidation: 'close < 543',
        payloadJson: { raw: true }
      }
    ];
    const fetchMock = vi.fn(async () => jsonResponse(live));
    vi.stubGlobal('fetch', fetchMock);

    const { getSignals, usingLiveApi } = await import('./client');
    expect(usingLiveApi).toBe(true);

    const signals = await getSignals();
    expect(fetchMock).toHaveBeenCalledWith(
      `${API_BASE}/api/signals`,
      expect.objectContaining({ headers: { accept: 'application/json' } })
    );
    expect(signals).toHaveLength(1);
    expect(signals[0].signalId).toBe('sig_live_1');
  });

  it('appends from/to query params to the live request', async () => {
    const fetchMock = vi.fn(async () => jsonResponse([]));
    vi.stubGlobal('fetch', fetchMock);

    const { getSignals } = await import('./client');
    await getSignals(undefined, { from: '2026-06-01', to: '2026-06-30' });
    expect(fetchMock).toHaveBeenCalledWith(
      `${API_BASE}/api/signals?from=2026-06-01&to=2026-06-30`,
      expect.objectContaining({ headers: { accept: 'application/json' } })
    );
  });

  it('omits absent bounds and the query string entirely when unbounded', async () => {
    const fetchMock = vi.fn(async () => jsonResponse([]));
    vi.stubGlobal('fetch', fetchMock);

    const { getJournal } = await import('./client');
    await getJournal(undefined, { from: '2026-06-01' });
    expect(fetchMock).toHaveBeenLastCalledWith(
      `${API_BASE}/api/journal?from=2026-06-01`,
      expect.anything()
    );

    await getJournal(undefined, {});
    expect(fetchMock).toHaveBeenLastCalledWith(`${API_BASE}/api/journal`, expect.anything());
  });

  it('falls back to fixtures when the backend errors', async () => {
    const fetchMock = vi.fn(async () => jsonResponse(null, false, 500));
    vi.stubGlobal('fetch', fetchMock);
    vi.spyOn(console, 'warn').mockImplementation(() => {});

    const { getPopulation } = await import('./client');
    const pop = await getPopulation();
    // fixtures are non-empty, so a fallback yields data rather than throwing.
    expect(pop.length).toBeGreaterThan(0);
  });

  it('falls back to fixtures when the payload fails schema validation', async () => {
    // missing required fields -> zod throws -> fallback.
    const fetchMock = vi.fn(async () => jsonResponse([{ nope: true }]));
    vi.stubGlobal('fetch', fetchMock);
    vi.spyOn(console, 'warn').mockImplementation(() => {});

    const { getMonitorEvents } = await import('./client');
    const events = await getMonitorEvents();
    expect(events.length).toBeGreaterThan(0); // fixtures
  });

  it('returns null (not a fixture) when the backend 404s a single signal', async () => {
    const fetchMock = vi.fn(async () => jsonResponse(null, false, 404));
    vi.stubGlobal('fetch', fetchMock);
    vi.spyOn(console, 'warn').mockImplementation(() => {});

    const { getSignal } = await import('./client');
    const sig = await getSignal('does-not-exist');
    expect(sig).toBeNull();
  });
});
