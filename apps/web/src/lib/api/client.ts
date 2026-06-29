import { env } from '$env/dynamic/public';
import type { ChangelogWeek, MonitorEvent, Signal, Strategy, Trade } from '@swing-x/shared-types';
import { signalFixtures } from '$lib/fixtures/signals';
import { populationFixtures } from '$lib/fixtures/population';
import { monitorFixtures } from '$lib/fixtures/monitor';
import { journalFixtures } from '$lib/fixtures/journal';
import { changelogFixtures } from '$lib/fixtures/changelog';
import { filterByRange, type DateRange } from '$lib/date-range';
import {
  changelogSchema,
  journalSchema,
  monitorEventsSchema,
  populationSchema,
  signalSchema,
  signalsSchema
} from './schemas';
import type { z } from 'zod';

/**
 * Typed API client for the operator dashboard.
 *
 * When `PUBLIC_API_BASE` is set, each getter fetches `${PUBLIC_API_BASE}/api/...`,
 * validates the payload with the zod schema in ./schemas.ts at the trust boundary,
 * and returns it. When `PUBLIC_API_BASE` is unset — or any fetch/parse step fails —
 * it transparently falls back to the bundled fixtures so the UI always renders.
 *
 * `PUBLIC_API_BASE` is read via `$env/dynamic/public` so it can be set at runtime
 * (adapter-node) without a rebuild.
 */

/** Base URL of the Rust API, or '' when running on fixtures only. */
export const API_BASE: string = env.PUBLIC_API_BASE ?? '';

/** True when a backend base URL is configured. Drives a "fixture mode" banner. */
export const usingLiveApi: boolean = API_BASE.length > 0;

/**
 * Fetch + JSON helper. Throws on a non-2xx response so the caller falls back to
 * fixtures. `fetchFn` lets SvelteKit loaders pass their instrumented `fetch`.
 *
 * @internal
 */
export async function getJson(path: string, fetchFn: typeof fetch = fetch): Promise<unknown> {
  const res = await fetchFn(`${API_BASE}${path}`, {
    headers: { accept: 'application/json' }
  });
  if (!res.ok) {
    throw new Error(`API ${path} failed: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

/** Structured-clone a fixture so callers never mutate shared module state. */
function clone<T>(value: T): T {
  return structuredClone(value);
}

/** Simulate async I/O so loaders behave like the eventual network calls. */
function settle<T>(value: T): Promise<T> {
  return Promise.resolve(value);
}

/** Which scanner a request targets. Absent/`'etf'` selects the default ETF scanner. */
export type Scanner = 'etf' | 'equity';

/**
 * Append the `?from=&to=&scanner=` query to a path. Only present bounds are
 * emitted, and `scanner` is only emitted when it is the non-default `'equity'`,
 * so a bare ETF request leaves the path unchanged (backend stays unfiltered and
 * on the default scanner).
 */
function withParams(path: string, range?: DateRange, scanner?: Scanner): string {
  const params = new URLSearchParams();
  if (range?.from) params.set('from', range.from);
  if (range?.to) params.set('to', range.to);
  if (scanner === 'equity') params.set('scanner', scanner);
  const qs = params.toString();
  return qs ? `${path}?${qs}` : path;
}

/**
 * Live-or-fixture resolver with an optional date window.
 *
 * Live mode: appends `from`/`to` to the request path so the Rust API does the
 * filtering, then parses the body with `schema`. On any error (network, non-2xx,
 * or schema mismatch) it logs a warning and falls back to the range-filtered
 * fixture.
 *
 * Fixture mode (no `PUBLIC_API_BASE`): filters the bundled fixture client-side by
 * `range` (via `dateKey`) so the picker still visibly works offline. When no
 * `dateKey` is given the fixture is returned as-is.
 */
async function resolve<T extends object>(
  path: string,
  schema: z.ZodType<T[]>,
  fixture: T[],
  fetchFn: typeof fetch = fetch,
  range?: DateRange,
  dateKey?: (row: T) => string,
  scanner?: Scanner
): Promise<T[]> {
  if (!usingLiveApi) {
    const rows = dateKey && range ? filterByRange(fixture, range, dateKey) : fixture;
    return settle(clone(rows));
  }
  try {
    const raw = await getJson(withParams(path, range, scanner), fetchFn);
    return schema.parse(raw);
  } catch (err) {
    console.warn(`[api] falling back to fixtures for ${path}:`, err);
    const rows = dateKey && range ? filterByRange(fixture, range, dateKey) : fixture;
    return clone(rows);
  }
}

// ---------------------------------------------------------------------------
// Public API surface
// ---------------------------------------------------------------------------

export function getSignals(
  fetchFn?: typeof fetch,
  range?: DateRange,
  scanner?: Scanner
): Promise<Signal[]> {
  return resolve(
    '/api/signals',
    signalsSchema,
    signalFixtures,
    fetchFn,
    range,
    (s) => s.decisionTs,
    scanner
  );
}

export async function getSignal(
  id: string,
  fetchFn?: typeof fetch,
  scanner?: Scanner
): Promise<Signal | null> {
  if (!usingLiveApi) {
    const found = signalFixtures.find((s) => s.signalId === id) ?? null;
    return settle(found ? clone(found) : null);
  }
  try {
    const raw = await getJson(withParams(`/api/signals/${id}`, undefined, scanner), fetchFn);
    return signalSchema.parse(raw);
  } catch (err) {
    // Distinguish "backend says 404" from a transient/parse failure: on a 404 the
    // signal genuinely does not exist, so surface null rather than a stale fixture.
    if (err instanceof Error && /\b404\b/.test(err.message)) {
      return null;
    }
    console.warn(`[api] falling back to fixtures for /api/signals/${id}:`, err);
    const found = signalFixtures.find((s) => s.signalId === id) ?? null;
    return found ? clone(found) : null;
  }
}

export function getPopulation(
  fetchFn?: typeof fetch,
  range?: DateRange,
  scanner?: Scanner
): Promise<Strategy[]> {
  // Window strategies by their freshness timestamp — the latest OOS score's
  // `evaluatedAt` when present (matching the backend's COALESCE on `created_at`).
  return resolve(
    '/api/population',
    populationSchema,
    populationFixtures,
    fetchFn,
    range,
    (s) => s.latestScore?.evaluatedAt ?? '',
    scanner
  );
}

export function getMonitorEvents(
  fetchFn?: typeof fetch,
  range?: DateRange,
  scanner?: Scanner
): Promise<MonitorEvent[]> {
  return resolve(
    '/api/monitor',
    monitorEventsSchema,
    monitorFixtures,
    fetchFn,
    range,
    (e) => e.ts,
    scanner
  );
}

export function getJournal(
  fetchFn?: typeof fetch,
  range?: DateRange,
  scanner?: Scanner
): Promise<Trade[]> {
  return resolve(
    '/api/journal',
    journalSchema,
    journalFixtures,
    fetchFn,
    range,
    (t) => t.entryTs,
    scanner
  );
}

export function getChangelog(fetchFn?: typeof fetch): Promise<ChangelogWeek[]> {
  return resolve('/api/changelog', changelogSchema, changelogFixtures, fetchFn);
}
