import { env } from '$env/dynamic/public';
import type { ChangelogWeek, MonitorEvent, Signal, Strategy, Trade } from '@swing-x/shared-types';
import { signalFixtures } from '$lib/fixtures/signals';
import { populationFixtures } from '$lib/fixtures/population';
import { monitorFixtures } from '$lib/fixtures/monitor';
import { journalFixtures } from '$lib/fixtures/journal';
import { changelogFixtures } from '$lib/fixtures/changelog';
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

/**
 * Live-or-fixture resolver. When a backend is configured it fetches `path`, parses
 * the body with `schema`, and returns the result; on any error (network, non-2xx,
 * or schema mismatch) it logs a warning and returns the cloned `fixture`. When no
 * backend is configured it returns the fixture directly.
 */
async function resolve<T>(
  path: string,
  schema: z.ZodType<T>,
  fixture: T,
  fetchFn: typeof fetch = fetch
): Promise<T> {
  if (!usingLiveApi) {
    return settle(clone(fixture));
  }
  try {
    const raw = await getJson(path, fetchFn);
    return schema.parse(raw);
  } catch (err) {
    console.warn(`[api] falling back to fixtures for ${path}:`, err);
    return clone(fixture);
  }
}

// ---------------------------------------------------------------------------
// Public API surface
// ---------------------------------------------------------------------------

export function getSignals(fetchFn?: typeof fetch): Promise<Signal[]> {
  return resolve('/api/signals', signalsSchema, signalFixtures, fetchFn);
}

export async function getSignal(id: string, fetchFn?: typeof fetch): Promise<Signal | null> {
  if (!usingLiveApi) {
    const found = signalFixtures.find((s) => s.signalId === id) ?? null;
    return settle(found ? clone(found) : null);
  }
  try {
    const raw = await getJson(`/api/signals/${id}`, fetchFn);
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

export function getPopulation(fetchFn?: typeof fetch): Promise<Strategy[]> {
  return resolve('/api/population', populationSchema, populationFixtures, fetchFn);
}

export function getMonitorEvents(fetchFn?: typeof fetch): Promise<MonitorEvent[]> {
  return resolve('/api/monitor', monitorEventsSchema, monitorFixtures, fetchFn);
}

export function getJournal(fetchFn?: typeof fetch): Promise<Trade[]> {
  return resolve('/api/journal', journalSchema, journalFixtures, fetchFn);
}

export function getChangelog(fetchFn?: typeof fetch): Promise<ChangelogWeek[]> {
  return resolve('/api/changelog', changelogSchema, changelogFixtures, fetchFn);
}
