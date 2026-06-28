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

/**
 * Typed API client for the operator dashboard.
 *
 * Today every function resolves from local fixture data so the UI is fully
 * exercisable while the Rust (axum) API is still being built. Each function is
 * structured so the swap to a live fetch is a one-line change:
 *
 *   TODO(api): replace `return clone(fixture)` with
 *     `return parsed(<schema>, await getJson('/v1/...'))`
 *   once PUBLIC_API_BASE points at a running backend. The zod schemas in
 *   ./schemas.ts already validate the live payloads at the trust boundary.
 *
 * `PUBLIC_API_BASE` is read via `$env/dynamic/public` so it can be set at
 * runtime (adapter-node) without a rebuild.
 */

/** Base URL of the Rust API, or '' when running on fixtures only. */
export const API_BASE: string = env.PUBLIC_API_BASE ?? '';

/** True when a backend base URL is configured. Drives a "fixture mode" banner. */
export const usingLiveApi: boolean = API_BASE.length > 0;

/**
 * Fetch + parse helper for when the live API is wired in. Currently unused by the
 * fixture path but kept here (and exported) so the migration is mechanical.
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

// ---------------------------------------------------------------------------
// Public API surface
// ---------------------------------------------------------------------------

export function getSignals(): Promise<Signal[]> {
  // TODO(api): return parsed(signalsSchema, await getJson('/v1/signals'))
  void signalsSchema;
  return settle(clone(signalFixtures));
}

export function getSignal(id: string): Promise<Signal | null> {
  // TODO(api): return parsed(signalSchema, await getJson(`/v1/signals/${id}`))
  void signalSchema;
  const found = signalFixtures.find((s) => s.signalId === id) ?? null;
  return settle(found ? clone(found) : null);
}

export function getPopulation(): Promise<Strategy[]> {
  // TODO(api): return parsed(populationSchema, await getJson('/v1/population'))
  void populationSchema;
  return settle(clone(populationFixtures));
}

export function getMonitorEvents(): Promise<MonitorEvent[]> {
  // TODO(api): return parsed(monitorEventsSchema, await getJson('/v1/monitor'))
  void monitorEventsSchema;
  return settle(clone(monitorFixtures));
}

export function getJournal(): Promise<Trade[]> {
  // TODO(api): return parsed(journalSchema, await getJson('/v1/journal'))
  void journalSchema;
  return settle(clone(journalFixtures));
}

export function getChangelog(): Promise<ChangelogWeek[]> {
  // TODO(api): return parsed(changelogSchema, await getJson('/v1/changelog'))
  void changelogSchema;
  return settle(clone(changelogFixtures));
}
