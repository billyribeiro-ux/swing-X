/**
 * Date-window helpers shared by the {@link DateRangePicker} and the API client.
 *
 * A "range" is an inclusive `{ from?, to? }` pair of `YYYY-MM-DD` calendar-date
 * strings (matching the `from`/`to` query params the Rust API accepts). Either
 * bound may be absent; an all-absent range means "no filter" (the default).
 */

/** An inclusive date window. Bounds are `YYYY-MM-DD`; absent = unbounded. */
export interface DateRange {
  from?: string;
  to?: string;
}

/** The preset windows offered by the picker, in display order. */
export const RANGE_PRESETS = ['1M', '3M', '6M', '1Y', 'YTD', 'All'] as const;
export type RangePreset = (typeof RANGE_PRESETS)[number];

/** Format a `Date` as a UTC `YYYY-MM-DD` calendar-date string. */
export function toDateParam(d: Date): string {
  const pad = (x: number) => String(x).padStart(2, '0');
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`;
}

/**
 * Resolve a preset into a concrete range, relative to `now` (default: today).
 * "All" yields an empty range (no params). All other presets set `from` to the
 * window start and `to` to today, both inclusive `YYYY-MM-DD`.
 */
export function presetToRange(preset: RangePreset, now: Date = new Date()): DateRange {
  if (preset === 'All') return {};
  const to = toDateParam(now);
  const start = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
  switch (preset) {
    case '1M':
      start.setUTCMonth(start.getUTCMonth() - 1);
      break;
    case '3M':
      start.setUTCMonth(start.getUTCMonth() - 3);
      break;
    case '6M':
      start.setUTCMonth(start.getUTCMonth() - 6);
      break;
    case '1Y':
      start.setUTCFullYear(start.getUTCFullYear() - 1);
      break;
    case 'YTD':
      start.setUTCMonth(0, 1);
      break;
  }
  return { from: toDateParam(start), to };
}

/**
 * Identify which preset (if any) a range corresponds to, relative to `now`.
 * Returns `'All'` for an empty range and `null` for a custom range that matches
 * no preset, so the picker can highlight the active preset button.
 */
export function rangeToPreset(range: DateRange, now: Date = new Date()): RangePreset | null {
  if (!range.from && !range.to) return 'All';
  for (const preset of RANGE_PRESETS) {
    if (preset === 'All') continue;
    const candidate = presetToRange(preset, now);
    if (candidate.from === range.from && candidate.to === range.to) return preset;
  }
  return null;
}

/**
 * Read a {@link DateRange} from a URL's search params. Empty/absent params are
 * omitted so the result is `{}` (no filter) on a bare URL.
 */
export function rangeFromSearchParams(params: URLSearchParams): DateRange {
  const range: DateRange = {};
  const from = params.get('from');
  const to = params.get('to');
  if (from) range.from = from;
  if (to) range.to = to;
  return range;
}

/**
 * Lower-bound a value against the window's `from` (inclusive). The `from` param
 * is a date; a same-day timestamp passes by comparing against `from`'s midnight.
 * Comparison is lexical on the ISO string prefix, which is correct for RFC-3339
 * UTC timestamps (`YYYY-MM-DDT...`) vs a `YYYY-MM-DD` bound.
 */
function afterFrom(iso: string, from: string): boolean {
  return iso.slice(0, 10) >= from;
}

/** Upper-bound a value against the window's `to` (inclusive, whole day). */
function beforeTo(iso: string, to: string): boolean {
  return iso.slice(0, 10) <= to;
}

/**
 * True when an ISO timestamp falls inside the inclusive window. An empty range
 * matches everything. Used for client-side fixture filtering so the picker still
 * visibly works offline (fixture mode).
 */
export function isWithinRange(iso: string, range: DateRange): boolean {
  if (range.from && !afterFrom(iso, range.from)) return false;
  if (range.to && !beforeTo(iso, range.to)) return false;
  return true;
}

/**
 * Filter a list by the timestamp returned by `key`, keeping rows whose time is
 * inside the inclusive window. An empty range returns the list unchanged.
 */
export function filterByRange<T>(
  rows: readonly T[],
  range: DateRange,
  key: (row: T) => string
): T[] {
  if (!range.from && !range.to) return rows.slice();
  return rows.filter((row) => isWithinRange(key(row), range));
}
