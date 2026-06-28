/**
 * Display formatters for the operator console. All numeric formatters return
 * strings intended to render with the `.num` (tabular monospace) class.
 */

/** Format a price with a fixed 2 decimals. */
export function fmtPrice(n: number | undefined | null): string {
  if (n == null || Number.isNaN(n)) return '—';
  return n.toFixed(2);
}

/** Format an R-multiple, signed, 2 decimals (e.g. "+0.42R", "-1.60R"). */
export function fmtR(n: number | undefined | null): string {
  if (n == null || Number.isNaN(n)) return '—';
  const sign = n > 0 ? '+' : '';
  return `${sign}${n.toFixed(2)}R`;
}

/** Format a ratio with 2 decimals and an "×" suffix (e.g. R:R, profit factor). */
export function fmtRatio(n: number | undefined | null): string {
  if (n == null || Number.isNaN(n)) return '—';
  return `${n.toFixed(2)}×`;
}

/** Format a [0,1] probability as a percentage with no decimals (e.g. "71%"). */
export function fmtPct(n: number | undefined | null): string {
  if (n == null || Number.isNaN(n)) return '—';
  return `${Math.round(n * 100)}%`;
}

/** Format a unit-interval value with 2 decimals (e.g. PBO "0.18"). */
export function fmtUnit(n: number | undefined | null): string {
  if (n == null || Number.isNaN(n)) return '—';
  return n.toFixed(2);
}

/** Format a signed number with 2 decimals (e.g. DSR "+0.82"). */
export function fmtSigned(n: number | undefined | null): string {
  if (n == null || Number.isNaN(n)) return '—';
  const sign = n > 0 ? '+' : '';
  return `${sign}${n.toFixed(2)}`;
}

/** Format minutes of lead time (e.g. "34m"). */
export function fmtLeadTime(n: number | undefined | null): string {
  if (n == null || Number.isNaN(n)) return '—';
  return `${Math.round(n)}m`;
}

/** Format an integer with thousands separators. */
export function fmtInt(n: number | undefined | null): string {
  if (n == null || Number.isNaN(n)) return '—';
  return n.toLocaleString('en-US');
}

/** Compact UTC datetime: "06-28 13:32Z". */
export function fmtTs(iso: string | undefined | null): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  const pad = (x: number) => String(x).padStart(2, '0');
  return `${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(
    d.getUTCMinutes()
  )}Z`;
}

/** Human-readable enum label: snake_case -> "Title Case". */
export function humanize(s: string): string {
  return s
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}
