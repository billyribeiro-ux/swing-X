<script lang="ts">
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { SvelteURLSearchParams } from 'svelte/reactivity';
  import {
    RANGE_PRESETS,
    presetToRange,
    rangeFromSearchParams,
    rangeToPreset,
    type DateRange,
    type RangePreset
  } from '$lib/date-range';

  /**
   * Reusable date-window filter. Reflects the selected window into the URL as
   * `?from=YYYY-MM-DD&to=YYYY-MM-DD` so it is shareable and survives reload; the
   * page `load` reads the same params and passes them to the API client.
   *
   * Preset buttons (1M/3M/6M/1Y/YTD/All) plus two `<input type="date">` for a
   * custom range. Default is "All" — no params. State is derived from the URL,
   * not held locally, so back/forward navigation stays consistent.
   */

  // Derive the active range straight from the URL so it's the single source of truth.
  const range = $derived<DateRange>(rangeFromSearchParams(page.url.searchParams));
  const activePreset = $derived<RangePreset | null>(rangeToPreset(range));

  /** Push a new range into the URL, preserving other params and scroll/focus. */
  function apply(next: DateRange): void {
    // Start from the current params so unrelated ones survive; SvelteURLSearchParams is the
    // reactive primitive. Navigate to the current pathname + new query through `resolve()`
    // (base-path-correct, matching the rest of the app).
    const params = new SvelteURLSearchParams(page.url.searchParams);
    if (next.from) params.set('from', next.from);
    else params.delete('from');
    if (next.to) params.set('to', next.to);
    else params.delete('to');
    const qs = params.toString();
    const target = qs ? `${page.url.pathname}?${qs}` : page.url.pathname;
    // `page.url` is already resolved and the app configures no base path, so resolve() would be
    // a no-op here; the route id also isn't known statically in this reusable component. This is
    // a same-origin query-only update, so navigate to the current pathname directly.
    // eslint-disable-next-line svelte/no-navigation-without-resolve -- same-origin, already-resolved URL
    goto(target, {
      keepFocus: true,
      noScroll: true
    });
  }

  function selectPreset(preset: RangePreset): void {
    apply(presetToRange(preset));
  }

  function onFromInput(e: Event): void {
    const value = (e.currentTarget as HTMLInputElement).value;
    apply({ from: value || undefined, to: range.to });
  }

  function onToInput(e: Event): void {
    const value = (e.currentTarget as HTMLInputElement).value;
    apply({ from: range.from, to: value || undefined });
  }

  const presetBtnBase =
    'num rounded px-2 py-1 text-[11px] font-medium tracking-wide transition-colors';
  const inputClass =
    'num rounded border border-base-700 bg-base-900/60 px-1.5 py-1 text-[11px] text-base-200 ' +
    'focus:border-accent/60 focus:outline-none [color-scheme:dark]';
</script>

<div class="flex items-center gap-2" role="group" aria-label="Filter by decision date">
  <div class="flex items-center gap-0.5 rounded-md border border-base-800 bg-base-900/60 p-0.5">
    {#each RANGE_PRESETS as preset (preset)}
      <button
        type="button"
        aria-pressed={activePreset === preset}
        class="{presetBtnBase} {activePreset === preset
          ? 'bg-base-700/80 text-base-100'
          : 'text-base-400 hover:bg-base-800/60 hover:text-base-200'}"
        onclick={() => selectPreset(preset)}
      >
        {preset}
      </button>
    {/each}
  </div>

  <div class="flex items-center gap-1 text-[11px] text-base-500">
    <input
      type="date"
      class={inputClass}
      aria-label="From date"
      value={range.from ?? ''}
      max={range.to ?? undefined}
      oninput={onFromInput}
    />
    <span aria-hidden="true">–</span>
    <input
      type="date"
      class={inputClass}
      aria-label="To date"
      value={range.to ?? ''}
      min={range.from ?? undefined}
      oninput={onToInput}
    />
  </div>
</div>
