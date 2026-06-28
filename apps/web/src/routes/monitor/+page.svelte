<script lang="ts">
  import Badge from '$lib/components/Badge.svelte';
  import PageHeader from '$lib/components/PageHeader.svelte';
  import DateRangePicker from '$lib/components/DateRangePicker.svelte';
  import { actionBadgeClass, actionSeverity, severityRailClass } from '$lib/ui/theme';
  import { fmtSigned, fmtTs, humanize } from '$lib/format';
  import type { PageData } from './$types';

  let { data }: { data: PageData } = $props();

  const highCount = $derived(
    data.events.filter((e) => actionSeverity(e.actionTaken) === 'high').length
  );
</script>

<PageHeader
  title="Adaptation Monitor"
  subtitle="Forward-adaptation detectors and the actions the engine took. Newest first."
>
  {#snippet actions()}
    <DateRangePicker />
    <span
      class="num rounded border px-2 py-1 text-xs {highCount > 0
        ? 'border-down/40 bg-down/10 text-down'
        : 'border-base-800 bg-base-900/60 text-base-300'}"
    >
      {highCount} high-severity
    </span>
  {/snippet}
</PageHeader>

<div class="flex flex-col gap-2">
  {#each data.events as e, i (e.ts + e.detector + i)}
    {@const sev = actionSeverity(e.actionTaken)}
    <article
      class="flex flex-col gap-2 rounded-lg border border-base-800 border-l-2 bg-base-900/40 p-3 {severityRailClass(
        sev
      )}"
    >
      <div class="flex flex-wrap items-center justify-between gap-2">
        <div class="flex flex-wrap items-center gap-2">
          <span class="num text-xs font-semibold text-base-100">{humanize(e.detector)}</span>
          {#if e.ticker}
            <Badge class="border-base-700 bg-base-800/60 text-base-300" mono>{e.ticker}</Badge>
          {/if}
          {#if e.strategyId}
            <span class="num text-[11px] text-base-500">{e.strategyId}</span>
          {/if}
        </div>
        <div class="flex items-center gap-2">
          <Badge class={actionBadgeClass(e.actionTaken)}>{e.actionTaken}</Badge>
          <span class="num text-[11px] text-base-500">{fmtTs(e.ts)}</span>
        </div>
      </div>

      <p class="text-xs leading-snug text-base-300">{e.detail}</p>

      {#if e.metricValue != null && e.threshold != null}
        <div class="flex items-center gap-2 text-[11px]">
          <span class="num rounded bg-base-800/60 px-1.5 py-0.5 text-base-200">
            metric {fmtSigned(e.metricValue)}
          </span>
          <span class="text-base-600">vs</span>
          <span class="num rounded bg-base-800/60 px-1.5 py-0.5 text-base-400">
            threshold {fmtSigned(e.threshold)}
          </span>
          <span
            class="num {e.metricValue > e.threshold ? 'text-down' : 'text-up'}"
            title="Direction of breach"
          >
            {e.metricValue > e.threshold ? '▲ over' : '▼ under'}
          </span>
        </div>
      {/if}
    </article>
  {/each}
</div>
