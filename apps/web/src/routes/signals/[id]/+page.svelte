<script lang="ts">
  import { browser } from '$app/environment';
  import { resolve } from '$app/paths';
  import { ArrowLeft } from 'phosphor-svelte';
  import SignalChart from '$lib/components/SignalChart.svelte';
  import WhyPanel from '$lib/components/WhyPanel.svelte';
  import JsonInspector from '$lib/components/JsonInspector.svelte';
  import Badge from '$lib/components/Badge.svelte';
  import Metric from '$lib/components/Metric.svelte';
  import ConvictionBar from '$lib/components/ConvictionBar.svelte';
  import { sideBadgeClass } from '$lib/ui/theme';
  import { fmtInt, fmtLeadTime, fmtPrice, fmtR, fmtRatio, fmtTs, humanize } from '$lib/format';
  import type { PageData } from './$types';

  let { data }: { data: PageData } = $props();
  const s = $derived(data.signal);
</script>

<div class="flex items-center gap-2">
  <a
    href={resolve('/scoreboard')}
    class="flex items-center gap-1 text-xs text-base-400 hover:text-base-100"
  >
    <ArrowLeft size={13} weight="bold" />
    <span>Scoreboard</span>
  </a>
</div>

<header class="flex flex-wrap items-center justify-between gap-3 border-b border-base-800 pb-3">
  <div class="flex items-center gap-3">
    <span class="num text-2xl font-bold tracking-tight text-base-100">{s.ticker}</span>
    <Badge class={sideBadgeClass(s.side)}>{s.side}</Badge>
    <Badge class="border-base-700 bg-base-800/60 text-base-300">{humanize(s.horizon)}</Badge>
    <span class="num text-xs text-base-500">{s.signalId}</span>
  </div>
  <div class="flex items-center gap-2 text-xs text-base-400">
    <span class="num">{fmtTs(s.decisionTs)}</span>
    <span class="text-base-700">·</span>
    <span class="num">{s.strategyId}</span>
  </div>
</header>

<!-- Top metric strip -->
<div class="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
  <Metric label="Entry" value={fmtPrice(s.entry)} />
  <Metric label="Stop" value={fmtPrice(s.stop)} valueClass="text-down" />
  <Metric label="Target 1" value={fmtPrice(s.target1)} valueClass="text-up" sub={fmtRatio(s.rr1)} />
  <Metric
    label="Target 2"
    value={fmtPrice(s.target2)}
    valueClass="text-up-dim"
    sub={s.rr2 != null ? fmtRatio(s.rr2) : undefined}
  />
  <Metric label="Cohort n" value={fmtInt(s.cohortN)} />
  <Metric label="Lead time" value={fmtLeadTime(s.leadTime)} />
</div>

<div class="grid grid-cols-1 gap-4 lg:grid-cols-3">
  <!-- Chart -->
  <section class="flex flex-col gap-2 lg:col-span-2">
    {#if browser}
      <SignalChart
        candles={data.candles}
        vwap={data.vwap}
        side={s.side}
        entry={s.entry}
        stop={s.stop}
        target1={s.target1}
        target2={s.target2}
      />
    {:else}
      <div
        class="flex h-[420px] w-full items-center justify-center rounded-lg border border-base-800 bg-base-950/40 text-xs text-base-500"
      >
        Loading chart…
      </div>
    {/if}
    <div class="flex flex-wrap items-center gap-3 px-1 text-[11px] text-base-400">
      <span class="flex items-center gap-1"
        ><span class="inline-block h-2 w-3 bg-up"></span> Up</span
      >
      <span class="flex items-center gap-1"
        ><span class="inline-block h-2 w-3 bg-down"></span> Down</span
      >
      <span class="flex items-center gap-1"
        ><span class="inline-block h-0.5 w-4 bg-info"></span> VWAP</span
      >
      <span class="flex items-center gap-1"
        ><span class="inline-block h-0.5 w-4 bg-up"></span> Targets</span
      >
      <span class="flex items-center gap-1"
        ><span class="inline-block h-0.5 w-4 bg-down"></span> Stop</span
      >
    </div>
  </section>

  <!-- Right column: conviction + cohort + why -->
  <aside class="flex flex-col gap-4">
    <div class="rounded-lg border border-base-800 bg-base-900/40 p-3">
      <div class="mb-2 flex items-center justify-between">
        <span class="text-[10px] font-semibold tracking-wider text-base-400 uppercase">
          Calibrated conviction
        </span>
      </div>
      <ConvictionBar value={s.conviction} />
      <p class="mt-2 text-[11px] leading-snug text-base-400">{s.regimeDesc}</p>
    </div>

    <div class="grid grid-cols-2 gap-2">
      <Metric
        label="Cohort exp"
        value={fmtR(s.cohortExpectancy)}
        valueClass={(s.cohortExpectancy ?? 0) >= 0 ? 'text-up' : 'text-down'}
      />
      <Metric label="CVaR 5%" value={fmtR(s.cvar5)} valueClass="text-down" />
    </div>

    <div
      class="rounded-lg border border-down/30 bg-down/5 p-3"
      title="Hard invalidation — voids the thesis if hit"
    >
      <span class="text-[10px] font-semibold tracking-wider text-down uppercase">
        Hard invalidation
      </span>
      <p class="mt-1 text-xs leading-snug text-base-200">{s.invalidation}</p>
    </div>
  </aside>
</div>

<!-- Why panel + JSON -->
<div class="grid grid-cols-1 gap-4 lg:grid-cols-3">
  <section class="rounded-lg border border-base-800 bg-base-900/40 p-4 lg:col-span-2">
    <h2 class="mb-3 text-sm font-semibold text-base-100">Why — driver attribution</h2>
    <WhyPanel drivers={s.why} />
  </section>
  <div class="flex flex-col gap-3 lg:col-span-1">
    <JsonInspector value={s.payloadJson} />
  </div>
</div>
