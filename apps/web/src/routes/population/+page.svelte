<script lang="ts">
  import type { Strategy } from '@swing-x/shared-types';
  import DataTable from '$lib/components/DataTable.svelte';
  import type { Column } from '$lib/components/data-table';
  import Badge from '$lib/components/Badge.svelte';
  import GatePill from '$lib/components/GatePill.svelte';
  import PageHeader from '$lib/components/PageHeader.svelte';
  import { statusBadgeClass } from '$lib/ui/theme';
  import { fmtR, fmtRatio, fmtSigned, fmtUnit, humanize } from '$lib/format';
  import type { PageData } from './$types';

  let { data }: { data: PageData } = $props();

  const promotedCount = $derived(data.population.filter((p) => p.status === 'promoted').length);

  const columns: Column<Strategy>[] = [
    { id: 'strategyId', header: 'Strategy', sortValue: (r) => r.strategyId, cell: strategy },
    { id: 'horizon', header: 'Horizon', sortValue: (r) => r.horizon, cell: horizon },
    { id: 'status', header: 'Status', sortValue: (r) => r.status, cell: status },
    { id: 'generation', header: 'Gen', numeric: true, sortValue: (r) => r.generation, cell: gen },
    {
      id: 'dsr',
      header: 'DSR',
      numeric: true,
      sortValue: (r) => r.latestScore?.dsr ?? -Infinity,
      cell: dsr
    },
    {
      id: 'pbo',
      header: 'PBO',
      numeric: true,
      sortValue: (r) => r.latestScore?.pbo ?? Infinity,
      cell: pbo
    },
    {
      id: 'oosExp',
      header: 'OOS Exp',
      numeric: true,
      sortValue: (r) => r.latestScore?.oosExpectancyCostAware ?? -Infinity,
      cell: oosExp
    },
    {
      id: 'pf',
      header: 'PF',
      numeric: true,
      sortValue: (r) => r.latestScore?.profitFactor ?? 0,
      cell: pf
    },
    {
      id: 'cvar5',
      header: 'CVaR 5%',
      numeric: true,
      sortValue: (r) => r.latestScore?.cvar5 ?? 0,
      cell: cvar
    },
    {
      id: 'mar',
      header: 'MAR',
      numeric: true,
      sortValue: (r) => r.latestScore?.mar ?? -Infinity,
      cell: mar
    },
    {
      id: 'nReg',
      header: '#Reg+',
      numeric: true,
      sortValue: (r) => r.latestScore?.nRegimesPositive ?? 0,
      cell: nReg
    },
    {
      id: 'gate',
      header: 'Gate',
      sortValue: (r) => (r.latestScore?.passedGate ? 1 : 0),
      cell: gate
    }
  ];
</script>

{#snippet strategy(r: Strategy)}
  <div class="flex flex-col">
    <span class="num text-xs font-semibold text-base-100">{r.strategyId}</span>
    <span class="line-clamp-1 text-[11px] text-base-400" title={r.genomeSummary}>
      {r.genomeSummary}
    </span>
  </div>
{/snippet}

{#snippet horizon(r: Strategy)}
  <span class="text-xs text-base-300">{humanize(r.horizon)}</span>
{/snippet}

{#snippet status(r: Strategy)}
  <Badge class={statusBadgeClass(r.status)}>{r.status}</Badge>
{/snippet}

{#snippet gen(r: Strategy)}
  <span class="text-base-300">{r.generation}</span>
{/snippet}

{#snippet dsr(r: Strategy)}
  {#if r.latestScore}
    {@const v = r.latestScore.dsr}
    <span
      class="rounded px-1 font-semibold {v > 0 ? 'bg-up/15 text-up' : 'bg-down/15 text-down'}"
      title="Deflated Sharpe Ratio — healthy > 0"
    >
      {fmtSigned(v)}
    </span>
  {:else}
    <span class="text-base-500">—</span>
  {/if}
{/snippet}

{#snippet pbo(r: Strategy)}
  {#if r.latestScore}
    {@const v = r.latestScore.pbo}
    <span
      class="rounded px-1 font-semibold {v < 0.5 ? 'bg-up/15 text-up' : 'bg-down/15 text-down'}"
      title="Probability of Backtest Overfit — healthy < 0.50"
    >
      {fmtUnit(v)}
    </span>
  {:else}
    <span class="text-base-500">—</span>
  {/if}
{/snippet}

{#snippet oosExp(r: Strategy)}
  {#if r.latestScore}
    {@const v = r.latestScore.oosExpectancyCostAware}
    <span class={v >= 0 ? 'text-up' : 'text-down'}>{fmtR(v)}</span>
  {:else}
    <span class="text-base-500">—</span>
  {/if}
{/snippet}

{#snippet pf(r: Strategy)}
  {#if r.latestScore}
    {@const v = r.latestScore.profitFactor}
    <span class={v >= 1 ? 'text-base-200' : 'text-down'}>{fmtRatio(v)}</span>
  {:else}
    <span class="text-base-500">—</span>
  {/if}
{/snippet}

{#snippet cvar(r: Strategy)}
  {#if r.latestScore}
    <span class="text-down">{fmtR(r.latestScore.cvar5)}</span>
  {:else}
    <span class="text-base-500">—</span>
  {/if}
{/snippet}

{#snippet mar(r: Strategy)}
  {#if r.latestScore}
    {@const v = r.latestScore.mar}
    <span class={v >= 1 ? 'text-up' : v >= 0 ? 'text-base-300' : 'text-down'}>{fmtUnit(v)}</span>
  {:else}
    <span class="text-base-500">—</span>
  {/if}
{/snippet}

{#snippet nReg(r: Strategy)}
  <span class="text-base-300">{r.latestScore?.nRegimesPositive ?? '—'}</span>
{/snippet}

{#snippet gate(r: Strategy)}
  {#if r.latestScore}
    <GatePill passed={r.latestScore.passedGate} />
  {:else}
    <span class="text-base-500">—</span>
  {/if}
{/snippet}

<PageHeader
  title="Strategy Population"
  subtitle="Evolving genomes ranked by overfit-penalized, cost-aware OOS metrics. Win rate is intentionally omitted — it is a banned selection metric."
>
  {#snippet actions()}
    <span class="num rounded border border-base-800 bg-base-900/60 px-2 py-1 text-xs text-base-300">
      {promotedCount} promoted / {data.population.length} total
    </span>
  {/snippet}
</PageHeader>

<DataTable
  rows={data.population}
  {columns}
  rowKey={(r) => r.strategyId}
  initialSort="dsr"
  initialDir="desc"
  empty="Population empty."
/>

<p class="num text-[11px] text-base-500">
  Promotion gate requires DSR &gt; 0 and PBO &lt; 0.50 alongside positive cost-aware OOS expectancy
  across multiple regimes. Green = healthy, red = breaches the bound.
</p>
