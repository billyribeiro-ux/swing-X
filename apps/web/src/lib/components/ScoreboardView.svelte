<script lang="ts">
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import type { Signal } from '@swing-x/shared-types';
  import DataTable from '$lib/components/DataTable.svelte';
  import type { Column } from '$lib/components/data-table';
  import Badge from '$lib/components/Badge.svelte';
  import PageHeader from '$lib/components/PageHeader.svelte';
  import DateRangePicker from '$lib/components/DateRangePicker.svelte';
  import ConvictionBar from '$lib/components/ConvictionBar.svelte';
  import { sideBadgeClass, signTextClass } from '$lib/ui/theme';
  import { fmtInt, fmtLeadTime, fmtPrice, fmtR, fmtRatio, humanize } from '$lib/format';

  /**
   * Shared signal-scoreboard table + header, reused by both the ETF scanner
   * (`/scoreboard`) and the equity scanner (`/equity`). The route supplies the
   * windowed `signals` and scanner-specific heading copy; row-click opens the
   * shared signal detail page regardless of scanner.
   */
  interface Props {
    signals: Signal[];
    title: string;
    subtitle: string;
  }

  let { signals, title, subtitle }: Props = $props();

  function open(row: Signal) {
    goto(resolve('/signals/[id]', { id: row.signalId }));
  }

  const columns: Column<Signal>[] = [
    {
      id: 'ticker',
      header: 'Ticker',
      sortValue: (r) => r.ticker,
      cell: ticker
    },
    {
      id: 'side',
      header: 'Side',
      sortValue: (r) => r.side,
      cell: side
    },
    { id: 'entry', header: 'Entry', numeric: true, sortValue: (r) => r.entry, cell: entry },
    { id: 'stop', header: 'Stop', numeric: true, sortValue: (r) => r.stop, cell: stop },
    {
      id: 'targets',
      header: 'T1 / T2',
      sortable: false,
      numeric: true,
      cell: targets
    },
    { id: 'rr1', header: 'R:R', numeric: true, sortValue: (r) => r.rr1 ?? 0, cell: rr },
    {
      id: 'conviction',
      header: 'Conviction',
      sortValue: (r) => r.conviction,
      cell: conviction,
      class: 'w-40'
    },
    { id: 'regime', header: 'Regime', sortable: false, cell: regime, class: 'max-w-56' },
    { id: 'cohortN', header: 'Cohort n', numeric: true, sortValue: (r) => r.cohortN, cell: cohort },
    {
      id: 'cohortExpectancy',
      header: 'Exp R',
      numeric: true,
      sortValue: (r) => r.cohortExpectancy ?? 0,
      cell: expectancy
    },
    {
      id: 'cvar5',
      header: 'CVaR 5%',
      numeric: true,
      sortValue: (r) => r.cvar5 ?? 0,
      cell: cvar
    },
    {
      id: 'leadTime',
      header: 'Lead',
      numeric: true,
      sortValue: (r) => r.leadTime ?? 0,
      cell: lead
    }
  ];
</script>

{#snippet ticker(r: Signal)}
  <span class="num font-semibold text-base-100">{r.ticker}</span>
{/snippet}

{#snippet side(r: Signal)}
  <Badge class={sideBadgeClass(r.side)}>{r.side}</Badge>
{/snippet}

{#snippet entry(r: Signal)}
  <span class="text-base-100">{fmtPrice(r.entry)}</span>
{/snippet}

{#snippet stop(r: Signal)}
  <span class="text-down">{fmtPrice(r.stop)}</span>
{/snippet}

{#snippet targets(r: Signal)}
  <span class="text-up">{fmtPrice(r.target1)}</span>
  <span class="text-base-600"> / </span>
  <span class="text-up-dim">{fmtPrice(r.target2)}</span>
{/snippet}

{#snippet rr(r: Signal)}
  <span class={signTextClass(r.rr1 ?? 0)}>{fmtRatio(r.rr1)}</span>
{/snippet}

{#snippet conviction(r: Signal)}
  <ConvictionBar value={r.conviction} />
{/snippet}

{#snippet regime(r: Signal)}
  <span class="line-clamp-1 text-xs text-base-300" title={r.regimeDesc}>{r.regimeDesc}</span>
{/snippet}

{#snippet cohort(r: Signal)}
  <span class="text-base-300">{fmtInt(r.cohortN)}</span>
{/snippet}

{#snippet expectancy(r: Signal)}
  <span class={signTextClass(r.cohortExpectancy)}>{fmtR(r.cohortExpectancy)}</span>
{/snippet}

{#snippet cvar(r: Signal)}
  <span class="text-down">{fmtR(r.cvar5)}</span>
{/snippet}

{#snippet lead(r: Signal)}
  <span class="text-base-300">{fmtLeadTime(r.leadTime)}</span>
{/snippet}

<PageHeader {title} {subtitle}>
  {#snippet actions()}
    <DateRangePicker />
    <span class="num rounded border border-base-800 bg-base-900/60 px-2 py-1 text-xs text-base-300">
      {signals.length} live · {humanize('out_of_distribution')} guarded
    </span>
  {/snippet}
</PageHeader>

<DataTable
  rows={signals}
  {columns}
  rowKey={(r) => r.signalId}
  initialSort="conviction"
  initialDir="desc"
  onRowClick={open}
  empty="No signals surfaced."
/>
