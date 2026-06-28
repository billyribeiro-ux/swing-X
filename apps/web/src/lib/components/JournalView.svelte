<script lang="ts">
  import { resolve } from '$app/paths';
  import type { Trade } from '@swing-x/shared-types';
  import DataTable from '$lib/components/DataTable.svelte';
  import type { Column } from '$lib/components/data-table';
  import Badge from '$lib/components/Badge.svelte';
  import Metric from '$lib/components/Metric.svelte';
  import PageHeader from '$lib/components/PageHeader.svelte';
  import DateRangePicker from '$lib/components/DateRangePicker.svelte';
  import { sideBadgeClass, signTextClass } from '$lib/ui/theme';
  import { fmtPct, fmtPrice, fmtR, fmtTs } from '$lib/format';

  /**
   * Shared paper-trade journal table + summary metrics, reused by both the ETF
   * scanner (`/journal`) and the equity scanner (`/equity/journal`). The route
   * supplies the windowed `trades` and scanner-specific heading copy.
   */
  interface Props {
    trades: Trade[];
    title: string;
    subtitle: string;
  }

  let { trades, title, subtitle }: Props = $props();

  const closed = $derived(trades.filter((t) => t.pnlR != null));
  const totalR = $derived(closed.reduce((acc, t) => acc + (t.pnlR ?? 0), 0));
  const avgR = $derived(closed.length ? totalR / closed.length : 0);
  const winners = $derived(closed.filter((t) => (t.pnlR ?? 0) > 0).length);
  const openCount = $derived(trades.length - closed.length);

  const columns: Column<Trade>[] = [
    { id: 'ticker', header: 'Ticker', sortValue: (r) => r.ticker, cell: ticker },
    { id: 'side', header: 'Side', sortValue: (r) => r.side, cell: side },
    { id: 'mode', header: 'Mode', sortValue: (r) => r.mode, cell: mode },
    { id: 'entryTs', header: 'Entry', sortValue: (r) => r.entryTs, cell: entryTs },
    { id: 'fillPx', header: 'Fill', numeric: true, sortValue: (r) => r.fillPx, cell: fillPx },
    { id: 'exitPx', header: 'Exit', numeric: true, sortValue: (r) => r.exitPx ?? 0, cell: exitPx },
    {
      id: 'pnlR',
      header: 'PnL (R)',
      numeric: true,
      sortValue: (r) => r.pnlR ?? -Infinity,
      cell: pnl
    },
    {
      id: 'costFrac',
      header: 'Cost',
      numeric: true,
      sortValue: (r) => r.costFrac ?? 0,
      cell: cost
    },
    { id: 'link', header: 'Linked', sortable: false, cell: link }
  ];
</script>

{#snippet ticker(r: Trade)}
  <span class="num font-semibold text-base-100">{r.ticker}</span>
{/snippet}

{#snippet side(r: Trade)}
  <Badge class={sideBadgeClass(r.side)}>{r.side}</Badge>
{/snippet}

{#snippet mode(r: Trade)}
  <Badge
    class={r.mode === 'live'
      ? 'border-accent/30 bg-accent/10 text-accent'
      : 'border-base-700 bg-base-800/60 text-base-400'}
  >
    {r.mode}
  </Badge>
{/snippet}

{#snippet entryTs(r: Trade)}
  <span class="num text-xs text-base-300">{fmtTs(r.entryTs)}</span>
{/snippet}

{#snippet fillPx(r: Trade)}
  <span class="text-base-200">{fmtPrice(r.fillPx)}</span>
{/snippet}

{#snippet exitPx(r: Trade)}
  {#if r.exitPx != null}
    <span class="text-base-200">{fmtPrice(r.exitPx)}</span>
  {:else}
    <Badge class="border-accent/30 bg-accent/10 text-accent">open</Badge>
  {/if}
{/snippet}

{#snippet pnl(r: Trade)}
  {#if r.pnlR != null}
    <span class="font-semibold {signTextClass(r.pnlR)}">{fmtR(r.pnlR)}</span>
  {:else}
    <span class="text-base-500">—</span>
  {/if}
{/snippet}

{#snippet cost(r: Trade)}
  <span class="text-base-400">{r.costFrac != null ? fmtPct(r.costFrac) : '—'}</span>
{/snippet}

{#snippet link(r: Trade)}
  <div class="flex flex-col gap-0.5">
    {#if r.signalId}
      <a
        href={resolve('/signals/[id]', { id: r.signalId })}
        class="num text-[11px] text-info hover:underline"
      >
        {r.signalId}
      </a>
    {/if}
    {#if r.strategyId}
      <span class="num text-[11px] text-base-500">{r.strategyId}</span>
    {/if}
  </div>
{/snippet}

<PageHeader {title} {subtitle}>
  {#snippet actions()}
    <DateRangePicker />
  {/snippet}
</PageHeader>

<div class="grid grid-cols-2 gap-2 sm:grid-cols-4">
  <Metric label="Total R (closed)" value={fmtR(totalR)} valueClass={signTextClass(totalR)} />
  <Metric label="Avg R / trade" value={fmtR(avgR)} valueClass={signTextClass(avgR)} />
  <Metric label="Closed" value={`${winners}/${closed.length}`} sub="positive / closed" />
  <Metric label="Open" value={String(openCount)} valueClass="text-accent" />
</div>

<DataTable
  rows={trades}
  {columns}
  rowKey={(r) => r.tradeId}
  initialSort="entryTs"
  initialDir="desc"
  empty="No trades journaled."
/>
