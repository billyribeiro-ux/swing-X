<script lang="ts" generics="Row">
  import { untrack } from 'svelte';
  import { CaretDown, CaretUp, CaretUpDown } from 'phosphor-svelte';
  import type { Column } from './data-table';

  interface Props {
    rows: Row[];
    columns: Column<Row>[];
    /** Stable row key extractor. */
    rowKey: (row: Row) => string;
    /** Initial sort column id. */
    initialSort?: string;
    /** Initial sort direction. */
    initialDir?: 'asc' | 'desc';
    /** Optional row-click handler (e.g. navigate to detail). */
    onRowClick?: (row: Row) => void;
    /** Message shown when there are no rows. */
    empty?: string;
  }

  let {
    rows,
    columns,
    rowKey,
    initialSort,
    initialDir = 'desc',
    onRowClick,
    empty = 'No rows.'
  }: Props = $props();

  // Sort state is seeded once from the initial props; later prop changes do not
  // override an operator's active sort. The untrack keeps that intent explicit
  // and silences the "referenced locally" lint.
  let sortId = $state<string | undefined>(untrack(() => initialSort));
  let sortDir = $state<'asc' | 'desc'>(untrack(() => initialDir));

  function colById(id: string): Column<Row> | undefined {
    return columns.find((c) => c.id === id);
  }

  const sortedRows = $derived.by(() => {
    if (!sortId) return rows;
    const col = colById(sortId);
    if (!col?.sortValue) return rows;
    const sortValue = col.sortValue;
    const dir = sortDir === 'asc' ? 1 : -1;
    return [...rows].sort((a, b) => {
      const av = sortValue(a);
      const bv = sortValue(b);
      if (typeof av === 'number' && typeof bv === 'number') {
        return (av - bv) * dir;
      }
      return String(av).localeCompare(String(bv)) * dir;
    });
  });

  function toggleSort(col: Column<Row>) {
    if (col.sortable === false || !col.sortValue) return;
    if (sortId === col.id) {
      sortDir = sortDir === 'asc' ? 'desc' : 'asc';
    } else {
      sortId = col.id;
      sortDir = 'desc';
    }
  }

  const clickable = $derived(typeof onRowClick === 'function');
</script>

<div class="overflow-x-auto rounded-lg border border-base-800">
  <table class="w-full border-collapse text-sm">
    <thead>
      <tr class="border-b border-base-800 bg-base-900/80">
        {#each columns as col (col.id)}
          <th
            scope="col"
            class="px-3 py-2 text-[10px] font-semibold tracking-wider text-base-400 uppercase select-none {col.numeric
              ? 'text-right'
              : 'text-left'} {col.class ?? ''}"
            aria-sort={sortId === col.id
              ? sortDir === 'asc'
                ? 'ascending'
                : 'descending'
              : 'none'}
          >
            {#if col.sortable === false || !col.sortValue}
              <span>{col.header}</span>
            {:else}
              <button
                type="button"
                class="inline-flex items-center gap-1 hover:text-base-100 {col.numeric
                  ? 'flex-row-reverse'
                  : ''}"
                onclick={() => toggleSort(col)}
              >
                <span>{col.header}</span>
                {#if sortId === col.id}
                  {#if sortDir === 'asc'}
                    <CaretUp size={11} weight="bold" />
                  {:else}
                    <CaretDown size={11} weight="bold" />
                  {/if}
                {:else}
                  <CaretUpDown size={11} class="text-base-600" />
                {/if}
              </button>
            {/if}
          </th>
        {/each}
      </tr>
    </thead>
    <tbody>
      {#each sortedRows as row (rowKey(row))}
        <tr
          class="border-b border-base-850/60 transition-colors last:border-0 hover:bg-base-800/40 {clickable
            ? 'cursor-pointer'
            : ''}"
          onclick={() => onRowClick?.(row)}
        >
          {#each columns as col (col.id)}
            <td
              class="px-3 py-2 align-middle {col.numeric ? 'num text-right' : ''} {col.class ?? ''}"
            >
              {@render col.cell(row)}
            </td>
          {/each}
        </tr>
      {:else}
        <tr>
          <td colspan={columns.length} class="px-3 py-8 text-center text-base-400">
            {empty}
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
</div>
