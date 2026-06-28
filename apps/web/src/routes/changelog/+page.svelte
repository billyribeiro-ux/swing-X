<script lang="ts">
  import { TrendDown, Trash, ArrowsClockwise } from 'phosphor-svelte';
  import PageHeader from '$lib/components/PageHeader.svelte';
  import type { PageData } from './$types';

  let { data }: { data: PageData } = $props();
</script>

<PageHeader
  title="Weekly Self-Changelog"
  subtitle="What the engine decayed, retired, and adapted each week — its own audit trail."
/>

<div class="flex flex-col gap-5">
  {#each data.weeks as wk (wk.week)}
    <section class="rounded-lg border border-base-800 bg-base-900/40">
      <header class="flex items-center justify-between border-b border-base-800 px-4 py-2.5">
        <span class="num text-sm font-bold tracking-tight text-base-100">{wk.week}</span>
        <span class="num text-[11px] text-base-500">
          {wk.decayed.length} decayed · {wk.retired.length} retired · {wk.adapted.length} adapted
        </span>
      </header>

      <div class="grid grid-cols-1 gap-4 p-4 lg:grid-cols-3">
        <div class="flex flex-col gap-2">
          <h3
            class="flex items-center gap-1.5 text-[11px] font-semibold tracking-wider text-caution uppercase"
          >
            <TrendDown size={13} weight="bold" /> Decayed
          </h3>
          <ul class="flex flex-col gap-1.5">
            {#each wk.decayed as item, i (i)}
              <li class="text-[11px] leading-snug text-base-300">{item}</li>
            {/each}
          </ul>
        </div>

        <div class="flex flex-col gap-2">
          <h3
            class="flex items-center gap-1.5 text-[11px] font-semibold tracking-wider text-down uppercase"
          >
            <Trash size={13} weight="bold" /> Retired
          </h3>
          <ul class="flex flex-col gap-1.5">
            {#each wk.retired as item, i (i)}
              <li class="text-[11px] leading-snug text-base-300">{item}</li>
            {/each}
          </ul>
        </div>

        <div class="flex flex-col gap-2">
          <h3
            class="flex items-center gap-1.5 text-[11px] font-semibold tracking-wider text-up uppercase"
          >
            <ArrowsClockwise size={13} weight="bold" /> Adapted
          </h3>
          <ul class="flex flex-col gap-1.5">
            {#each wk.adapted as item, i (i)}
              <li class="text-[11px] leading-snug text-base-300">{item}</li>
            {/each}
          </ul>
        </div>
      </div>
    </section>
  {/each}
</div>
