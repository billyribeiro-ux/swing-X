<script lang="ts">
  import type { Driver } from '@swing-x/shared-types';
  import { humanize } from '$lib/format';
  import { layerColor } from '$lib/ui/theme';

  interface Props {
    drivers: Driver[];
  }

  let { drivers }: Props = $props();

  // Scale bar widths against the largest absolute contribution.
  const maxAbs = $derived(Math.max(0.0001, ...drivers.map((d) => Math.abs(d.contribution))));
</script>

<div class="flex flex-col divide-y divide-base-850">
  {#each drivers as d (d.layer + d.key)}
    {@const pct = (Math.abs(d.contribution) / maxAbs) * 100}
    {@const positive = d.contribution >= 0}
    <div class="flex flex-col gap-1 py-2.5 first:pt-0 last:pb-0">
      <div class="flex items-center justify-between gap-2">
        <div class="flex items-center gap-2">
          <span class="text-[10px] font-semibold tracking-wider uppercase {layerColor(d.layer)}">
            {d.layer}
          </span>
          <span class="num text-xs text-base-200">{humanize(d.key)}</span>
        </div>
        <span class="num text-xs font-semibold {positive ? 'text-up' : 'text-down'}">
          {positive ? '+' : ''}{d.contribution.toFixed(2)}
        </span>
      </div>
      <div class="h-1 w-full overflow-hidden rounded-full bg-base-800">
        <div
          class="h-full rounded-full {positive ? 'bg-up/70' : 'bg-down/70'}"
          style:width="{pct}%"
        ></div>
      </div>
      <p class="text-[11px] leading-snug text-base-400">{d.detail}</p>
    </div>
  {/each}
</div>
