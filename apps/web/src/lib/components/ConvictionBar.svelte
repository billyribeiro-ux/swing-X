<script lang="ts">
  import { fmtPct } from '$lib/format';

  interface Props {
    /** Calibrated conviction in [0, 1]. */
    value: number;
  }

  let { value }: Props = $props();

  const clamped = $derived(Math.max(0, Math.min(1, value)));
  // Color ramps from caution -> accent -> up as conviction rises.
  const barClass = $derived(
    clamped >= 0.65 ? 'bg-up' : clamped >= 0.5 ? 'bg-accent' : 'bg-caution'
  );
  const textClass = $derived(
    clamped >= 0.65 ? 'text-up' : clamped >= 0.5 ? 'text-accent' : 'text-caution'
  );
</script>

<div class="flex items-center gap-2">
  <div class="h-1.5 flex-1 overflow-hidden rounded-full bg-base-800">
    <div class="h-full rounded-full {barClass}" style:width="{clamped * 100}%"></div>
  </div>
  <span class="num w-9 text-right text-xs font-semibold {textClass}">{fmtPct(clamped)}</span>
</div>
