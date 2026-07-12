<script lang="ts">
  interface Props {
    label: string;
    value: string;
    /** Optional Tailwind text-color class for the value. */
    valueClass?: string;
    /** Optional sub-line under the value. */
    sub?: string;
  }

  let { label, value, valueClass = 'text-base-100', sub }: Props = $props();
</script>

<div class="flex flex-col gap-0.5 rounded border border-base-800 bg-base-900/60 px-3 py-2">
  <span class="text-[10px] font-semibold tracking-wider text-base-400 uppercase">{label}</span>
  <span class="num text-lg leading-none font-semibold {valueClass}">{value}</span>
  {#if sub}
    <span class="num text-[11px] text-base-400">{sub}</span>
  {/if}
</div>
