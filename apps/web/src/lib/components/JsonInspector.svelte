<script lang="ts">
  import { CaretRight } from 'phosphor-svelte';

  interface Props {
    value: unknown;
    label?: string;
  }

  let { value, label = 'Raw payload JSON' }: Props = $props();

  let open = $state(false);
  const json = $derived(JSON.stringify(value, null, 2));
</script>

<div class="rounded-lg border border-base-800 bg-base-900/40">
  <button
    type="button"
    class="flex w-full items-center gap-2 px-3 py-2 text-left text-xs font-medium text-base-200 hover:text-base-100"
    aria-expanded={open}
    onclick={() => (open = !open)}
  >
    <span class="transition-transform {open ? 'rotate-90' : ''}">
      <CaretRight size={12} weight="bold" />
    </span>
    <span>{label}</span>
  </button>
  {#if open}
    <pre
      class="num max-h-96 overflow-auto border-t border-base-800 px-3 py-2 text-[11px] leading-relaxed text-base-300">{json}</pre>
  {/if}
</div>
