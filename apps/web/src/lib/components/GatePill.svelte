<script lang="ts">
  import { CheckCircle, XCircle } from 'phosphor-svelte';

  interface Props {
    passed: boolean;
  }

  let { passed }: Props = $props();
</script>

{#if passed}
  <span
    class="inline-flex items-center gap-1 rounded border border-up/40 bg-up/15 px-1.5 py-0.5 text-[11px] font-bold tracking-wide text-up uppercase"
  >
    <CheckCircle size={12} weight="fill" />
    Pass
  </span>
{:else}
  <span
    class="inline-flex items-center gap-1 rounded border border-down/40 bg-down/15 px-1.5 py-0.5 text-[11px] font-bold tracking-wide text-down uppercase"
  >
    <XCircle size={12} weight="fill" />
    Fail
  </span>
{/if}
