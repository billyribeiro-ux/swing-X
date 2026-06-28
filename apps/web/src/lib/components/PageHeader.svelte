<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    title: string;
    subtitle?: string;
    /** Optional right-aligned slot for actions / filters. */
    actions?: Snippet;
  }

  let { title, subtitle, actions }: Props = $props();
</script>

<header class="flex items-end justify-between gap-4 border-b border-base-800 pb-3">
  <div class="flex flex-col gap-0.5">
    <h1 class="text-lg font-semibold tracking-tight text-base-100">{title}</h1>
    {#if subtitle}
      <p class="text-xs text-base-400">{subtitle}</p>
    {/if}
  </div>
  {#if actions}
    <div class="flex items-center gap-2">
      {@render actions()}
    </div>
  {/if}
</header>
