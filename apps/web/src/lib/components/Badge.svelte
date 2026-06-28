<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    /** Tailwind class fragment for bg/text/border, from ui/theme helpers. */
    class?: string;
    /** Render numerics in tabular monospace. */
    mono?: boolean;
    children: Snippet;
  }

  let { class: klass = '', mono = false, children }: Props = $props();
</script>

<span
  class="inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[11px] font-medium tracking-wide uppercase {klass} {mono
    ? 'num'
    : ''}"
>
  {@render children()}
</span>
