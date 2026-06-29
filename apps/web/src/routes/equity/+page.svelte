<script lang="ts">
  import ScoreboardView from '$lib/components/ScoreboardView.svelte';
  import type { PageData } from './$types';

  let { data }: { data: PageData } = $props();
</script>

<ScoreboardView
  signals={data.signals}
  title="Equity Scanner — Signal Scoreboard"
  subtitle="Surfaced, executable signals across the individual-stock universe. Click a row for full attribution."
/>
