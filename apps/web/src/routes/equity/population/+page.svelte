<script lang="ts">
  import PopulationView from '$lib/components/PopulationView.svelte';
  import type { PageData } from './$types';

  let { data }: { data: PageData } = $props();
</script>

<PopulationView
  population={data.population}
  title="Equity Scanner — Strategy Population"
  subtitle="Evolving equity genomes ranked by overfit-penalized, cost-aware OOS metrics. Win rate is intentionally omitted — it is a banned selection metric."
/>
