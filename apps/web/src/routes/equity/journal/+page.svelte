<script lang="ts">
  import JournalView from '$lib/components/JournalView.svelte';
  import type { PageData } from './$types';

  let { data }: { data: PageData } = $props();
</script>

<JournalView
  trades={data.trades}
  title="Equity Scanner — Journal"
  subtitle="Every equity fill linked to its originating signal and strategy, with an attribution snapshot."
/>
