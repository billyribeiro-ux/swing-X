<script lang="ts">
  import { page } from '$app/state';
  import { resolve } from '$app/paths';
  import { Warning } from 'phosphor-svelte';
</script>

<div class="flex flex-1 flex-col items-center justify-center gap-3 py-24 text-center">
  <Warning size={32} class="text-warn" weight="fill" />
  <span class="num text-3xl font-bold text-base-100">{page.status}</span>
  <p class="text-sm text-base-400">{page.error?.message ?? 'Something went wrong.'}</p>
  <a
    href={resolve('/scoreboard')}
    class="mt-2 rounded-md border border-base-700 px-3 py-1.5 text-xs text-base-200 hover:bg-base-800/60"
  >
    Back to scoreboard
  </a>
</div>
