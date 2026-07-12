<script lang="ts">
  import '../app.css';
  import { page } from '$app/state';
  import { resolve } from '$app/paths';
  import { navGroups, navItems } from '$lib/nav';
  import { usingLiveApi } from '$lib/api/client';
  import { WifiHigh, WifiSlash } from 'phosphor-svelte';
  import type { Snippet } from 'svelte';

  interface Props {
    children: Snippet;
  }

  let { children }: Props = $props();

  /**
   * The nav entry that best matches the current path: an exact hit, otherwise the
   * longest `href` that is a path-segment prefix. The longest-prefix rule keeps a
   * parent like `/equity` from also lighting up when a child route such as
   * `/equity/journal` (its own nav item) is active.
   */
  const activeHref = $derived.by(() => {
    const path = page.url.pathname;
    let best: string | null = null;
    for (const item of navItems) {
      if (path === item.href || path.startsWith(item.href + '/')) {
        if (best === null || item.href.length > best.length) best = item.href;
      }
    }
    return best;
  });

  function isActive(href: string): boolean {
    return activeHref === href;
  }
</script>

<div class="flex h-screen w-full overflow-hidden">
  <!-- Sidebar -->
  <aside class="flex w-52 shrink-0 flex-col border-r border-base-800 bg-base-900/60 px-3 py-4">
    <a href={resolve('/scoreboard')} class="mb-6 flex items-center gap-2 px-1">
      <span
        class="flex size-7 items-center justify-center rounded bg-up/15 text-up"
        aria-hidden="true"
      >
        <svg viewBox="0 0 24 24" class="size-4" fill="none" aria-hidden="true">
          <path
            d="M3 17 L9 11 L13 14 L21 5"
            stroke="currentColor"
            stroke-width="2.2"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
      </span>
      <div class="flex flex-col leading-none">
        <span class="num text-sm font-bold tracking-tight text-base-100">swing-X</span>
        <span class="text-[10px] font-semibold tracking-wider text-base-400 uppercase"
          >Operator Console</span
        >
      </div>
    </a>

    <nav class="flex flex-col gap-3">
      {#each navGroups as group (group.label ?? group.items[0].href)}
        <div class="flex flex-col gap-0.5">
          {#if group.label}
            <span
              class="px-2.5 pt-1 pb-0.5 text-[10px] font-semibold tracking-wider text-base-500 uppercase"
            >
              {group.label}
            </span>
          {/if}
          {#each group.items as item (item.href)}
            {@const Icon = item.icon}
            <a
              href={resolve(item.href)}
              title={item.hint}
              aria-current={isActive(item.href) ? 'page' : undefined}
              class="flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm transition-colors {isActive(
                item.href
              )
                ? 'bg-base-800/80 font-medium text-base-100'
                : 'text-base-300 hover:bg-base-800/40 hover:text-base-100'}"
            >
              <Icon size={17} weight={isActive(item.href) ? 'fill' : 'regular'} />
              <span>{item.label}</span>
            </a>
          {/each}
        </div>
      {/each}
    </nav>

    <div class="mt-auto flex flex-col gap-2 px-1">
      <div
        class="flex items-center gap-1.5 text-[11px] {usingLiveApi ? 'text-up' : 'text-warn'}"
        title={usingLiveApi
          ? 'Connected to live API (PUBLIC_API_BASE set)'
          : 'Running on local fixtures — set PUBLIC_API_BASE to connect the Rust API'}
      >
        {#if usingLiveApi}
          <WifiHigh size={14} weight="bold" />
          <span>Live API</span>
        {:else}
          <WifiSlash size={14} weight="bold" />
          <span>Fixture mode</span>
        {/if}
      </div>
      <span class="num text-[10px] text-base-600">ETF + equity scanners · OOS-gated</span>
    </div>
  </aside>

  <!-- Main -->
  <main class="flex-1 overflow-y-auto">
    <div class="mx-auto flex max-w-[1400px] flex-col gap-4 px-6 py-5">
      {@render children()}
    </div>
  </main>
</div>
