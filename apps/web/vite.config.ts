import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [tailwindcss(), sveltekit()],
  test: {
    projects: [
      {
        // Component / DOM tests run against the Svelte browser entrypoint in jsdom.
        extends: './vite.config.ts',
        // `browser` resolves Svelte's client runtime so `mount()` is available;
        // without it vitest picks the SSR build and component renders throw.
        resolve: {
          conditions: ['browser']
        },
        test: {
          name: 'client',
          environment: 'jsdom',
          clearMocks: true,
          include: ['src/**/*.svelte.{test,spec}.{js,ts}'],
          setupFiles: ['./vitest-setup-client.ts']
        }
      },
      {
        // Pure-logic unit tests run in node.
        extends: './vite.config.ts',
        test: {
          name: 'server',
          environment: 'node',
          include: ['src/**/*.{test,spec}.{js,ts}'],
          exclude: ['src/**/*.svelte.{test,spec}.{js,ts}']
        }
      }
    ]
  }
});
