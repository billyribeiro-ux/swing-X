import { defineConfig, devices } from '@playwright/test';

/**
 * E2E smoke config. Builds and previews the production app, then drives Chromium.
 *
 * Chromium is preinstalled at /opt/pw-browsers — we never run `playwright install`.
 * The bundled revision can lag this Playwright's expected build, so we point
 * `executablePath` at the full Chromium binary that is actually present. Override
 * with PW_CHROMIUM_PATH if your image places it elsewhere.
 */
const chromiumPath =
  process.env.PW_CHROMIUM_PATH ?? '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';

export default defineConfig({
  testDir: 'e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: 'list',
  webServer: {
    command: 'pnpm build && pnpm preview --port 4173',
    port: 4173,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000
  },
  use: {
    baseURL: 'http://localhost:4173',
    trace: 'on-first-retry'
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: { executablePath: chromiumPath }
      }
    }
  ]
});
