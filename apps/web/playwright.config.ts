import { existsSync } from 'node:fs';
import { defineConfig, devices } from '@playwright/test';

/**
 * E2E smoke config. Builds and previews the production app, then drives Chromium.
 *
 * Browser resolution, in order:
 *   1. PW_CHROMIUM_PATH if set (explicit override).
 *   2. This image's preinstalled Chromium at /opt/pw-browsers, if present (we never run
 *      `playwright install` here — the bundled revision can lag Playwright's expected build).
 *   3. Otherwise `undefined` -> Playwright's own managed browser. This is the GitHub-CI path,
 *      where /opt/pw-browsers does not exist and the e2e job runs `playwright install chromium`.
 */
const PREINSTALLED = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
const chromiumPath =
  process.env.PW_CHROMIUM_PATH ?? (existsSync(PREINSTALLED) ? PREINSTALLED : undefined);

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
        launchOptions: chromiumPath ? { executablePath: chromiumPath } : {}
      }
    }
  ]
});
