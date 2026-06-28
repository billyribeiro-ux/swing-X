import { expect, test } from '@playwright/test';

test('scoreboard loads, row click opens signal detail', async ({ page }) => {
  // Root redirects to the scoreboard.
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Signal Scoreboard' })).toBeVisible();

  // Wait for hydration so the row's click handler is attached (the sort caret is
  // an interactive button that only exists once the table has rendered).
  await page.waitForLoadState('networkidle');
  const firstRow = page.locator('tbody tr').first();
  await expect(firstRow).toBeVisible();

  // Click the row -> navigate to its detail page. Retry the click until the SPA
  // navigation registers, in case hydration finishes a beat after the row paints.
  await expect(async () => {
    await firstRow.click();
    await page.waitForURL(/\/signals\//, { timeout: 1500 });
  }).toPass();

  // Detail page shows the Why / driver-attribution panel.
  await expect(page.getByRole('heading', { name: 'Why — driver attribution' })).toBeVisible();

  // And the in-page back link (scoped to main, not the sidebar) returns to the
  // scoreboard.
  await page.getByRole('main').getByRole('link', { name: 'Scoreboard' }).click();
  await expect(page.getByRole('heading', { name: 'Signal Scoreboard' })).toBeVisible();
});
