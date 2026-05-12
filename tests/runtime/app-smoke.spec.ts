import { expect, test } from '@playwright/test';

test('loads the calculator shell', async ({ page }) => {
  await page.goto('/');

  await expect(
    page.getByRole('heading', {
      name: 'LLM Inference Hardware Calculator',
    })
  ).toBeVisible();
  await expect(page.getByText('Hardware Requirements')).toBeVisible();
});
