import { expect, test } from '@playwright/test';

const canonicalUrl =
  'https://dipenbhuva.github.io/llm-inference-hardware-calculator/';

test('exposes search and social metadata', async ({ page }) => {
  await page.goto('/');

  await expect(page).toHaveTitle(
    'LLM Inference Hardware Calculator | GPU VRAM and Serving Capacity'
  );
  await expect(page.locator('link[rel="canonical"]')).toHaveAttribute(
    'href',
    canonicalUrl
  );
  await expect(page.locator('meta[name="description"]')).toHaveAttribute(
    'content',
    /Estimate GPU VRAM/
  );
  await expect(page.locator('meta[name="robots"]')).toHaveAttribute(
    'content',
    /index, follow/
  );
  await expect(page.locator('meta[property="og:url"]')).toHaveAttribute(
    'content',
    canonicalUrl
  );
  await expect(page.locator('meta[name="twitter:card"]')).toHaveAttribute(
    'content',
    'summary'
  );
});

test('exposes structured data for the calculator', async ({ page }) => {
  await page.goto('/');

  const jsonLd = await page
    .locator('script[type="application/ld+json"]')
    .textContent();

  expect(jsonLd).not.toBeNull();
  const parsed = JSON.parse(jsonLd ?? '{}') as {
    '@graph'?: Array<Record<string, unknown>>;
  };
  const graph = parsed['@graph'] ?? [];

  expect(graph).toEqual(
    expect.arrayContaining([
      expect.objectContaining({
        '@type': 'WebApplication',
        name: 'LLM Inference Hardware Calculator',
        url: canonicalUrl,
      }),
      expect.objectContaining({
        '@type': 'LearningResource',
        name: 'GPU Hosting Labs for Production-Ready AI',
      }),
    ])
  );
});

test('serves crawler discovery files', async ({ request }) => {
  const robots = await request.get('/robots.txt');
  expect(robots.ok()).toBe(true);
  const robotsText = await robots.text();
  expect(robotsText).toContain('User-agent: OAI-SearchBot');
  expect(robotsText).toContain('User-agent: Claude-SearchBot');
  expect(robotsText).toContain(`${canonicalUrl}sitemap.xml`);

  const sitemap = await request.get('/sitemap.xml');
  expect(sitemap.ok()).toBe(true);
  const sitemapText = await sitemap.text();
  expect(sitemapText).toContain(`<loc>${canonicalUrl}</loc>`);

  const llms = await request.get('/llms.txt');
  expect(llms.ok()).toBe(true);
  const llmsText = await llms.text();
  expect(llmsText).toContain('LLM Inference Hardware Calculator');
  expect(llmsText).toContain('KV cache memory');
});
