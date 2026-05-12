import { expect, type Page, test } from '@playwright/test';

const getPanelValue = async (
  page: Page,
  panelClass: string,
  label: string
): Promise<string> => {
  const row = page.locator(`${panelClass} dl > div`).filter({
    has: page.locator('dt', { hasText: label }),
  });
  return (await row.locator('dd').innerText()).trim();
};

const getGbValue = async (
  page: Page,
  panelClass: string,
  label: string
): Promise<number> => {
  const value = await getPanelValue(page, panelClass, label);
  return Number(value.replace(/[^0-9.]/g, ''));
};

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test('RT-L01 shows model memory breakdown and updates with quantization', async ({
  page,
}) => {
  await page.getByLabel(/Number of Parameters/).fill('7');
  await page.getByLabel('Model Quantization').selectOption('F16');
  await page.getByLabel('Enable KV Cache').uncheck();

  await expect(page.locator('.memory-breakdown')).toBeVisible();
  await expect(
    page.locator('.memory-breakdown').getByText('14.00 GB')
  ).toBeVisible();

  await page.getByLabel('Model Quantization').selectOption('Q4');
  await expect(
    page.locator('.memory-breakdown').getByText('3.50 GB')
  ).toBeVisible();

  await page.getByLabel('Enable KV Cache').check();
  await page.getByLabel(/Context Length/).fill('8192');
  const kvCacheGb = await getGbValue(page, '.memory-breakdown', 'KV cache');

  expect(kvCacheGb).toBeGreaterThan(0);
});

test('RT-L02 KV cache scales with context, concurrency, and dtype', async ({
  page,
}) => {
  await page.getByLabel(/Number of Parameters/).fill('7');
  await page.getByLabel('Model Quantization').selectOption('Q4');
  await page.getByLabel(/Layers/).fill('32');
  await page.getByLabel(/Attention Heads/).fill('32');
  await page.getByLabel(/KV Heads/).fill('8');
  await page.getByLabel(/Head Dimension/).fill('128');
  await page.getByLabel('Enable KV Cache').check();
  await page.getByLabel('KV Cache Quantization').selectOption('F16');
  await page.getByLabel(/Context Length/).fill('4096');
  await page.getByLabel(/Concurrent Requests/).fill('1');

  const baseKv = await getGbValue(page, '.memory-breakdown', 'KV cache');

  await page.getByLabel(/Context Length/).fill('8192');
  const longContextKv = await getGbValue(page, '.memory-breakdown', 'KV cache');
  expect(longContextKv).toBeGreaterThan(baseKv * 1.9);

  await page.getByLabel(/Concurrent Requests/).fill('2');
  const concurrentKv = await getGbValue(page, '.memory-breakdown', 'KV cache');
  expect(concurrentKv).toBeGreaterThan(longContextKv * 1.9);

  await page.getByLabel('KV Cache Quantization').selectOption('Q8');
  const q8Kv = await getGbValue(page, '.memory-breakdown', 'KV cache');
  expect(q8Kv).toBeLessThan(concurrentKv * 0.6);
});

test('RT-L03 model presets populate architecture fields', async ({ page }) => {
  await page.getByLabel('Model Preset').selectOption('generic-7b');
  await expect(page.getByLabel(/Number of Parameters/)).toHaveValue('7');
  await expect(page.getByLabel(/Layers/)).toHaveValue('32');

  await page.getByLabel('Model Preset').selectOption('generic-70b');
  await expect(page.getByLabel(/Number of Parameters/)).toHaveValue('70');
  await expect(page.getByLabel(/Layers/)).toHaveValue('80');
});

test('RT-L04 GPU presets update VRAM and custom mode allows editing', async ({
  page,
}) => {
  await page.getByLabel('GPU Preset').selectOption('nvidia-l40s-48gb');
  await expect(page.getByLabel(/GPU VRAM/)).toHaveValue('48');
  await expect(page.getByLabel(/GPU VRAM/)).toBeDisabled();

  await page.getByLabel('GPU Preset').selectOption('custom');
  await expect(page.getByLabel(/GPU VRAM/)).toBeEnabled();
  await page.getByLabel(/GPU VRAM/).fill('96');
  await expect(page.getByLabel(/GPU VRAM/)).toHaveValue('96');
});

test('RT-L05 vLLM fit estimate responds to utilization and max length', async ({
  page,
}) => {
  await page.getByLabel('Model Preset').selectOption('generic-7b');
  await page.getByLabel('Model Quantization').selectOption('Q4');
  await page.getByLabel(/GPU Memory Utilization/).fill('0.90');
  await page.getByLabel(/Max Model Length/).fill('4096');

  const highUtilizationMemory = await getGbValue(
    page,
    '.serving-capacity',
    'Usable memory per GPU'
  );
  const shortContextConcurrency = Number(
    (await getPanelValue(
      page,
      '.serving-capacity',
      'Max concurrent requests'
    )).replace(/,/g, '')
  );

  await page.getByLabel(/GPU Memory Utilization/).fill('0.80');
  const lowUtilizationMemory = await getGbValue(
    page,
    '.serving-capacity',
    'Usable memory per GPU'
  );
  expect(lowUtilizationMemory).toBeLessThan(highUtilizationMemory);

  await page.getByLabel(/Max Model Length/).fill('8192');
  const longContextConcurrency = Number(
    (await getPanelValue(
      page,
      '.serving-capacity',
      'Max concurrent requests'
    )).replace(/,/g, '')
  );
  expect(longContextConcurrency).toBeLessThan(shortContextConcurrency);
});

test('RT-L07 diagnostics explain impossible serving configurations', async ({
  page,
}) => {
  await page.getByLabel('Model Preset').selectOption('generic-70b');
  await page.getByLabel('Model Quantization').selectOption('F16');
  await page.getByLabel('GPU Preset').selectOption('rtx-4090-24gb');
  await page.getByLabel(/GPU Count/).fill('1');
  await page.getByLabel(/Tensor Parallel Size/).fill('1');

  await expect(
    page.locator('.diagnostics-panel').getByText('Model weights do not fit')
  ).toBeVisible();
});

test('RT-L08 scaling plan separates tensor parallelism from replicas', async ({
  page,
}) => {
  await page.getByLabel('Model Preset').selectOption('generic-7b');
  await page.getByLabel('Model Quantization').selectOption('Q4');
  await page.getByLabel(/Concurrent Requests/).fill('500');

  await expect(page.locator('.scaling-plan')).toBeVisible();
  const replicas = Number(
    (await getPanelValue(page, '.scaling-plan', 'Replicas needed')).replace(
      /,/g,
      ''
    )
  );
  expect(replicas).toBeGreaterThan(1);
});

test('RT-L06 serving command reflects vLLM settings', async ({ page }) => {
  await page.getByLabel(/Tensor Parallel Size/).fill('2');
  await page.getByLabel(/Max Model Length/).fill('8192');
  await page.getByLabel(/GPU Memory Utilization/).fill('0.90');
  await page.getByLabel('KV Cache Quantization').selectOption('FP8');

  const command = page.locator('.serving-command pre');
  await expect(command).toContainText('vllm serve <model-or-path>');
  await expect(command).toContainText('--tensor-parallel-size 2');
  await expect(command).toContainText('--max-model-len 8192');
  await expect(command).toContainText('--gpu-memory-utilization 0.90');
  await expect(command).toContainText('--kv-cache-dtype fp8');
});
