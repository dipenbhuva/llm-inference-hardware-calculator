import { describe, expect, it } from 'vitest';
import { calculateServingCapacity } from './servingCapacity';

const architecture = {
  layers: 32,
  hiddenSize: 4096,
  attentionHeads: 32,
  kvHeads: 8,
  headDim: 128,
};

const baseInput = {
  params: 7,
  modelQuant: 'Q4' as const,
  kvCacheQuant: 'F16' as const,
  gpuVram: 24,
  architecture,
  servingConfig: {
    gpuCount: 1,
    gpuMemoryUtilization: 0.9,
    maxModelLen: 4096,
    tensorParallelSize: 1,
    targetConcurrentRequests: 1,
  },
};

describe('calculateServingCapacity', () => {
  it('reduces usable memory when GPU memory utilization is lower', () => {
    const highUtilization = calculateServingCapacity(baseInput);
    const lowUtilization = calculateServingCapacity({
      ...baseInput,
      servingConfig: {
        ...baseInput.servingConfig,
        gpuMemoryUtilization: 0.8,
      },
    });

    expect(highUtilization.usableGpuMemoryGb).toBeGreaterThan(
      lowUtilization.usableGpuMemoryGb
    );
  });

  it('reduces per-GPU model weight memory with higher tensor parallel size', () => {
    const tp1 = calculateServingCapacity({
      ...baseInput,
      servingConfig: {
        ...baseInput.servingConfig,
        gpuCount: 2,
        tensorParallelSize: 1,
      },
    });
    const tp2 = calculateServingCapacity({
      ...baseInput,
      servingConfig: {
        ...baseInput.servingConfig,
        gpuCount: 2,
        tensorParallelSize: 2,
      },
    });

    expect(tp2.modelWeightsPerGpuGb).toBeCloseTo(
      tp1.modelWeightsPerGpuGb / 2,
      5
    );
  });

  it('reduces max concurrency when max model length increases', () => {
    const shortContext = calculateServingCapacity(baseInput);
    const longContext = calculateServingCapacity({
      ...baseInput,
      servingConfig: {
        ...baseInput.servingConfig,
        maxModelLen: 8192,
      },
    });

    expect(longContext.maxConcurrentRequests).toBeLessThan(
      shortContext.maxConcurrentRequests
    );
  });

  it('separates model weight fit from target concurrency fit', () => {
    const capacity = calculateServingCapacity({
      ...baseInput,
      servingConfig: {
        ...baseInput.servingConfig,
        targetConcurrentRequests: 1000,
      },
    });

    expect(capacity.weightsFit).toBe(true);
    expect(capacity.targetConcurrencyFits).toBe(false);
  });
});
