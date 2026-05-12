import { describe, expect, it } from 'vitest';
import {
  calculateScalingPlan,
  calculateServingCapacity,
} from './servingCapacity';

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

describe('calculateScalingPlan', () => {
  it('recommends tensor parallelism when the model is too large for one GPU', () => {
    const plan = calculateScalingPlan({
      ...baseInput,
      params: 70,
      modelQuant: 'F16',
      servingConfig: {
        ...baseInput.servingConfig,
        gpuCount: 8,
        tensorParallelSize: 1,
        targetConcurrentRequests: 1,
      },
    });

    expect(plan.canFitOneReplica).toBe(true);
    expect(plan.recommendedTensorParallelSize).toBeGreaterThan(1);
    expect(plan.totalGpusNeeded).toBe(plan.recommendedTensorParallelSize);
  });

  it('recommends replicas when one model fits but target concurrency is high', () => {
    const plan = calculateScalingPlan({
      ...baseInput,
      servingConfig: {
        ...baseInput.servingConfig,
        gpuCount: 16,
        tensorParallelSize: 1,
        targetConcurrentRequests: 500,
      },
    });

    expect(plan.recommendedTensorParallelSize).toBe(1);
    expect(plan.replicasNeeded).toBeGreaterThan(1);
    expect(plan.totalGpusNeeded).toBe(plan.replicasNeeded);
  });

  it('calculates total GPUs as tensor parallel size times replicas', () => {
    const plan = calculateScalingPlan({
      ...baseInput,
      params: 70,
      modelQuant: 'Q4',
      servingConfig: {
        ...baseInput.servingConfig,
        gpuCount: 16,
        tensorParallelSize: 1,
        targetConcurrentRequests: 500,
      },
    });

    expect(plan.totalGpusNeeded).toBe(
      plan.recommendedTensorParallelSize * plan.replicasNeeded
    );
  });

  it('marks one replica as not feasible when there are too few GPUs', () => {
    const plan = calculateScalingPlan({
      ...baseInput,
      params: 70,
      modelQuant: 'F16',
      servingConfig: {
        ...baseInput.servingConfig,
        gpuCount: 1,
        tensorParallelSize: 1,
        targetConcurrentRequests: 1,
      },
    });

    expect(plan.canFitOneReplica).toBe(false);
    expect(plan.totalGpusNeeded).toBe(0);
  });
});
