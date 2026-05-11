import { describe, expect, it } from 'vitest';
import {
  calculateHardwareRecommendation,
  calculateKvCacheMemoryGb,
  calculateMemoryBreakdown,
  calculateOnDiskSize,
  calculateRequiredVram,
  getKvCacheQuantFactor,
  getModelQuantFactor,
} from './calculations';

describe('quantization factors', () => {
  it('returns expected model memory factors in GB per billion parameters', () => {
    expect(getModelQuantFactor('F32')).toBe(4);
    expect(getModelQuantFactor('F16')).toBe(2);
    expect(getModelQuantFactor('Q8')).toBe(1);
    expect(getModelQuantFactor('Q4')).toBe(0.5);
    expect(getModelQuantFactor('AWQ')).toBe(0.35);
  });

  it('returns expected KV cache memory factors', () => {
    expect(getKvCacheQuantFactor('F16')).toBe(2);
    expect(getKvCacheQuantFactor('Q8')).toBe(1);
    expect(getKvCacheQuantFactor('Q4')).toBe(0.5);
  });
});

describe('calculateRequiredVram', () => {
  it('keeps base model memory independent of context when KV cache is disabled in incremental mode', () => {
    const shortContext = calculateRequiredVram(
      7,
      'F16',
      2048,
      false,
      'F16',
      'incremental'
    );
    const longContext = calculateRequiredVram(
      7,
      'F16',
      8192,
      false,
      'F16',
      'incremental'
    );

    expect(shortContext).toBeCloseTo(15.4, 5);
    expect(longContext).toBeCloseTo(shortContext, 5);
  });

  it('increases memory when KV cache is enabled and context grows', () => {
    const shortContext = calculateRequiredVram(
      70,
      'Q4',
      2048,
      true,
      'Q4',
      'incremental'
    );
    const longContext = calculateRequiredVram(
      70,
      'Q4',
      8192,
      true,
      'Q4',
      'incremental'
    );

    expect(longContext).toBeGreaterThan(shortContext);
  });

  it('adds more context-dependent memory in bulk mode than incremental mode without KV cache', () => {
    const incremental = calculateRequiredVram(
      13,
      'Q4',
      4096,
      false,
      'Q4',
      'incremental'
    );
    const bulk = calculateRequiredVram(
      13,
      'Q4',
      4096,
      false,
      'Q4',
      'bulk'
    );

    expect(bulk).toBeGreaterThan(incremental);
  });
});

describe('calculateMemoryBreakdown', () => {
  it('returns visible components that add up to the total', () => {
    const breakdown = calculateMemoryBreakdown(
      7,
      'F16',
      2048,
      false,
      'F16',
      'incremental'
    );

    expect(breakdown.modelWeightsGb).toBeCloseTo(14, 5);
    expect(breakdown.kvCacheGb).toBe(0);
    expect(breakdown.activationGb).toBe(0);
    expect(breakdown.runtimeOverheadGb).toBeCloseTo(1.4, 5);
    expect(breakdown.totalGb).toBeCloseTo(
      breakdown.modelWeightsGb +
        breakdown.kvCacheGb +
        breakdown.activationGb +
        breakdown.runtimeOverheadGb,
      5
    );
  });

  it('shows KV cache as its own component when enabled', () => {
    const breakdown = calculateMemoryBreakdown(
      70,
      'Q4',
      8192,
      true,
      'Q4',
      'incremental'
    );

    expect(breakdown.modelWeightsGb).toBeCloseTo(35, 5);
    expect(breakdown.kvCacheGb).toBeGreaterThan(0);
    expect(breakdown.activationGb).toBe(0);
  });

  it('shows activation memory as its own component in bulk mode', () => {
    const incremental = calculateMemoryBreakdown(
      13,
      'Q4',
      4096,
      false,
      'Q4',
      'incremental'
    );
    const bulk = calculateMemoryBreakdown(
      13,
      'Q4',
      4096,
      false,
      'Q4',
      'bulk'
    );

    expect(bulk.activationGb).toBeGreaterThan(incremental.activationGb);
    expect(bulk.totalGb).toBeGreaterThan(incremental.totalGb);
  });

  it('uses architecture-level KV cache math when architecture is provided', () => {
    const breakdown = calculateMemoryBreakdown(
      7,
      'Q4',
      4096,
      true,
      'F16',
      'incremental',
      {
        layers: 32,
        hiddenSize: 4096,
        attentionHeads: 32,
        kvHeads: 8,
        headDim: 128,
      },
      1
    );

    expect(breakdown.kvCacheGb).toBeCloseTo(0.536870912, 5);
  });
});

describe('calculateKvCacheMemoryGb', () => {
  const architecture = {
    layers: 32,
    hiddenSize: 4096,
    attentionHeads: 32,
    kvHeads: 8,
    headDim: 128,
  };

  it('calculates KV cache from layers, KV heads, head dimension, dtype, context, and concurrency', () => {
    expect(
      calculateKvCacheMemoryGb({
        architecture,
        contextLength: 4096,
        kvCacheQuant: 'F16',
        concurrentRequests: 1,
      })
    ).toBeCloseTo(0.536870912, 5);
  });

  it('scales linearly with context length', () => {
    const shortContext = calculateKvCacheMemoryGb({
      architecture,
      contextLength: 4096,
      kvCacheQuant: 'F16',
      concurrentRequests: 1,
    });
    const longContext = calculateKvCacheMemoryGb({
      architecture,
      contextLength: 8192,
      kvCacheQuant: 'F16',
      concurrentRequests: 1,
    });

    expect(longContext).toBeCloseTo(shortContext * 2, 5);
  });

  it('scales linearly with concurrent requests', () => {
    const singleRequest = calculateKvCacheMemoryGb({
      architecture,
      contextLength: 4096,
      kvCacheQuant: 'F16',
      concurrentRequests: 1,
    });
    const twoRequests = calculateKvCacheMemoryGb({
      architecture,
      contextLength: 4096,
      kvCacheQuant: 'F16',
      concurrentRequests: 2,
    });

    expect(twoRequests).toBeCloseTo(singleRequest * 2, 5);
  });

  it('uses KV heads rather than total attention heads', () => {
    const gqaModel = calculateKvCacheMemoryGb({
      architecture,
      contextLength: 4096,
      kvCacheQuant: 'F16',
      concurrentRequests: 1,
    });
    const fullAttentionKvModel = calculateKvCacheMemoryGb({
      architecture: {
        ...architecture,
        kvHeads: architecture.attentionHeads,
      },
      contextLength: 4096,
      kvCacheQuant: 'F16',
      concurrentRequests: 1,
    });

    expect(fullAttentionKvModel).toBeCloseTo(gqaModel * 4, 5);
  });

  it('scales with KV cache quantization bytes per element', () => {
    const fp16 = calculateKvCacheMemoryGb({
      architecture,
      contextLength: 4096,
      kvCacheQuant: 'F16',
      concurrentRequests: 1,
    });
    const q8 = calculateKvCacheMemoryGb({
      architecture,
      contextLength: 4096,
      kvCacheQuant: 'Q8',
      concurrentRequests: 1,
    });

    expect(q8).toBeCloseTo(fp16 / 2, 5);
  });
});

describe('calculateHardwareRecommendation', () => {
  it('calculates a single discrete GPU recommendation when the model fits', () => {
    const recommendation = calculateHardwareRecommendation(
      7,
      'Q4',
      4096,
      false,
      'Q4',
      'DISCRETE_GPU',
      64,
      24,
      'incremental'
    );

    expect(recommendation.gpusRequired).toBe(1);
    expect(recommendation.gpuType).toBe('Single 24GB GPU');
  });

  it('marks very large discrete GPU requirements as not feasible', () => {
    const recommendation = calculateHardwareRecommendation(
      1000,
      'F32',
      32768,
      true,
      'F32',
      'DISCRETE_GPU',
      64,
      8,
      'bulk'
    );

    expect(recommendation.gpusRequired).toBe(0);
    expect(recommendation.gpuType).toMatch(/^Exceeds 8x/);
  });

  it('reports unified memory fit based on required VRAM', () => {
    const recommendation = calculateHardwareRecommendation(
      7,
      'Q4',
      4096,
      false,
      'Q4',
      'UNIFIED_MEMORY',
      64,
      24,
      'incremental'
    );

    expect(recommendation.fitsUnified).toBe(true);
    expect(recommendation.gpuType).toBe('Unified memory (64GB)');
  });
});

describe('calculateOnDiskSize', () => {
  it('estimates model size from parameter count and quantization', () => {
    expect(calculateOnDiskSize(7, 'F16')).toBeCloseTo(14, 5);
    expect(calculateOnDiskSize(7, 'Q4')).toBeCloseTo(3.5, 5);
  });
});
