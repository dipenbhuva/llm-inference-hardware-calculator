import { describe, expect, it } from 'vitest';
import {
  buildVllmServeCommand,
  calculateDiagnostics,
  getVllmKvCacheDtype,
} from './recommendations';
import { calculateServingCapacity } from './servingCapacity';

const architecture = {
  layers: 32,
  hiddenSize: 4096,
  attentionHeads: 32,
  kvHeads: 8,
  headDim: 128,
};

const createCapacity = (
  overrides: Partial<Parameters<typeof calculateServingCapacity>[0]> = {}
) => {
  const input = {
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
    ...overrides,
  };

  return {
    input,
    capacity: calculateServingCapacity(input),
  };
};

describe('calculateDiagnostics', () => {
  it('detects model weights that exceed usable VRAM', () => {
    const { input, capacity } = createCapacity({
      params: 70,
      modelQuant: 'F16',
      gpuVram: 24,
    });

    const diagnostics = calculateDiagnostics({
      capacity,
      servingConfig: input.servingConfig,
    });

    expect(diagnostics.map((diagnostic) => diagnostic.id)).toContain(
      'weights_exceed_usable_vram'
    );
  });

  it('detects KV cache that exceeds remaining VRAM', () => {
    const { input, capacity } = createCapacity({
      servingConfig: {
        gpuCount: 1,
        gpuMemoryUtilization: 0.9,
        maxModelLen: 262144,
        tensorParallelSize: 1,
        targetConcurrentRequests: 1,
      },
    });

    const diagnostics = calculateDiagnostics({
      capacity,
      servingConfig: input.servingConfig,
    });

    expect(diagnostics.map((diagnostic) => diagnostic.id)).toContain(
      'kv_cache_exceeds_remaining_vram'
    );
  });

  it('detects target concurrency above capacity', () => {
    const { input, capacity } = createCapacity({
      servingConfig: {
        gpuCount: 1,
        gpuMemoryUtilization: 0.9,
        maxModelLen: 4096,
        tensorParallelSize: 1,
        targetConcurrentRequests: 1000,
      },
    });

    const diagnostics = calculateDiagnostics({
      capacity,
      servingConfig: input.servingConfig,
    });

    expect(diagnostics.map((diagnostic) => diagnostic.id)).toContain(
      'target_concurrency_exceeds_capacity'
    );
  });

  it('detects low headroom from high GPU memory utilization', () => {
    const { input, capacity } = createCapacity({
      servingConfig: {
        gpuCount: 1,
        gpuMemoryUtilization: 1,
        maxModelLen: 4096,
        tensorParallelSize: 1,
        targetConcurrentRequests: 1,
      },
    });

    const diagnostics = calculateDiagnostics({
      capacity,
      servingConfig: input.servingConfig,
    });

    expect(diagnostics.map((diagnostic) => diagnostic.id)).toContain(
      'low_headroom'
    );
  });

  it('detects invalid GPU memory utilization', () => {
    const { input, capacity } = createCapacity({
      servingConfig: {
        gpuCount: 1,
        gpuMemoryUtilization: 1.2,
        maxModelLen: 4096,
        tensorParallelSize: 1,
        targetConcurrentRequests: 1,
      },
    });

    const diagnostics = calculateDiagnostics({
      capacity,
      servingConfig: input.servingConfig,
    });

    expect(diagnostics.map((diagnostic) => diagnostic.id)).toContain(
      'invalid_gpu_memory_utilization'
    );
  });

  it('detects tensor parallel size greater than GPU count', () => {
    const { input, capacity } = createCapacity({
      servingConfig: {
        gpuCount: 1,
        gpuMemoryUtilization: 0.9,
        maxModelLen: 4096,
        tensorParallelSize: 2,
        targetConcurrentRequests: 1,
      },
    });

    const diagnostics = calculateDiagnostics({
      capacity,
      servingConfig: input.servingConfig,
    });

    expect(diagnostics.map((diagnostic) => diagnostic.id)).toContain(
      'tensor_parallel_greater_than_gpu_count'
    );
  });

  it('returns an informational diagnostic when no blocking issues are detected', () => {
    const { input, capacity } = createCapacity();

    const diagnostics = calculateDiagnostics({
      capacity,
      servingConfig: input.servingConfig,
    });

    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0].id).toBe('no_blocking_fit_issues');
    expect(diagnostics[0].severity).toBe('info');
  });
});

describe('buildVllmServeCommand', () => {
  it('includes tensor parallel size, max model length, utilization, and KV cache dtype', () => {
    const command = buildVllmServeCommand({
      modelIdentifier: '<model-or-path>',
      tensorParallelSize: 2,
      maxModelLen: 8192,
      gpuMemoryUtilization: 0.9,
      kvCacheQuant: 'FP8',
    });

    expect(command).toContain('vllm serve <model-or-path>');
    expect(command).toContain('--tensor-parallel-size 2');
    expect(command).toContain('--max-model-len 8192');
    expect(command).toContain('--gpu-memory-utilization 0.90');
    expect(command).toContain('--kv-cache-dtype fp8');
  });

  it('updates command values deterministically', () => {
    const command = buildVllmServeCommand({
      tensorParallelSize: 4,
      maxModelLen: 16384,
      gpuMemoryUtilization: 0.85,
      kvCacheQuant: 'F16',
    });

    expect(command).toContain('--tensor-parallel-size 4');
    expect(command).toContain('--max-model-len 16384');
    expect(command).toContain('--gpu-memory-utilization 0.85');
    expect(command).toContain('--kv-cache-dtype auto');
  });
});

describe('getVllmKvCacheDtype', () => {
  it('maps FP8-like KV cache settings to fp8 and others to auto', () => {
    expect(getVllmKvCacheDtype('FP8')).toBe('fp8');
    expect(getVllmKvCacheDtype('Q8')).toBe('fp8');
    expect(getVllmKvCacheDtype('F16')).toBe('auto');
  });
});
