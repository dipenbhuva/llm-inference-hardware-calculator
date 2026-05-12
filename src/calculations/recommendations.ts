import {
  type DiagnosticMessage,
  type KvCacheQuantization,
  type ServingCapacity,
  type ServingConfig,
} from '../types';

interface DiagnosticsInput {
  capacity: ServingCapacity;
  servingConfig: ServingConfig;
}

export const calculateDiagnostics = ({
  capacity,
  servingConfig,
}: DiagnosticsInput): DiagnosticMessage[] => {
  const diagnostics: DiagnosticMessage[] = [];

  if (
    servingConfig.gpuMemoryUtilization <= 0 ||
    servingConfig.gpuMemoryUtilization > 1
  ) {
    diagnostics.push({
      id: 'invalid_gpu_memory_utilization',
      severity: 'critical',
      title: 'Invalid GPU memory utilization',
      detail: 'GPU memory utilization must be greater than 0 and no more than 1.',
      action: 'Set utilization between 0.80 and 0.95 for a realistic serving estimate.',
    });
  } else if (servingConfig.gpuMemoryUtilization >= 0.95) {
    diagnostics.push({
      id: 'low_headroom',
      severity: 'warning',
      title: 'Low runtime headroom',
      detail:
        'This configuration leaves little VRAM for allocator fragmentation, CUDA graphs, and runtime buffers.',
      action: 'Use a lower GPU memory utilization value or select a GPU with more VRAM.',
    });
  }

  if (servingConfig.tensorParallelSize > servingConfig.gpuCount) {
    diagnostics.push({
      id: 'tensor_parallel_greater_than_gpu_count',
      severity: 'critical',
      title: 'Tensor parallel size exceeds GPU count',
      detail:
        'A single model replica cannot be split across more GPUs than are available.',
      action: 'Increase GPU count or lower tensor parallel size.',
    });
  }

  if (!capacity.weightsFit) {
    diagnostics.push({
      id: 'weights_exceed_usable_vram',
      severity: 'critical',
      title: 'Model weights do not fit',
      detail:
        'The model weights alone exceed usable memory per GPU for the selected tensor parallel size.',
      action:
        'Use a smaller model, lower model quantization, add tensor parallel GPUs, or select a GPU with more VRAM.',
    });
  }

  const requiredKvMemoryForTargetGb =
    (capacity.kvBytesPerToken *
      servingConfig.maxModelLen *
      servingConfig.targetConcurrentRequests) /
    1e9;

  if (
    capacity.weightsFit &&
    requiredKvMemoryForTargetGb > capacity.remainingKvCacheMemoryGb
  ) {
    diagnostics.push({
      id: 'kv_cache_exceeds_remaining_vram',
      severity: 'critical',
      title: 'KV cache exceeds remaining VRAM',
      detail:
        'The selected max model length and target concurrency need more KV cache memory than remains after loading model weights.',
      action:
        'Reduce max model length, reduce target concurrency, lower KV cache precision, or add tensor parallel GPUs.',
    });
  }

  if (
    capacity.weightsFit &&
    servingConfig.targetConcurrentRequests > capacity.maxConcurrentRequests
  ) {
    diagnostics.push({
      id: 'target_concurrency_exceeds_capacity',
      severity: 'warning',
      title: 'Target concurrency exceeds capacity',
      detail:
        'The estimated KV token capacity cannot support the requested concurrent requests at the selected max model length.',
      action:
        'Lower target concurrency per replica, add replicas, reduce max model length, or increase available GPU memory.',
    });
  }

  if (diagnostics.length === 0) {
    diagnostics.push({
      id: 'no_blocking_fit_issues',
      severity: 'info',
      title: 'No blocking fit issues detected',
      detail:
        'Model weights and target KV cache fit within the current serving estimate.',
      action: 'Validate with an actual serving benchmark before treating this as production capacity.',
    });
  }

  return diagnostics;
};

interface VllmCommandInput {
  modelIdentifier?: string;
  tensorParallelSize: number;
  maxModelLen: number;
  gpuMemoryUtilization: number;
  kvCacheQuant: KvCacheQuantization;
}

export const buildVllmServeCommand = ({
  modelIdentifier = '<model-or-path>',
  tensorParallelSize,
  maxModelLen,
  gpuMemoryUtilization,
  kvCacheQuant,
}: VllmCommandInput): string => {
  return [
    `vllm serve ${modelIdentifier}`,
    `  --tensor-parallel-size ${tensorParallelSize}`,
    `  --max-model-len ${maxModelLen}`,
    `  --gpu-memory-utilization ${gpuMemoryUtilization.toFixed(2)}`,
    `  --kv-cache-dtype ${getVllmKvCacheDtype(kvCacheQuant)}`,
  ].join(' \\\n');
};

export const getVllmKvCacheDtype = (
  kvCacheQuant: KvCacheQuantization
): string => {
  switch (kvCacheQuant) {
    case 'FP8':
    case 'Q8':
      return 'fp8';
    case 'F16':
    case 'F32':
    case 'Q5':
    case 'Q4':
      return 'auto';
    default:
      return 'auto';
  }
};
