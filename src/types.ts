/** Memory mode: discrete GPU or unified memory. */
export type MemoryMode = 'DISCRETE_GPU' | 'UNIFIED_MEMORY';

/** Model quantization set: F32, F16, Q8, Q6, Q5, Q4, Q3, Q2, GPTQ, AWQ. */
export type ModelQuantization =
  | 'F32'
  | 'F16'
  | 'Q8'
  | 'Q6'
  | 'Q5'
  | 'Q4'
  | 'Q3'
  | 'Q2'
  | 'GPTQ'
  | 'AWQ';

/** KV cache quantization: F32, F16, Q8, Q5, Q4. */
export type KvCacheQuantization = 'F32' | 'F16' | 'Q8' | 'Q5' | 'Q4';

/** Inference style: token-by-token generation or full-context pass. */
export type InferenceMode = 'incremental' | 'bulk';

/** VRAM estimate split into visible components. */
export interface MemoryBreakdown {
  modelWeightsGb: number;
  kvCacheGb: number;
  activationGb: number;
  runtimeOverheadGb: number;
  totalGb: number;
}

/** Transformer architecture fields needed for architecture-level KV cache math. */
export interface ModelArchitecture {
  layers: number;
  hiddenSize: number;
  attentionHeads: number;
  kvHeads: number;
  headDim: number;
}

/** Selectable model architecture preset. */
export interface ModelPreset extends ModelArchitecture {
  id: string;
  name: string;
  paramsBillion: number;
  defaultContextLength: number;
}

/** Selectable GPU preset for discrete inference hardware sizing. */
export interface GpuPreset {
  id: string;
  name: string;
  vramGb: number;
  class: 'consumer' | 'datacenter' | 'custom';
}

/** Serving settings used to estimate vLLM-style model fit and KV capacity. */
export interface ServingConfig {
  gpuCount: number;
  gpuMemoryUtilization: number;
  maxModelLen: number;
  tensorParallelSize: number;
  targetConcurrentRequests: number;
}

/** Capacity result for one tensor-parallel model replica. */
export interface ServingCapacity {
  usableGpuMemoryGb: number;
  usableTotalMemoryGb: number;
  modelWeightsGb: number;
  modelWeightsPerGpuGb: number;
  remainingKvCacheMemoryGb: number;
  kvBytesPerToken: number;
  maxKvTokens: number;
  maxConcurrentRequests: number;
  weightsFit: boolean;
  targetConcurrencyFits: boolean;
}

export type DiagnosticSeverity = 'info' | 'warning' | 'critical';

export type DiagnosticId =
  | 'weights_exceed_usable_vram'
  | 'kv_cache_exceeds_remaining_vram'
  | 'target_concurrency_exceeds_capacity'
  | 'low_headroom'
  | 'tensor_parallel_greater_than_gpu_count'
  | 'invalid_gpu_memory_utilization'
  | 'no_blocking_fit_issues';

/** Data-only diagnostic for invalid or risky serving configurations. */
export interface DiagnosticMessage {
  id: DiagnosticId;
  severity: DiagnosticSeverity;
  title: string;
  detail: string;
  action: string;
}

/** Recommendation for final output. */
export interface Recommendation {
  gpuType: string; // e.g., 'Single 24GB GPU' or 'Unified memory...'
  vramNeeded: string; // e.g., "32.5"
  fitsUnified: boolean; // relevant if memoryMode = 'UNIFIED_MEMORY'
  systemRamNeeded: number; // in GB
  gpusRequired: number; // discrete GPUs required (0 if doesn't fit)
}
