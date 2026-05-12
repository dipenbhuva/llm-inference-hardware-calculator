import {
  InferenceMode,
  KvCacheQuantization,
  MemoryMode,
  ModelArchitecture,
  ModelQuantization,
  Recommendation,
} from './types';
import {
  calculateRequiredVram,
  getModelQuantFactor,
} from './calculations/memory';

export { calculateKvCacheMemoryGb } from './calculations/kvCache';
export { calculateDiagnostics } from './calculations/recommendations';
export {
  calculateScalingPlan,
  calculateServingCapacity,
} from './calculations/servingCapacity';
export {
  calculateMemoryBreakdown,
  calculateOnDiskSize,
  calculateRequiredVram,
  getKvCacheQuantFactor,
  getModelQuantFactor,
} from './calculations/memory';

/**
 * Calculate hardware recommendation based on model parameters and system configuration.
 *
 * Changes made:
 *  - The system RAM calculation now includes an estimate for extra activation memory in bulk mode.
 *  - GPU requirement is calculated with a buffer multiplier (1.2) to account for fragmentation.
 *  - The logic now differentiates between unified memory and discrete GPU setups.
 */
export const calculateHardwareRecommendation = (
  params: number,
  modelQuant: ModelQuantization,
  contextLength: number,
  useKvCache: boolean,
  kvCacheQuant: KvCacheQuantization,
  memoryMode: MemoryMode,
  systemMemory: number,
  gpuVram: number,
  inferenceMode: InferenceMode,
  architecture?: ModelArchitecture,
  concurrentRequests = 1
): Recommendation => {
  const requiredVram = calculateRequiredVram(
    params,
    modelQuant,
    contextLength,
    useKvCache,
    kvCacheQuant,
    inferenceMode,
    architecture,
    concurrentRequests
  );

  // Adjust system RAM calculation to include additional bulk activation memory.
  // For incremental mode, assume half of base memory is needed.
  // For bulk mode, add extra memory proportional to the context length.
  const baseSystemRamNeeded =
    params * getModelQuantFactor(modelQuant) * 0.5 +
    (inferenceMode === 'bulk' ? contextLength / 1024 : 0);
  const systemRamNeeded = Math.max(8, baseSystemRamNeeded); // Ensure at least 8GB is recommended.

  // Determine if the required VRAM fits within unified memory mode.
  const fitsUnified =
    memoryMode === 'UNIFIED_MEMORY' && systemMemory >= requiredVram;

  // Calculate discrete GPU requirements.
  let gpusRequired = 0;
  let gpuType = '';

  if (memoryMode === 'DISCRETE_GPU') {
    // Use a buffer multiplier (1.2) to account for real-world fragmentation.
    gpusRequired = Math.ceil((requiredVram * 1.2) / gpuVram);
    if (gpusRequired === 1) {
      gpuType = `Single ${gpuVram}GB GPU`;
    } else if (gpusRequired <= 8) {
      gpuType = `${gpusRequired}x ${gpuVram}GB GPUs`;
    } else {
      // If more than 8 GPUs are needed, mark it as not feasible.
      gpuType = `Exceeds 8x ${gpuVram}GB GPUs`;
      gpusRequired = 0;
    }
  } else {
    // For unified memory mode, no discrete GPUs are needed.
    gpuType = `Unified memory (${systemMemory}GB)`;
    gpusRequired = 0;
  }

  return {
    gpuType,
    vramNeeded: requiredVram.toFixed(2),
    fitsUnified,
    systemRamNeeded,
    gpusRequired,
  };
};
