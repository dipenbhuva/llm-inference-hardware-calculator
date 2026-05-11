import {
  type KvCacheQuantization,
  type ModelArchitecture,
  type ModelQuantization,
  type ServingCapacity,
  type ServingConfig,
} from '../types';
import { calculateKvCacheMemoryGb } from './kvCache';
import { getModelQuantFactor } from './memory';

interface ServingCapacityInput {
  params: number;
  modelQuant: ModelQuantization;
  kvCacheQuant: KvCacheQuantization;
  gpuVram: number;
  architecture: ModelArchitecture;
  servingConfig: ServingConfig;
}

export const calculateServingCapacity = ({
  params,
  modelQuant,
  kvCacheQuant,
  gpuVram,
  architecture,
  servingConfig,
}: ServingCapacityInput): ServingCapacity => {
  const tensorParallelSize = Math.max(
    1,
    Math.min(servingConfig.tensorParallelSize, servingConfig.gpuCount)
  );
  const usableGpuMemoryGb = gpuVram * servingConfig.gpuMemoryUtilization;
  const usableTotalMemoryGb = usableGpuMemoryGb * tensorParallelSize;
  const modelWeightsGb = params * getModelQuantFactor(modelQuant);
  const modelWeightsPerGpuGb = modelWeightsGb / tensorParallelSize;
  const remainingKvCacheMemoryGb = Math.max(
    0,
    usableTotalMemoryGb - modelWeightsGb
  );
  const kvBytesPerToken =
    calculateKvCacheMemoryGb({
      architecture,
      contextLength: 1,
      kvCacheQuant,
      concurrentRequests: 1,
    }) * 1e9;
  const maxKvTokens =
    kvBytesPerToken > 0
      ? Math.floor((remainingKvCacheMemoryGb * 1e9) / kvBytesPerToken)
      : 0;
  const maxConcurrentRequests =
    servingConfig.maxModelLen > 0
      ? Math.floor(maxKvTokens / servingConfig.maxModelLen)
      : 0;
  const weightsFit = modelWeightsPerGpuGb <= usableGpuMemoryGb;

  return {
    usableGpuMemoryGb,
    usableTotalMemoryGb,
    modelWeightsGb,
    modelWeightsPerGpuGb,
    remainingKvCacheMemoryGb,
    kvBytesPerToken,
    maxKvTokens,
    maxConcurrentRequests,
    weightsFit,
    targetConcurrencyFits:
      weightsFit &&
      maxConcurrentRequests >= servingConfig.targetConcurrentRequests,
  };
};
