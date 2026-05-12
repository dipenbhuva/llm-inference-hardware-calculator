import {
  type KvCacheQuantization,
  type ModelArchitecture,
  type ModelQuantization,
  type ScalingPlan,
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

export const calculateScalingPlan = ({
  params,
  modelQuant,
  kvCacheQuant,
  gpuVram,
  architecture,
  servingConfig,
}: ServingCapacityInput): ScalingPlan => {
  const usableGpuMemoryGb = gpuVram * servingConfig.gpuMemoryUtilization;
  const modelWeightsGb = params * getModelQuantFactor(modelQuant);
  const minimumTensorParallelSize =
    usableGpuMemoryGb > 0
      ? Math.max(1, Math.ceil(modelWeightsGb / usableGpuMemoryGb))
      : Number.POSITIVE_INFINITY;
  const canFitOneReplica = minimumTensorParallelSize <= servingConfig.gpuCount;
  const recommendedTensorParallelSize = canFitOneReplica
    ? minimumTensorParallelSize
    : Math.max(1, servingConfig.gpuCount);

  if (!canFitOneReplica) {
    return {
      minimumTensorParallelSize,
      recommendedTensorParallelSize,
      maxConcurrentRequestsPerReplica: 0,
      replicasNeeded: 0,
      totalGpusNeeded: 0,
      canFitOneReplica,
    };
  }

  const perReplicaCapacity = calculateServingCapacity({
    params,
    modelQuant,
    kvCacheQuant,
    gpuVram,
    architecture,
    servingConfig: {
      ...servingConfig,
      tensorParallelSize: recommendedTensorParallelSize,
    },
  });
  const maxConcurrentRequestsPerReplica =
    perReplicaCapacity.maxConcurrentRequests;
  const replicasNeeded =
    maxConcurrentRequestsPerReplica > 0
      ? Math.max(
          1,
          Math.ceil(
            servingConfig.targetConcurrentRequests /
              maxConcurrentRequestsPerReplica
          )
        )
      : 0;

  return {
    minimumTensorParallelSize,
    recommendedTensorParallelSize,
    maxConcurrentRequestsPerReplica,
    replicasNeeded,
    totalGpusNeeded: recommendedTensorParallelSize * replicasNeeded,
    canFitOneReplica,
  };
};
