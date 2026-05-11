import {
  InferenceMode,
  KvCacheQuantization,
  MemoryBreakdown,
  ModelQuantization,
} from '../types';

/**
 * Returns GB factor for a 1B param model based on quantization.
 * This function converts the quantization setting into an estimate of how many
 * gigabytes are required per 1B parameters. For example, FP16 (F16) uses 2GB per 1B.
 */
export const getModelQuantFactor = (q: ModelQuantization): number => {
  switch (q) {
    case 'F32':
      return 4.0;
    case 'F16':
      return 2.0;
    case 'Q8':
      return 1.0;
    case 'Q6':
      return 0.75;
    case 'Q5':
      return 0.625;
    case 'Q4':
      return 0.5;
    case 'Q3':
      return 0.375;
    case 'Q2':
      return 0.25;
    case 'GPTQ':
      return 0.4; // approximate factor for GPTQ quantization
    case 'AWQ':
      return 0.35; // approximate factor for AWQ quantization
    default:
      return 1.0;
  }
};

/**
 * Returns GB factor for KV cache usage (per 1B params), depending on quantization.
 * This is used to adjust the additional memory required when KV caching is enabled.
 */
export const getKvCacheQuantFactor = (k: KvCacheQuantization): number => {
  switch (k) {
    case 'F32':
      return 4.0;
    case 'F16':
      return 2.0;
    case 'Q8':
      return 1.0;
    case 'Q5':
      return 0.625;
    case 'Q4':
      return 0.5;
    default:
      return 1.0;
  }
};

/**
 * Calculate VRAM requirement (GB) for single-user inference and return the
 * individual memory components that make up the final estimate.
 */
export const calculateMemoryBreakdown = (
  params: number, // number of model parameters in *billions*
  modelQuant: ModelQuantization,
  contextLength: number,
  useKvCache: boolean,
  kvCacheQuant: KvCacheQuantization,
  inferenceMode: InferenceMode
): MemoryBreakdown => {
  const modelFactor = getModelQuantFactor(modelQuant);
  const modelWeightsGb = params * modelFactor;

  let kvCacheGb = 0;
  let activationGb = 0;

  if (inferenceMode === 'incremental') {
    if (useKvCache) {
      const alphaAt2048 = 0.2;
      const kvFactor = getKvCacheQuantFactor(kvCacheQuant);
      const kvScale = contextLength / 2048;
      kvCacheGb = modelWeightsGb * alphaAt2048 * kvScale * kvFactor;
    }
  } else {
    const bulkAlphaAt2048 = 0.5;
    const bulkScale = contextLength / 2048;
    activationGb = modelWeightsGb * bulkAlphaAt2048 * bulkScale;

    if (useKvCache) {
      const kvFactor = getKvCacheQuantFactor(kvCacheQuant);
      kvCacheGb = modelWeightsGb * 0.1 * kvFactor * bulkScale;
    }
  }

  const subtotalGb = modelWeightsGb + kvCacheGb + activationGb;
  const runtimeOverheadGb = subtotalGb * 0.1;

  return {
    modelWeightsGb,
    kvCacheGb,
    activationGb,
    runtimeOverheadGb,
    totalGb: subtotalGb + runtimeOverheadGb,
  };
};

export const calculateRequiredVram = (
  params: number,
  modelQuant: ModelQuantization,
  contextLength: number,
  useKvCache: boolean,
  kvCacheQuant: KvCacheQuantization,
  inferenceMode: InferenceMode
): number => {
  return calculateMemoryBreakdown(
    params,
    modelQuant,
    contextLength,
    useKvCache,
    kvCacheQuant,
    inferenceMode
  ).totalGb;
};

/**
 * Estimate on-disk model size.
 */
export const calculateOnDiskSize = (
  params: number,
  modelQuant: ModelQuantization
): number => {
  const modelFactor = getModelQuantFactor(modelQuant);
  const bitsPerParam = modelFactor * 8;
  const totalBits = params * 1e9 * bitsPerParam;
  return totalBits / 8 / 1e9;
};
