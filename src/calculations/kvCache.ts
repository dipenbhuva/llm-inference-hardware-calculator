import {
  type KvCacheQuantization,
  type ModelArchitecture,
} from '../types';

interface KvCacheMemoryInput {
  architecture: ModelArchitecture;
  contextLength: number;
  kvCacheQuant: KvCacheQuantization;
  concurrentRequests: number;
}

export const calculateKvCacheMemoryGb = ({
  architecture,
  contextLength,
  kvCacheQuant,
  concurrentRequests,
}: KvCacheMemoryInput): number => {
  const bytesPerElement = getKvCacheBytesPerElement(kvCacheQuant);
  const kvCacheBytes =
    2 *
    architecture.layers *
    architecture.kvHeads *
    architecture.headDim *
    bytesPerElement *
    contextLength *
    concurrentRequests;

  return kvCacheBytes / 1e9;
};

const getKvCacheBytesPerElement = (kvCacheQuant: KvCacheQuantization): number => {
  switch (kvCacheQuant) {
    case 'F32':
      return 4;
    case 'F16':
      return 2;
    case 'FP8':
      return 1;
    case 'Q8':
      return 1;
    case 'Q5':
      return 0.625;
    case 'Q4':
      return 0.5;
    default:
      return 1;
  }
};
