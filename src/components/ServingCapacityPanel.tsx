import { type ServingCapacity } from '../types';

interface ServingCapacityPanelProps {
  capacity: ServingCapacity;
}

const formatGb = (value: number): string => `${value.toFixed(2)} GB`;

export const ServingCapacityPanel = ({
  capacity,
}: ServingCapacityPanelProps) => {
  return (
    <div className="serving-capacity">
      <h3>vLLM Fit Estimate</h3>
      <dl>
        <div>
          <dt>Usable memory per GPU</dt>
          <dd>{formatGb(capacity.usableGpuMemoryGb)}</dd>
        </div>
        <div>
          <dt>Weights per GPU</dt>
          <dd>{formatGb(capacity.modelWeightsPerGpuGb)}</dd>
        </div>
        <div>
          <dt>Remaining KV memory</dt>
          <dd>{formatGb(capacity.remainingKvCacheMemoryGb)}</dd>
        </div>
        <div>
          <dt>KV token capacity</dt>
          <dd>{capacity.maxKvTokens.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Max concurrent requests</dt>
          <dd>{capacity.maxConcurrentRequests.toLocaleString()}</dd>
        </div>
      </dl>
      <p
        className={
          capacity.weightsFit ? 'capacity-status pass' : 'capacity-status fail'
        }
      >
        {capacity.weightsFit ? 'Weights fit' : 'Weights exceed usable memory'}
      </p>
      <p
        className={
          capacity.targetConcurrencyFits
            ? 'capacity-status pass'
            : 'capacity-status fail'
        }
      >
        {capacity.targetConcurrencyFits
          ? 'Target concurrency fits'
          : 'Target concurrency exceeds KV capacity'}
      </p>
    </div>
  );
};
