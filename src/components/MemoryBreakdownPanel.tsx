import { type MemoryBreakdown } from '../types';

interface MemoryBreakdownPanelProps {
  breakdown: MemoryBreakdown;
}

const formatGb = (value: number): string => `${value.toFixed(2)} GB`;

export const MemoryBreakdownPanel = ({
  breakdown,
}: MemoryBreakdownPanelProps) => {
  return (
    <div className="memory-breakdown">
      <h3>Memory Breakdown</h3>
      <dl>
        <div>
          <dt>Model weights</dt>
          <dd>{formatGb(breakdown.modelWeightsGb)}</dd>
        </div>
        <div>
          <dt>KV cache</dt>
          <dd>{formatGb(breakdown.kvCacheGb)}</dd>
        </div>
        <div>
          <dt>Activations/runtime</dt>
          <dd>{formatGb(breakdown.activationGb)}</dd>
        </div>
        <div>
          <dt>Safety overhead</dt>
          <dd>{formatGb(breakdown.runtimeOverheadGb)}</dd>
        </div>
        <div className="memory-breakdown-total">
          <dt>Total</dt>
          <dd>{formatGb(breakdown.totalGb)}</dd>
        </div>
      </dl>
    </div>
  );
};
