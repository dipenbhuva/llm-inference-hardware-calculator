import { type ScalingPlan } from '../types';

interface ScalingPlanPanelProps {
  plan: ScalingPlan;
}

export const ScalingPlanPanel = ({ plan }: ScalingPlanPanelProps) => {
  return (
    <div className="scaling-plan">
      <h3>Tensor Parallel vs Replicas</h3>
      <dl>
        <div>
          <dt>Minimum GPUs to fit one model</dt>
          <dd>{formatCount(plan.minimumTensorParallelSize)}</dd>
        </div>
        <div>
          <dt>Recommended tensor parallel size</dt>
          <dd>{formatCount(plan.recommendedTensorParallelSize)}</dd>
        </div>
        <div>
          <dt>Max concurrency per replica</dt>
          <dd>{plan.maxConcurrentRequestsPerReplica.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Replicas needed</dt>
          <dd>{plan.replicasNeeded.toLocaleString()}</dd>
        </div>
        <div className="memory-breakdown-total">
          <dt>Total GPUs needed</dt>
          <dd>{plan.totalGpusNeeded.toLocaleString()}</dd>
        </div>
      </dl>
      <p
        className={
          plan.canFitOneReplica ? 'capacity-status pass' : 'capacity-status fail'
        }
      >
        {plan.canFitOneReplica
          ? 'Use tensor parallelism to fit one replica, then replicas to scale concurrency.'
          : 'This GPU pool cannot fit one model replica.'}
      </p>
    </div>
  );
};

const formatCount = (value: number): string => {
  return Number.isFinite(value) ? value.toLocaleString() : 'Not feasible';
};
