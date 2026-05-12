# Lab 07: Tensor Parallel vs Replicas

Status: Done

## Objective

Teach the difference between using multiple GPUs to fit one model and using replicas to serve more users.

## Files

- `src/calculations/servingCapacity.ts`
- `src/calculations/servingCapacity.test.ts`
- `src/components/ScalingPlanPanel.tsx`
- `src/App.tsx`
- `src/types.ts`

## Run

```bash
npm run dev -- --host 127.0.0.1
```

## Unit Tests

```bash
npm run test -- src/calculations/servingCapacity.test.ts
```

Key cases:

- Large model recommends tensor parallelism.
- Small model with high concurrency recommends replicas.
- Total GPUs equals tensor parallel size times replicas.
- Too-small GPU pool cannot fit one replica.

## Runtime Test

Use PRD runtime test `RT-L08`.

Manual path:

1. Configure a large model that does not fit on one GPU.
2. Confirm recommended tensor parallel size increases.
3. Configure a smaller model.
4. Increase target concurrency.
5. Confirm replicas increase.

## Reflection

- Why do replicas not increase VRAM for one model replica?
- When should you add tensor parallelism instead of replicas?
