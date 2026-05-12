# Lab 05: vLLM Fit Mode

Status: Done

## Objective

Map calculator inputs to vLLM-style serving capacity: usable GPU memory, tensor parallel size, KV capacity, and max concurrency.

## Files

- `src/calculations/servingCapacity.ts`
- `src/calculations/servingCapacity.test.ts`
- `src/components/ServingCapacityPanel.tsx`
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

- Lower GPU utilization lowers usable memory.
- Higher tensor parallel size lowers weights per GPU.
- Higher max model length lowers max concurrency.
- Weight fit and target concurrency fit are separate checks.

## Runtime Test

Use PRD runtime test `RT-L05`.

Manual path:

1. Set GPU memory utilization to `0.90`.
2. Record usable memory and max concurrency.
3. Lower utilization to `0.80`.
4. Confirm usable memory falls.
5. Increase max model length.
6. Confirm max concurrency falls.

## Reflection

- Why is `gpu-memory-utilization` not usually `1.0` in production?
- Why does longer context reduce concurrency?
