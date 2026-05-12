# Lab 01: Memory Breakdown

Status: Done

## Objective

Show students why total VRAM is made of model weights, KV cache, activations, and runtime overhead.

## Files

- `src/calculations/memory.ts`
- `src/components/MemoryBreakdownPanel.tsx`
- `src/types.ts`
- `src/calculations.test.ts`

## Run

```bash
npm run dev -- --host 127.0.0.1
```

## Unit Tests

```bash
npm run test -- src/calculations.test.ts
```

Key cases:

- `7B F16` model weights are about `14 GB`.
- `7B Q4` model weights are about `3.5 GB`.
- Enabling KV cache makes KV memory non-zero.
- Bulk mode increases activation memory.

## Runtime Test

Use PRD runtime test `RT-L01`.

Manual path:

1. Set parameters to `7`.
2. Set model quantization to `F16`.
3. Disable KV cache.
4. Confirm Memory Breakdown shows about `14 GB` model weights.
5. Switch quantization to `Q4`.
6. Confirm model weights drop to about `3.5 GB`.
7. Enable KV cache and raise context length.
8. Confirm KV cache changes.

## Reflection

- Why are model weights independent of context length?
- Why does KV cache grow with context length?
- Why should production calculators show the breakdown, not only the total?
