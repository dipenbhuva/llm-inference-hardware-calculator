# Lab 02: Real KV Cache Math

Status: Done

## Objective

Replace rough KV cache estimates with architecture-level math using layers, KV heads, head dimension, context length, dtype, and concurrency.

## Files

- `src/calculations/kvCache.ts`
- `src/calculations/memory.ts`
- `src/App.tsx`
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

- Doubling context length doubles KV memory.
- Doubling concurrent requests doubles KV memory.
- `F16` KV cache uses twice the memory of `Q8`/`FP8`.
- KV cache uses `kvHeads`, not total attention heads.

## Runtime Test

Use PRD runtime test `RT-L02`.

Manual path:

1. Set layers `32`, attention heads `32`, KV heads `8`, head dimension `128`.
2. Enable KV cache.
3. Record KV cache at context `4096`.
4. Change context to `8192`.
5. Confirm KV cache approximately doubles.
6. Change concurrent requests from `1` to `2`.
7. Confirm KV cache approximately doubles again.

## Reflection

- Why do grouped-query attention models have lower KV memory?
- Why does concurrency affect VRAM even if the model weights stay fixed?
