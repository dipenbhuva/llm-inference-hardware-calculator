# Lab 08: Serving Command Generator

Status: Done

## Objective

Generate a starter vLLM command from the sizing configuration.

## Files

- `src/calculations/recommendations.ts`
- `src/calculations/recommendations.test.ts`
- `src/components/ServingCommandPanel.tsx`
- `src/App.tsx`
- `src/types.ts`

## Run

```bash
npm run dev -- --host 127.0.0.1
```

## Unit Tests

```bash
npm run test -- src/calculations/recommendations.test.ts
```

Key cases:

- Command includes `--tensor-parallel-size`.
- Command includes `--max-model-len`.
- Command includes `--gpu-memory-utilization`.
- FP8-like KV cache settings emit `--kv-cache-dtype fp8`.

## Runtime Test

Use PRD runtime test `RT-L06`.

Manual path:

1. Set tensor parallel size to `2`.
2. Set max model length to `8192`.
3. Set GPU memory utilization to `0.90`.
4. Set KV cache quantization to `FP8`.
5. Confirm the command reflects those values.

## Reflection

- Why is the generated command only a starting point?
- What should be benchmarked after running it?
