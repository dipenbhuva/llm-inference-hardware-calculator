# Lab 06: OOM Diagnostics

Status: Done

## Objective

Generate actionable diagnostics for configurations that do not fit or leave risky headroom.

## Files

- `src/calculations/recommendations.ts`
- `src/calculations/recommendations.test.ts`
- `src/components/DiagnosticsPanel.tsx`
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

- Model weights exceed usable VRAM.
- KV cache exceeds remaining VRAM.
- Target concurrency exceeds capacity.
- Tensor parallel size exceeds GPU count.
- GPU memory utilization is invalid or too high.

## Runtime Test

Use PRD runtime test `RT-L07`.

Manual path:

1. Select `70B F16` style settings on a `24GB` GPU.
2. Confirm a model-weights diagnostic appears.
3. Increase GPU count or tensor parallel size.
4. Increase max model length until KV cache warning appears.
5. Set utilization to `1.0`.
6. Confirm low-headroom warning appears.

## Reflection

- Which diagnostic should be fixed first: weights fit, KV cache, or concurrency?
