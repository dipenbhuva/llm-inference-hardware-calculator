# Lab 04: GPU Presets

Status: Done

## Objective

Teach how GPU VRAM affects whether a model fits and how many GPUs are needed.

## Files

- `src/data/gpuPresets.ts`
- `src/data/gpuPresets.test.ts`
- `src/App.tsx`
- `src/types.ts`

## Run

```bash
npm run dev -- --host 127.0.0.1
```

## Unit Tests

```bash
npm run test -- src/data/gpuPresets.test.ts
```

Key cases:

- Preset IDs are unique.
- VRAM values are positive.
- RTX 4090 maps to `24 GB`.
- L40S maps to `48 GB`.

## Runtime Test

Use PRD runtime test `RT-L04`.

Manual path:

1. Select discrete GPU mode.
2. Select `RTX 4090 24GB`.
3. Configure a model above `24 GB`.
4. Select `NVIDIA L40S 48GB`.
5. Confirm recommendation improves.
6. Select `Custom GPU` and enter `96`.

## Reflection

- Why does more VRAM help model fit but not automatically solve throughput?
