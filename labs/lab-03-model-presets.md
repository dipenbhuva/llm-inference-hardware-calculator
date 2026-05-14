# Lab 03: Model Templates

Status: Done

## Objective

Let students start from realistic model-size templates instead of manually entering every architecture field.

## Files

- `src/data/modelPresets.ts`
- `src/data/modelPresets.test.ts`
- `src/App.tsx`
- `src/types.ts`

## Run

```bash
npm run dev -- --host 127.0.0.1
```

## Unit Tests

```bash
npm run test -- src/data/modelPresets.test.ts
```

Key cases:

- Preset IDs are unique.
- Preset architecture values are positive.
- Presets can be looked up by ID.

## Runtime Test

Use PRD runtime test `RT-L03`.

Manual path:

1. Select `7B reference template`.
2. Confirm parameter and architecture fields populate.
3. Select `70B reference template`.
4. Confirm VRAM increases.
5. Switch back to `Custom model architecture`.
6. Confirm manual editing still works.

## Reflection

- Which preset fields affect model weights?
- Which preset fields affect KV cache?
