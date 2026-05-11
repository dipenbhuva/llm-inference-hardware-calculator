# GPU Hosting Labs PRD

Status: Draft
Owner: TBD
Repo: `llm-inference-hardware-calculator`
Last updated: 2026-05-11

## Purpose

Turn the current LLM inference hardware calculator into a lab-driven teaching tool for students learning how to size GPU hardware for hosting LLMs.

The app should remain focused on one production question:

> Given a model, quantization, context length, KV cache settings, concurrency target, and available GPUs, what hardware do I need to host this LLM?

## Current Repo Baseline

The current app is a client-side React + TypeScript + Vite calculator.

- UI entry point: `src/App.tsx`
- Domain math: `src/calculations.ts`
- Shared types: `src/types.ts`
- Theme state: `src/contexts/ThemeContext.tsx`
- Styling: `src/App.css`, `src/index.css`, component CSS files
- Build: `npm run build`
- Lint: `npm run lint`
- Deployment: static Vite build served through nginx in `Dockerfile`
- Tests: no test runner currently configured

Current calculator capabilities:

- Model parameter count in billions
- Model quantization factor
- Context length
- Optional KV cache
- KV cache quantization
- Incremental vs bulk inference mode
- Discrete GPU vs unified memory
- GPU VRAM size
- System memory
- Required VRAM
- On-disk size
- Number of GPUs using rough VRAM division

Current limitations:

- KV cache is estimated from base model memory instead of architecture-level KV math.
- There are no model architecture inputs such as layer count, KV heads, or head dimension.
- There are no model or GPU presets.
- GPU count does not distinguish tensor parallelism, pipeline parallelism, and replicas.
- The UI does not explain the memory breakdown.
- No vLLM-oriented serving configuration is generated.
- No automated tests protect calculation behavior.

## Product Goals

1. Teach how model weights, KV cache, activations, runtime overhead, context length, and concurrency contribute to VRAM.
2. Teach why "the model fits" is different from "the model serves production traffic."
3. Teach concrete GPU hosting tradeoffs: quantization, KV cache dtype, max model length, tensor parallelism, replicas, and headroom.
4. Produce practical outputs students can use in labs: memory breakdowns, diagnostics, GPU recommendations, and starter serving commands.
5. Keep calculations pure and testable so future labs can be implemented safely.

## Non-Goals

- Do not add authentication.
- Do not add a database.
- Do not add live cloud pricing in the first implementation phase.
- Do not call external APIs at runtime.
- Do not replace this with a generic AI app readiness checklist.
- Do not hide formulas behind unexplained magic constants.

## Architecture Direction

Keep the app static and client-side for now.

Recommended structure:

```txt
src/
  calculations/
    memory.ts
    kvCache.ts
    servingCapacity.ts
    recommendations.ts
  components/
    ModelConfigPanel.tsx
    GpuConfigPanel.tsx
    MemoryBreakdownPanel.tsx
    ServingCapacityPanel.tsx
    ServingCommandPanel.tsx
    DiagnosticsPanel.tsx
  data/
    modelPresets.ts
    gpuPresets.ts
    servingEnginePresets.ts
  types.ts
```

`src/App.tsx` should become orchestration only: hold state, call pure calculation functions, and compose panels.

Calculation modules should stay framework-free TypeScript so they can be unit tested without rendering React.

## Status Legend

- Not Started: no implementation exists yet
- In Progress: implementation started but not accepted
- Blocked: waiting on decision or dependency
- Done: implemented, tested, and documented

## Implementation Tracker

| ID | Lab | Status | Primary Files | Verification |
| --- | --- | --- | --- | --- |
| L01 | Memory Breakdown | Done | `src/calculations/memory.ts`, `src/components/MemoryBreakdownPanel.tsx`, `src/types.ts` | Unit tests, runtime test RT-L01 |
| L02 | Real KV Cache Math | Done | `src/calculations/kvCache.ts`, `src/types.ts`, `src/App.tsx` | Unit tests, runtime test RT-L02 |
| L03 | Model Presets | Done | `src/data/modelPresets.ts`, `src/App.tsx` | Unit tests, runtime test RT-L03 |
| L04 | GPU Presets | Done | `src/data/gpuPresets.ts`, `src/App.tsx` | Unit tests, runtime test RT-L04 |
| L05 | vLLM Fit Mode | Not Started | `src/calculations/servingCapacity.ts`, `src/components/ServingCapacityPanel.tsx` | Unit tests, runtime test RT-L05 |
| L06 | Serving Command Generator | Not Started | `src/components/ServingCommandPanel.tsx`, `src/calculations/recommendations.ts` | String tests, runtime test RT-L06 |
| L07 | Diagnostics/OOM Lab | Not Started | `src/calculations/recommendations.ts`, `src/components/DiagnosticsPanel.tsx` | Unit tests, runtime test RT-L07 |
| L08 | Tensor Parallel vs Replicas | Not Started | `src/calculations/servingCapacity.ts`, `src/types.ts` | Unit tests, runtime test RT-L08 |
| L09 | Student Lab Guides | Not Started | `labs/*.md`, `README.md` | Docs review, runtime test RT-L09 |
| L10 | Test Harness | Done | `package.json`, `vitest.config.ts`, test files | `npm run test`, runtime test RT-L10 |

## Lab Details

### L01: Memory Breakdown

Student objective:

Understand that total VRAM is not a single mystery number. It is a sum of weights, KV cache, activations, and overhead.

Implementation:

- Create `MemoryBreakdown` in `src/types.ts`.
- Refactor current `calculateRequiredVram` so it can return a breakdown.
- Keep compatibility with the current total VRAM result until the UI migration is complete.
- Add a panel that renders:
  - Model weights GB
  - KV cache GB
  - Activation/runtime GB
  - Safety overhead GB
  - Total VRAM GB

Suggested type:

```ts
export interface MemoryBreakdown {
  modelWeightsGb: number;
  kvCacheGb: number;
  activationGb: number;
  runtimeOverheadGb: number;
  totalGb: number;
}
```

Acceptance criteria:

- Existing user inputs still produce a total VRAM estimate.
- The breakdown numbers add up to the displayed total within rounding tolerance.
- Bulk mode shows activation memory.
- Incremental mode without KV cache shows zero or near-zero KV cache.

Test cases:

| Case | Input | Expected |
| --- | --- | --- |
| 7B F16, no KV, incremental | params `7`, quant `F16` | model weights approx `14 GB` before overhead |
| 7B Q4, no KV, incremental | params `7`, quant `Q4` | model weights approx `3.5 GB` before overhead |
| 70B Q4, KV enabled, 8192 context | params `70`, quant `Q4`, context `8192` | KV cache contributes non-zero memory |
| Bulk mode | inference mode `bulk` | activation memory is greater than incremental mode |

### L02: Real KV Cache Math

Student objective:

Learn that KV cache size depends on model architecture, context length, concurrency, and KV dtype.

Implementation:

- Add architecture fields:
  - `layers`
  - `hiddenSize`
  - `attentionHeads`
  - `kvHeads`
  - `headDim`
  - `concurrentRequests`
- Add `calculateKvCacheMemoryGb()` in `src/calculations/kvCache.ts`.
- Replace the rough `alphaAt2048` KV estimate after tests are in place.

Formula:

```txt
kvCacheBytes =
  2
  * layers
  * kvHeads
  * headDim
  * bytesPerKvElement
  * contextLength
  * concurrentRequests
```

The leading `2` accounts for key and value tensors.

Acceptance criteria:

- Increasing context length increases KV cache linearly.
- Increasing concurrent requests increases KV cache linearly.
- FP16 KV cache is twice Q8 KV cache.
- Models with grouped-query attention can use fewer KV heads than attention heads.

Test cases:

| Case | Input | Expected |
| --- | --- | --- |
| Context doubles | context `4096` -> `8192` | KV cache doubles |
| Concurrency doubles | concurrency `1` -> `2` | KV cache doubles |
| KV dtype changes | FP16 -> Q8 | KV cache halves |
| GQA model | attention heads `32`, KV heads `8` | KV memory uses `8` KV heads, not `32` attention heads |

### L03: Model Presets

Student objective:

Start from realistic model shapes instead of only parameter count.

Implementation:

- Add `src/data/modelPresets.ts`.
- Include generic presets first to avoid locking the app to fast-changing model catalogs.
- Presets should populate params and architecture fields.
- Keep manual mode for custom models.

Suggested presets:

```txt
Generic 7B
Generic 8B
Generic 13B
Generic 34B
Generic 70B
Generic 120B
```

Suggested type:

```ts
export interface ModelPreset {
  id: string;
  name: string;
  paramsBillion: number;
  layers: number;
  hiddenSize: number;
  attentionHeads: number;
  kvHeads: number;
  headDim: number;
  defaultContextLength: number;
}
```

Acceptance criteria:

- Selecting a preset updates all relevant model fields.
- Switching back to custom mode allows manual edits.
- Preset IDs are stable and unique.

Test cases:

| Case | Input | Expected |
| --- | --- | --- |
| Select Generic 7B | preset dropdown | params and architecture fields update |
| Select custom | custom mode | manual inputs are enabled |
| Preset validity | all presets | no zero or negative architecture values |

### L04: GPU Presets

Student objective:

Learn how VRAM differs across common inference GPUs.

Implementation:

- Add `src/data/gpuPresets.ts`.
- Replace hardcoded VRAM dropdown with GPU preset plus custom VRAM.
- Include consumer and datacenter GPUs.

Suggested presets:

```txt
RTX 4090 24GB
NVIDIA L4 24GB
NVIDIA A10G 24GB
NVIDIA L40S 48GB
NVIDIA A100 40GB
NVIDIA A100 80GB
NVIDIA H100 80GB
Custom GPU
```

Suggested type:

```ts
export interface GpuPreset {
  id: string;
  name: string;
  vramGb: number;
  class: 'consumer' | 'datacenter' | 'custom';
}
```

Acceptance criteria:

- Selecting a GPU preset updates GPU VRAM.
- Custom mode allows arbitrary GPU VRAM.
- Existing GPU recommendation output still works.

Test cases:

| Case | Input | Expected |
| --- | --- | --- |
| Select RTX 4090 | preset | GPU VRAM becomes `24` |
| Select L40S | preset | GPU VRAM becomes `48` |
| Custom GPU | custom value `96` | recommendation uses `96` |

### L05: vLLM Fit Mode

Student objective:

Map calculator concepts to vLLM-style serving settings.

Implementation:

- Add serving config inputs:
  - `gpuCount`
  - `gpuMemoryUtilization`
  - `maxModelLen`
  - `kvCacheDtype`
  - `tensorParallelSize`
- Calculate:
  - usable GPU memory
  - memory needed for weights per tensor-parallel group
  - remaining memory for KV cache
  - KV tokens capacity
  - max concurrent requests at `maxModelLen`

Suggested type:

```ts
export interface ServingConfig {
  gpuCount: number;
  gpuMemoryUtilization: number;
  maxModelLen: number;
  tensorParallelSize: number;
  targetConcurrentRequests: number;
}
```

Acceptance criteria:

- Lower `gpuMemoryUtilization` lowers available memory.
- Higher `tensorParallelSize` lowers model weight memory per GPU.
- Higher `maxModelLen` lowers maximum concurrency.
- App clearly distinguishes "fits model weights" from "fits target concurrency."

Test cases:

| Case | Input | Expected |
| --- | --- | --- |
| Utilization 0.90 vs 0.80 | same GPU | 0.90 has more usable memory |
| TP 1 vs TP 2 | same model | per-GPU weight memory decreases with TP 2 |
| Max length doubles | same KV capacity | max concurrency halves |
| Target concurrency too high | target > max | diagnostic warning appears |

### L06: Serving Command Generator

Student objective:

Turn sizing output into a first serving command to try.

Implementation:

- Add `ServingCommandPanel`.
- Generate a vLLM command from current state.
- Keep command generation deterministic and testable.

Example output:

```bash
vllm serve <model-or-path> \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8
```

Acceptance criteria:

- Command reflects selected tensor parallel size.
- Command reflects selected max model length.
- Command reflects selected KV cache dtype.
- Command warns that model ID/path must be replaced when using generic presets.

Test cases:

| Case | Input | Expected |
| --- | --- | --- |
| TP 2 | tensor parallel `2` | command contains `--tensor-parallel-size 2` |
| FP8 KV | KV dtype `fp8` | command contains `--kv-cache-dtype fp8` |
| Generic preset | no model path | command contains placeholder |

### L07: Diagnostics/OOM Lab

Student objective:

Learn how to diagnose why an LLM serving configuration fails to fit.

Implementation:

- Add diagnostic messages from calculation results.
- Diagnostics should be generated as data, not hardcoded JSX.

Suggested diagnostic IDs:

```txt
weights_exceed_usable_vram
kv_cache_exceeds_remaining_vram
target_concurrency_exceeds_capacity
low_headroom
tensor_parallel_greater_than_gpu_count
invalid_gpu_memory_utilization
```

Acceptance criteria:

- Diagnostics include severity: `info`, `warning`, `critical`.
- Each diagnostic includes a recommended action.
- Diagnostics update when inputs change.

Test cases:

| Case | Input | Expected |
| --- | --- | --- |
| 70B F16 on 24GB GPU, TP 1 | too large | `weights_exceed_usable_vram` |
| Long context with small remaining VRAM | context high | `kv_cache_exceeds_remaining_vram` |
| Target concurrency > capacity | high target | `target_concurrency_exceeds_capacity` |
| Utilization 1.0 | no headroom | `low_headroom` |

### L08: Tensor Parallel vs Replicas

Student objective:

Understand the difference between splitting one model across GPUs and running more copies of the model.

Implementation:

- Add capacity outputs:
  - minimum GPUs to fit one model
  - recommended tensor parallel size
  - max concurrent requests per replica
  - replicas needed for target concurrency
  - total GPUs needed

Acceptance criteria:

- If a model does not fit on one GPU, recommend tensor parallelism before replicas.
- If one replica fits but target concurrency is high, recommend more replicas.
- Total GPU count equals tensor parallel size times replicas.

Test cases:

| Case | Input | Expected |
| --- | --- | --- |
| Model too large for 1 GPU | 70B F16, 24GB | tensor parallel recommended |
| Model fits but target high | 7B Q4, high concurrency | replicas recommended |
| TP 2, replicas 4 | config | total GPUs `8` |

### L09: Student Lab Guides

Student objective:

Use the app as a structured lab sequence.

Implementation:

- Add `labs/lab-01-memory-breakdown.md`.
- Add one lab file per major feature.
- Each lab should include:
  - learning objective
  - starting branch or status
  - files to edit
  - implementation checklist
  - unit test cases
  - runtime test cases
  - reflection questions

Acceptance criteria:

- Labs can be followed independently.
- Each lab names exact files and expected commands.
- README links to the lab index.

### L10: Test Harness

Student objective:

Protect calculator formulas with automated tests.

Implementation:

- Add Vitest.
- Add test scripts.
- Add unit tests for calculation modules before deeper UI work.

Suggested install:

```bash
npm install -D vitest jsdom @testing-library/react @testing-library/jest-dom @testing-library/user-event
```

Suggested `package.json` scripts:

```json
{
  "test": "vitest run",
  "test:watch": "vitest",
  "test:coverage": "vitest run --coverage"
}
```

Acceptance criteria:

- `npm run test` passes locally.
- `npm run lint` passes.
- `npm run build` passes.
- CI runs tests before build.

## Runtime Test Protocol

Runtime tests are the browser-level checks that prove the calculator can actually be used after each lab. They are separate from unit tests.

For every implemented lab:

1. Start the app.

```bash
npm run dev -- --host 127.0.0.1
```

2. Open the local Vite URL printed by the command.
3. Run the matching runtime test from the matrix below.
4. Record the result in the lab guide or PR notes:

```txt
Runtime test: RT-L01
Browser: Chrome/Safari/Firefox
Viewport: desktop 1440x900, mobile 390x844 if UI changed
Result: Pass/Fail
Notes:
```

After L10 adds an automated browser runner, these runtime tests should become Playwright tests where practical. Until then, they are manual acceptance tests.

## Runtime Test Matrix

### RT-L01: Memory Breakdown Runtime Test

Purpose:

Verify that the UI explains total VRAM as a breakdown of model weights, KV cache, activations, and overhead.

Preconditions:

- L01 implemented.
- App is running with `npm run dev -- --host 127.0.0.1`.

Steps:

1. Set parameters to `7`.
2. Set model quantization to `F16`.
3. Set inference mode to `Incremental`.
4. Disable KV cache.
5. Confirm the memory breakdown panel is visible.
6. Change model quantization from `F16` to `Q4`.
7. Enable KV cache and set context length to `8192`.
8. Switch inference mode from `Incremental` to `Bulk`.

Expected result:

- Model weights show about `14 GB` for `7B F16` before overhead.
- Model weights drop to about `3.5 GB` for `7B Q4` before overhead.
- KV cache is zero or near zero when disabled.
- KV cache becomes non-zero when enabled.
- Bulk mode increases activation/runtime memory.
- Displayed total matches the visible components within rounding tolerance.
- No visible `NaN`, `Infinity`, or blank result fields appear.

### RT-L02: Real KV Cache Runtime Test

Purpose:

Verify that KV cache responds to architecture, context length, dtype, and concurrency.

Preconditions:

- L02 implemented.
- Architecture fields are visible or populated by a preset.

Steps:

1. Set layers to `32`.
2. Set attention heads to `32`.
3. Set KV heads to `8`.
4. Set head dimension to `128`.
5. Set context length to `4096`.
6. Set concurrent requests to `1`.
7. Set KV cache dtype to `F16`.
8. Record KV cache GB.
9. Change context length to `8192`.
10. Change concurrent requests to `2`.
11. Change KV cache dtype from `F16` to `Q8`.

Expected result:

- Doubling context length approximately doubles KV cache memory.
- Doubling concurrent requests approximately doubles KV cache memory.
- Changing KV dtype from `F16` to `Q8` approximately halves KV cache memory.
- KV memory uses KV heads, not attention heads.
- Updating any architecture field immediately updates the memory breakdown.

### RT-L03: Model Presets Runtime Test

Purpose:

Verify that model presets populate model architecture inputs consistently.

Preconditions:

- L03 implemented.

Steps:

1. Select `Generic 7B`.
2. Confirm parameter count and architecture fields are populated.
3. Select `Generic 70B`.
4. Confirm total VRAM increases.
5. Switch to `Custom`.
6. Edit parameter count and architecture fields manually.
7. Switch back to `Generic 7B`.

Expected result:

- Selecting a preset updates all model fields together.
- Larger presets produce larger model weight memory.
- Custom mode allows manual edits.
- Returning to a preset overwrites custom values with preset values.
- No architecture field is zero, negative, or blank after selecting a preset.

### RT-L04: GPU Presets Runtime Test

Purpose:

Verify that GPU presets drive hardware recommendation behavior.

Preconditions:

- L04 implemented.

Steps:

1. Select discrete GPU mode.
2. Select `RTX 4090 24GB`.
3. Configure a model that requires more than `24 GB` total VRAM.
4. Confirm recommendation requires more than one GPU or warns it does not fit.
5. Select `NVIDIA L40S 48GB`.
6. Confirm recommendation improves.
7. Select `Custom GPU`.
8. Enter `96 GB`.

Expected result:

- GPU VRAM changes to `24`, `48`, and `96` for the tested selections.
- Recommendations update immediately when GPU preset changes.
- Custom GPU VRAM is respected.
- No stale GPU name or stale VRAM value remains after switching presets.

### RT-L05: vLLM Fit Mode Runtime Test

Purpose:

Verify that the calculator distinguishes model fit, KV capacity, and target concurrency.

Preconditions:

- L05 implemented.

Steps:

1. Select a model preset or custom config that fits on the selected GPU.
2. Set GPU count to `1`.
3. Set tensor parallel size to `1`.
4. Set GPU memory utilization to `0.90`.
5. Set max model length to `4096`.
6. Record usable GPU memory, remaining KV memory, and max concurrency.
7. Change GPU memory utilization to `0.80`.
8. Change max model length to `8192`.
9. Increase tensor parallel size to `2` with GPU count `2`.

Expected result:

- Lower GPU memory utilization reduces usable GPU memory.
- Higher max model length reduces max concurrent requests.
- Higher tensor parallel size reduces per-GPU model weight memory.
- UI separately reports:
  - weights fit or do not fit
  - KV cache capacity
  - max concurrent requests
  - target concurrency pass/fail

### RT-L06: Serving Command Runtime Test

Purpose:

Verify that generated vLLM commands reflect current calculator inputs.

Preconditions:

- L06 implemented.

Steps:

1. Set tensor parallel size to `2`.
2. Set max model length to `8192`.
3. Set GPU memory utilization to `0.90`.
4. Set KV cache dtype to `fp8`.
5. View the serving command panel.
6. Change tensor parallel size to `4`.
7. Change max model length to `16384`.

Expected result:

- Command includes `--tensor-parallel-size 2`, then updates to `4`.
- Command includes `--max-model-len 8192`, then updates to `16384`.
- Command includes `--gpu-memory-utilization 0.90`.
- Command includes `--kv-cache-dtype fp8`.
- Generic presets produce a clear placeholder for model ID/path.
- Command text remains copyable and does not wrap in a way that hides flags.

### RT-L07: Diagnostics Runtime Test

Purpose:

Verify that OOM and invalid configuration diagnostics appear and clear correctly.

Preconditions:

- L07 implemented.

Steps:

1. Select a large model configuration such as `70B F16`.
2. Select a `24GB` GPU.
3. Set tensor parallel size to `1`.
4. Confirm a weights-fit diagnostic appears.
5. Increase tensor parallel size or GPU VRAM until weights fit.
6. Set a very large context length.
7. Confirm a KV cache diagnostic appears.
8. Set GPU memory utilization to an invalid value if the UI allows manual entry.
9. Return inputs to valid values.

Expected result:

- Critical diagnostics appear for impossible configurations.
- Warnings appear for low headroom or target concurrency misses.
- Each diagnostic includes a recommended action.
- Diagnostics clear when the configuration becomes valid.
- Invalid numeric inputs do not break the page.

### RT-L08: Tensor Parallel vs Replicas Runtime Test

Purpose:

Verify that the app distinguishes fitting one model from scaling traffic.

Preconditions:

- L08 implemented.

Steps:

1. Configure a model too large for one selected GPU.
2. Confirm app recommends tensor parallelism to fit the model.
3. Configure a small model that fits on one GPU.
4. Set target concurrency higher than one replica can serve.
5. Confirm app recommends replicas.
6. Set tensor parallel size to `2` and replicas to `4`, or use equivalent recommended output.

Expected result:

- Model-fit recommendations use tensor parallelism.
- Traffic-scale recommendations use replicas.
- Total GPU count equals tensor parallel size times replicas.
- UI does not describe replicas as increasing VRAM for a single model.

### RT-L09: Lab Guides Runtime Test

Purpose:

Verify that students can follow lab docs and run the app.

Preconditions:

- L09 implemented.

Steps:

1. Open the README.
2. Follow the link to the lab index.
3. Open one lab guide.
4. Confirm it includes objective, files to edit, implementation checklist, unit tests, runtime tests, and reflection questions.
5. Start the app using the command from the guide.

Expected result:

- README links to the lab index.
- Lab guide commands match actual package scripts.
- Runtime test instructions are specific enough to execute without guessing.

### RT-L10: Test Harness Runtime Test

Purpose:

Verify that the repo has repeatable validation commands before deeper labs begin.

Preconditions:

- L10 implemented.

Steps:

1. Run `npm run test`.
2. Run `npm run lint`.
3. Run `npm run build`.
4. Start the dev server.
5. Open the app.

Expected result:

- `npm run test` passes.
- `npm run lint` passes, or only documented pre-existing warnings remain.
- `npm run build` passes.
- App loads without console errors caused by the test harness.
- CI includes the test command before build.

## Cross-Lab Edge Cases

| Area | Edge Case | Expected Behavior |
| --- | --- | --- |
| Numeric inputs | Empty number input | Do not produce `NaN` results |
| Numeric inputs | Negative or zero values | Clamp or show validation warning |
| Quantization | Unknown quantization | TypeScript should prevent it |
| GPU memory | `gpuMemoryUtilization <= 0` or `> 1` | Show validation diagnostic |
| Tensor parallelism | TP greater than GPU count | Show critical diagnostic |
| Tensor parallelism | GPU count not divisible by TP | Show warning or block invalid config |
| Context length | Very large context | KV cache should dominate and diagnostics should explain why |
| Concurrency | High concurrency | Recommend replicas if model fits but KV capacity does not |
| Rounding | Breakdown total | Displayed parts should add to displayed total within tolerance |

## Recommended Verification Commands

Run these after every lab implementation:

```bash
npm run lint
npm run build
npm run dev -- --host 127.0.0.1
```

After L10 is implemented, also run:

```bash
npm run test
```

Then complete the matching runtime test from the Runtime Test Matrix.

For Docker verification:

```bash
docker compose up -d --build
```

Then open the configured port from `.env` or the default compose mapping.

## Definition Of Done For Each Lab

A lab is Done only when:

- The implementation is complete.
- Relevant status in this document is updated.
- Unit tests are added for changed calculation behavior.
- Runtime test cases are added or updated in this document.
- The matching runtime test is executed and recorded in PR notes.
- Existing lint and build commands pass.
- `npm run test` passes after L10 is implemented.
- The lab guide or README section explains how to run or verify it.
- Edge cases introduced by the lab are handled or explicitly documented.

## Initial Implementation Recommendation

Start with this order:

1. L10 Test Harness
2. L01 Memory Breakdown
3. L02 Real KV Cache Math
4. L03 Model Presets
5. L04 GPU Presets
6. L05 vLLM Fit Mode
7. L07 Diagnostics/OOM Lab
8. L08 Tensor Parallel vs Replicas
9. L06 Serving Command Generator
10. L09 Student Lab Guides

Reasoning:

- Tests should exist before replacing calculation formulas.
- Memory breakdown makes the existing calculator more educational immediately.
- Real KV cache math is the core production GPU hosting lesson.
- Presets make the lab usable by students without requiring model architecture memorization.
- vLLM fit mode and diagnostics are easier once the memory model is reliable.
