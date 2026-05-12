# GPU Hosting Calculator Labs

These labs turn the calculator into a production-oriented GPU sizing tool for hosting LLMs.

Run the app:

```bash
npm install
npm run dev -- --host 127.0.0.1
```

Verify any completed lab:

```bash
npm run test
npm run lint
npm run build
```

Lab sequence:

1. [Memory Breakdown](./lab-01-memory-breakdown.md)
2. [Real KV Cache Math](./lab-02-kv-cache-math.md)
3. [Model Presets](./lab-03-model-presets.md)
4. [GPU Presets](./lab-04-gpu-presets.md)
5. [vLLM Fit Mode](./lab-05-vllm-fit-mode.md)
6. [OOM Diagnostics](./lab-06-oom-diagnostics.md)
7. [Tensor Parallel vs Replicas](./lab-07-tensor-parallel-vs-replicas.md)
8. [Serving Command Generator](./lab-08-serving-command.md)

The PRD and status tracker live in [docs/gpu-hosting-labs-prd.md](../docs/gpu-hosting-labs-prd.md).
