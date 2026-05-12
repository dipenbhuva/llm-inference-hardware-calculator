import { useState } from 'react';
import './App.css';
import {
  MemoryMode,
  ModelQuantization,
  KvCacheQuantization,
  InferenceMode,
  ModelArchitecture,
} from './types';
import {
  calculateDiagnostics,
  calculateHardwareRecommendation,
  calculateMemoryBreakdown,
  calculateOnDiskSize,
  calculateScalingPlan,
  calculateServingCapacity,
  buildVllmServeCommand,
} from './calculations';
import { Tooltip } from './components/Tooltip';
import { ThemeToggle } from './components/ThemeToggle';
import { MemoryBreakdownPanel } from './components/MemoryBreakdownPanel';
import { ServingCapacityPanel } from './components/ServingCapacityPanel';
import { DiagnosticsPanel } from './components/DiagnosticsPanel';
import { ScalingPlanPanel } from './components/ScalingPlanPanel';
import { ServingCommandPanel } from './components/ServingCommandPanel';
import {
  CUSTOM_MODEL_PRESET_ID,
  getModelPresetById,
  modelPresets,
} from './data/modelPresets';
import {
  CUSTOM_GPU_PRESET_ID,
  getGpuPresetById,
  gpuPresets,
} from './data/gpuPresets';

function App() {
  // -----------------------------------
  // 1. STATE
  // -----------------------------------

  // Model config
  const [modelPresetId, setModelPresetId] = useState<string>(
    CUSTOM_MODEL_PRESET_ID
  );
  const [params, setParams] = useState<number>(65); // Billions of parameters
  const [modelQuant, setModelQuant] = useState<ModelQuantization>('Q4');

  // KV Cache
  const [useKvCache, setUseKvCache] = useState<boolean>(false); // Changed from true to false
  const [kvCacheQuant, setKvCacheQuant] = useState<KvCacheQuantization>('Q4'); // Changed from 'F16' to 'Q4'

  // Inference mode
  const [inferenceMode, setInferenceMode] =
    useState<InferenceMode>('incremental');

  // Misc
  const [contextLength, setContextLength] = useState<number>(4096);
  const [layers, setLayers] = useState<number>(80);
  const [hiddenSize, setHiddenSize] = useState<number>(8192);
  const [attentionHeads, setAttentionHeads] = useState<number>(64);
  const [kvHeads, setKvHeads] = useState<number>(8);
  const [headDim, setHeadDim] = useState<number>(128);
  const [concurrentRequests, setConcurrentRequests] = useState<number>(1);
  const [memoryMode, setMemoryMode] = useState<MemoryMode>('DISCRETE_GPU');
  const [systemMemory, setSystemMemory] = useState<number>(128); // in GB
  const [gpuPresetId, setGpuPresetId] = useState<string>('rtx-4090-24gb');
  const [gpuVram, setGpuVram] = useState<number>(24); // in GB, default 24GB
  const [gpuCount, setGpuCount] = useState<number>(1);
  const [gpuMemoryUtilization, setGpuMemoryUtilization] =
    useState<number>(0.9);
  const [maxModelLen, setMaxModelLen] = useState<number>(4096);
  const [tensorParallelSize, setTensorParallelSize] = useState<number>(1);

  // -----------------------------------
  // 2. HELPER FUNCTIONS
  // -----------------------------------

  const handleInputChange = (
    event: React.ChangeEvent<HTMLInputElement>,
    setter: React.Dispatch<React.SetStateAction<number>>
  ) => {
    const newValue = Number(event.target.value);
    if (!isNaN(newValue)) {
      setter(newValue);
    }
  };

  const handleModelPresetChange = (
    event: React.ChangeEvent<HTMLSelectElement>
  ) => {
    const presetId = event.target.value;
    setModelPresetId(presetId);

    const preset = getModelPresetById(presetId);
    if (!preset) {
      return;
    }

    setParams(preset.paramsBillion);
    setContextLength(preset.defaultContextLength);
    setMaxModelLen(preset.defaultContextLength);
    setLayers(preset.layers);
    setHiddenSize(preset.hiddenSize);
    setAttentionHeads(preset.attentionHeads);
    setKvHeads(preset.kvHeads);
    setHeadDim(preset.headDim);
  };

  const handleGpuPresetChange = (
    event: React.ChangeEvent<HTMLSelectElement>
  ) => {
    const presetId = event.target.value;
    setGpuPresetId(presetId);

    const preset = getGpuPresetById(presetId);
    if (preset) {
      setGpuVram(preset.vramGb);
    }
  };

  // -----------------------------------
  // 3. CALCULATE & RENDER
  // -----------------------------------
  const modelArchitecture: ModelArchitecture = {
    layers,
    hiddenSize,
    attentionHeads,
    kvHeads,
    headDim,
  };

  const recommendation = calculateHardwareRecommendation(
    params,
    modelQuant,
    contextLength,
    useKvCache,
    kvCacheQuant,
    memoryMode,
    systemMemory,
    gpuVram,
    inferenceMode,
    modelArchitecture,
    concurrentRequests
  );

  const memoryBreakdown = calculateMemoryBreakdown(
    params,
    modelQuant,
    contextLength,
    useKvCache,
    kvCacheQuant,
    inferenceMode,
    modelArchitecture,
    concurrentRequests
  );

  const servingConfig = {
    gpuCount,
    gpuMemoryUtilization,
    maxModelLen,
    tensorParallelSize,
    targetConcurrentRequests: concurrentRequests,
  };

  const servingCapacity = calculateServingCapacity({
    params,
    modelQuant,
    kvCacheQuant,
    gpuVram,
    architecture: modelArchitecture,
    servingConfig,
  });

  const diagnostics = calculateDiagnostics({
    capacity: servingCapacity,
    servingConfig,
  });

  const scalingPlan = calculateScalingPlan({
    params,
    modelQuant,
    kvCacheQuant,
    gpuVram,
    architecture: modelArchitecture,
    servingConfig,
  });

  const vllmCommand = buildVllmServeCommand({
    tensorParallelSize,
    maxModelLen,
    gpuMemoryUtilization,
    kvCacheQuant,
  });

  const onDiskSize = calculateOnDiskSize(params, modelQuant);

  return (
    <div className="App">
      <ThemeToggle />
      <h1>LLM Inference Hardware Calculator</h1>
      <p className="intro-text">
        Estimate VRAM & System RAM for single-user inference (Batch=1).
        <br />
        Model quant & KV cache quant are configured separately.
      </p>

      <div className="layout">
        {/* Left Panel: Inputs */}
        <div className="input-panel">
          <h2 className="section-title">Model Configuration</h2>

          <label className="label-range">
            Model Preset:
            <Tooltip text="Optional starting point for common model sizes. Presets populate parameter count and architecture fields.">
              i
            </Tooltip>
          </label>
          <select
            aria-label="Model Preset"
            value={modelPresetId}
            onChange={handleModelPresetChange}
          >
            <option value={CUSTOM_MODEL_PRESET_ID}>Custom model</option>
            {modelPresets.map((preset) => (
              <option key={preset.id} value={preset.id}>
                {preset.name}
              </option>
            ))}
          </select>

          <label className="label-range">
            Number of Parameters (Billions):
            <input
              className="text-input-group"
              type="number"
              min={1}
              max={1000}
              value={params}
              onChange={(e) => handleInputChange(e, setParams)}
            />
            <Tooltip text="The total number of model parameters in billions. For example, '13' means a 13B model.">
              i
            </Tooltip>
          </label>
          <div className="slider-input-group">
            <input
              type="range"
              min={1}
              max={1000}
              value={params}
              onChange={(e) => setParams(Number(e.target.value))}
            />
          </div>
          <label className="label-range">
            Model Quantization:
            <Tooltip text="The data format used to store model weights in GPU memory. For instance, F16 uses ~2GB per 1B params, Q4 ~0.5GB, etc.">
              i
            </Tooltip>
          </label>
          <select
            aria-label="Model Quantization"
            value={modelQuant}
            onChange={(e) => setModelQuant(e.target.value as ModelQuantization)}
          >
            {/* F32, F16, Q8, Q6, Q5, Q4, Q3, Q2, GPTQ, AWQ */}
            <option value="F32">F32</option>
            <option value="F16">F16</option>
            <option value="Q8">Q8</option>
            <option value="Q6">Q6</option>
            <option value="Q5">Q5</option>
            <option value="Q4">Q4</option>
            <option value="Q3">Q3</option>
            <option value="Q2">Q2</option>
            <option value="GPTQ">GPTQ</option>
            <option value="AWQ">AWQ</option>
          </select>

          <label className="label-range">
            Context Length (Tokens):
            <input
              className="text-input-group"
              type="number"
              min={128}
              max={32768}
              step={128}
              value={contextLength}
              onChange={(e) => handleInputChange(e, setContextLength)}
            />
            <Tooltip text="Maximum tokens (including prompt and history) available at once. Larger context = more memory usage.">
              i
            </Tooltip>
          </label>
          <div className="slider-input-group">
            <input
              type="range"
              min={128}
              max={32768}
              step={128}
              value={contextLength}
              onChange={(e) => setContextLength(Number(e.target.value))}
            />
          </div>

          <h2 className="section-title section-title-spaced">
            Model Architecture
          </h2>

          <label className="label-range">
            Layers:
            <input
              className="text-input-group"
              type="number"
              min={1}
              max={256}
              value={layers}
              onChange={(e) => handleInputChange(e, setLayers)}
            />
            <Tooltip text="Number of transformer layers. KV cache stores key/value tensors for each layer.">
              i
            </Tooltip>
          </label>

          <label className="label-range">
            Hidden Size:
            <input
              className="text-input-group"
              type="number"
              min={1}
              max={65536}
              value={hiddenSize}
              onChange={(e) => handleInputChange(e, setHiddenSize)}
            />
            <Tooltip text="Model hidden dimension. It is shown for architecture completeness and later preset validation.">
              i
            </Tooltip>
          </label>

          <label className="label-range">
            Attention Heads:
            <input
              className="text-input-group"
              type="number"
              min={1}
              max={512}
              value={attentionHeads}
              onChange={(e) => handleInputChange(e, setAttentionHeads)}
            />
            <Tooltip text="Total attention heads. Some models use fewer KV heads than attention heads with grouped-query attention.">
              i
            </Tooltip>
          </label>

          <label className="label-range">
            KV Heads:
            <input
              className="text-input-group"
              type="number"
              min={1}
              max={512}
              value={kvHeads}
              onChange={(e) => handleInputChange(e, setKvHeads)}
            />
            <Tooltip text="Number of key/value heads. KV cache memory uses this value, not necessarily total attention heads.">
              i
            </Tooltip>
          </label>

          <label className="label-range">
            Head Dimension:
            <input
              className="text-input-group"
              type="number"
              min={1}
              max={1024}
              value={headDim}
              onChange={(e) => handleInputChange(e, setHeadDim)}
            />
            <Tooltip text="Dimension of each attention head. For many models this is hidden size divided by attention heads.">
              i
            </Tooltip>
          </label>

          <label className="label-range">
            Concurrent Requests:
            <input
              className="text-input-group"
              type="number"
              min={1}
              max={512}
              value={concurrentRequests}
              onChange={(e) => handleInputChange(e, setConcurrentRequests)}
            />
            <Tooltip text="Number of simultaneous requests sharing KV cache memory at the selected context length.">
              i
            </Tooltip>
          </label>
          <div className="slider-input-group">
            <input
              type="range"
              min={1}
              max={128}
              value={concurrentRequests}
              onChange={(e) => setConcurrentRequests(Number(e.target.value))}
            />
          </div>

          {/* Inference Mode */}
          <label className="label-range">
            Inference Mode:
            <Tooltip text="'Incremental' is streaming token-by-token generation, 'Bulk' processes the entire context in one pass.">
              i
            </Tooltip>
          </label>
          <select
            aria-label="Inference Mode"
            value={inferenceMode}
            onChange={(e) =>
              setInferenceMode(e.target.value as InferenceMode)
            }
          >
            <option value="incremental">Incremental (streaming)</option>
            <option value="bulk">Bulk (all at once)</option>
          </select>

          {/* KV Cache Toggle */}
          <div className="checkbox-row">
            <input
              type="checkbox"
              checked={useKvCache}
              onChange={() => setUseKvCache(!useKvCache)}
              id="kvCache"
            />
            <label htmlFor="kvCache">
              Enable KV Cache
              <Tooltip text="Reuses key/value attention states to accelerate decoding, at the cost of additional VRAM.">
                i
              </Tooltip>
            </label>
          </div>

          {/* 
             (Animated) KV Cache Quant Section:
             We'll wrap it in a div that transitions "max-height"
             so the UI doesn't jump abruptly.
          */}
          <div className={`kvCacheAnimate ${useKvCache ? 'open' : 'closed'}`}>
            <label className="label-range">
              KV Cache Quantization:
              <Tooltip text="Data format for KV cache memory usage. Lower precision reduces memory but may affect performance/quality.">
                i
              </Tooltip>
            </label>
            <select
              aria-label="KV Cache Quantization"
              value={kvCacheQuant}
              onChange={(e) =>
                setKvCacheQuant(e.target.value as KvCacheQuantization)
              }
            >
              <option value="F32">F32</option>
              <option value="F16">F16</option>
              <option value="FP8">FP8</option>
              <option value="Q8">Q8</option>
              <option value="Q5">Q5</option>
              <option value="Q4">Q4</option>
            </select>
          </div>

          <hr style={{ margin: '1rem 0' }} />

          <h2 className="section-title">System Configuration</h2>

          <label className="label-range">System Type:</label>
          <select
            aria-label="System Type"
            value={memoryMode}
            onChange={(e) => setMemoryMode(e.target.value as MemoryMode)}
          >
            <option value="DISCRETE_GPU">Discrete GPU</option>
            <option value="UNIFIED_MEMORY">
              Unified memory (ex: Apple silicon, AMD Ryzen™ Al Max+ 395)
            </option>
          </select>

          {memoryMode === 'DISCRETE_GPU' && (
            <>
              <label className="label-range">GPU Preset:</label>
              <select
                aria-label="GPU Preset"
                value={gpuPresetId}
                onChange={handleGpuPresetChange}
              >
                {gpuPresets.map((preset) => (
                  <option key={preset.id} value={preset.id}>
                    {preset.name}
                  </option>
                ))}
                <option value={CUSTOM_GPU_PRESET_ID}>Custom GPU</option>
              </select>

              <label className="label-range">
                GPU VRAM (GB):
                <input
                  className="text-input-group"
                  type="number"
                  min={1}
                  max={512}
                  value={gpuVram}
                  disabled={gpuPresetId !== CUSTOM_GPU_PRESET_ID}
                  onChange={(e) => handleInputChange(e, setGpuVram)}
                />
                <Tooltip text="Usable VRAM on each GPU. Presets lock this value; Custom GPU allows manual entry.">
                  i
                </Tooltip>
              </label>
            </>
          )}

          {memoryMode === 'DISCRETE_GPU' && (
            <>
              <h2 className="section-title section-title-spaced">
                Serving Configuration
              </h2>

              <label className="label-range">
                GPU Count:
                <input
                  className="text-input-group"
                  type="number"
                  min={1}
                  max={128}
                  value={gpuCount}
                  onChange={(e) => handleInputChange(e, setGpuCount)}
                />
                <Tooltip text="Total GPUs available for one serving deployment. Tensor parallel size cannot use more GPUs than this.">
                  i
                </Tooltip>
              </label>

              <label className="label-range">
                Tensor Parallel Size:
                <input
                  className="text-input-group"
                  type="number"
                  min={1}
                  max={128}
                  value={tensorParallelSize}
                  onChange={(e) =>
                    handleInputChange(e, setTensorParallelSize)
                  }
                />
                <Tooltip text="Number of GPUs used to split one model replica. Higher values reduce model weight memory per GPU.">
                  i
                </Tooltip>
              </label>

              <label className="label-range">
                GPU Memory Utilization:
                <input
                  className="text-input-group"
                  type="number"
                  min={0.1}
                  max={1}
                  step={0.05}
                  value={gpuMemoryUtilization}
                  onChange={(e) =>
                    handleInputChange(e, setGpuMemoryUtilization)
                  }
                />
                <Tooltip text="Fraction of each GPU's VRAM available to the model executor, similar to vLLM's gpu-memory-utilization setting.">
                  i
                </Tooltip>
              </label>

              <label className="label-range">
                Max Model Length:
                <input
                  className="text-input-group"
                  type="number"
                  min={128}
                  max={131072}
                  step={128}
                  value={maxModelLen}
                  onChange={(e) => handleInputChange(e, setMaxModelLen)}
                />
                <Tooltip text="Maximum tokens per request for serving capacity. Higher values reduce maximum concurrent requests.">
                  i
                </Tooltip>
              </label>
            </>
          )}

          <label className="label-range">
            System Memory (GB):
            <input
              className="text-input-group"
              type="number"
              min={8}
              max={512}
              step={8}
              value={systemMemory}
              onChange={(e) => handleInputChange(e, setSystemMemory)}
            />
          </label>
          <div className="slider-input-group">
            <input
              type="range"
              min={8}
              max={512}
              step={8}
              value={systemMemory}
              onChange={(e) => setSystemMemory(Number(e.target.value))}
            />
          </div>
        </div>

        {/* Right Panel: Results */}
        <div className="results-panel">
          <h2 className="section-title">Hardware Requirements</h2>

          <p>
            <strong>VRAM Needed:</strong>{' '}
            <span className="result-highlight">
              {recommendation.vramNeeded} GB
            </span>
          </p>
          <p>
            <strong>On-Disk Size:</strong>{' '}
            <span className="result-highlight">{onDiskSize.toFixed(2)} GB</span>
          </p>
          <p>
            <strong>GPU Config:</strong> {recommendation.gpuType}
          </p>

          <MemoryBreakdownPanel breakdown={memoryBreakdown} />

          {memoryMode === 'DISCRETE_GPU' && (
            <ServingCapacityPanel capacity={servingCapacity} />
          )}

          {memoryMode === 'DISCRETE_GPU' && (
            <ScalingPlanPanel plan={scalingPlan} />
          )}

          {memoryMode === 'DISCRETE_GPU' && (
            <DiagnosticsPanel diagnostics={diagnostics} />
          )}

          {memoryMode === 'DISCRETE_GPU' && (
            <ServingCommandPanel
              command={vllmCommand}
              usesPlaceholderModel={true}
            />
          )}

          {recommendation.gpusRequired > 1 && (
            <p>
              <strong>Number of GPUs Required:</strong>{' '}
              {recommendation.gpusRequired}
            </p>
          )}
          {recommendation.gpusRequired === 1 && (
            <p>
              <strong>Number of GPUs Required:</strong> 1 (Fits on a single GPU)
            </p>
          )}

          {memoryMode === 'DISCRETE_GPU' && (
            <p>
              <strong>System RAM:</strong>{' '}
              {recommendation.systemRamNeeded.toFixed(1)} GB
            </p>
          )}

          {memoryMode === 'UNIFIED_MEMORY' && recommendation.fitsUnified && (
            <p style={{ color: 'green' }}>✅ Fits in unified memory!</p>
          )}
          {memoryMode === 'UNIFIED_MEMORY' && !recommendation.fitsUnified && (
            <p style={{ color: 'red' }}>
              ⚠️ Exceeds unified memory. Increase system RAM or reduce model
              size.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
