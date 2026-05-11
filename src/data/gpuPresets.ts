import { type GpuPreset } from '../types';

export const CUSTOM_GPU_PRESET_ID = 'custom';

export const gpuPresets: GpuPreset[] = [
  {
    id: 'rtx-4090-24gb',
    name: 'RTX 4090 24GB',
    vramGb: 24,
    class: 'consumer',
  },
  {
    id: 'nvidia-l4-24gb',
    name: 'NVIDIA L4 24GB',
    vramGb: 24,
    class: 'datacenter',
  },
  {
    id: 'nvidia-a10g-24gb',
    name: 'NVIDIA A10G 24GB',
    vramGb: 24,
    class: 'datacenter',
  },
  {
    id: 'nvidia-l40s-48gb',
    name: 'NVIDIA L40S 48GB',
    vramGb: 48,
    class: 'datacenter',
  },
  {
    id: 'nvidia-a100-40gb',
    name: 'NVIDIA A100 40GB',
    vramGb: 40,
    class: 'datacenter',
  },
  {
    id: 'nvidia-a100-80gb',
    name: 'NVIDIA A100 80GB',
    vramGb: 80,
    class: 'datacenter',
  },
  {
    id: 'nvidia-h100-80gb',
    name: 'NVIDIA H100 80GB',
    vramGb: 80,
    class: 'datacenter',
  },
];

export const getGpuPresetById = (presetId: string): GpuPreset | undefined => {
  return gpuPresets.find((preset) => preset.id === presetId);
};
