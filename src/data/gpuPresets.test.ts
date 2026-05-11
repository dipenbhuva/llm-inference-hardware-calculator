import { describe, expect, it } from 'vitest';
import {
  CUSTOM_GPU_PRESET_ID,
  getGpuPresetById,
  gpuPresets,
} from './gpuPresets';

describe('gpuPresets', () => {
  it('defines stable unique preset IDs', () => {
    const ids = gpuPresets.map((preset) => preset.id);

    expect(ids).not.toContain(CUSTOM_GPU_PRESET_ID);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('contains positive VRAM values for every preset', () => {
    for (const preset of gpuPresets) {
      expect(preset.vramGb).toBeGreaterThan(0);
      expect(['consumer', 'datacenter']).toContain(preset.class);
    }
  });

  it('looks up presets by ID', () => {
    expect(getGpuPresetById('rtx-4090-24gb')?.vramGb).toBe(24);
    expect(getGpuPresetById('nvidia-l40s-48gb')?.vramGb).toBe(48);
    expect(getGpuPresetById('missing')).toBeUndefined();
  });
});
