import { describe, expect, it } from 'vitest';
import {
  CUSTOM_MODEL_PRESET_ID,
  getModelPresetById,
  modelPresets,
} from './modelPresets';

describe('modelPresets', () => {
  it('defines stable unique preset IDs', () => {
    const ids = modelPresets.map((preset) => preset.id);

    expect(ids).not.toContain(CUSTOM_MODEL_PRESET_ID);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('contains valid positive architecture values for every preset', () => {
    for (const preset of modelPresets) {
      expect(preset.paramsBillion).toBeGreaterThan(0);
      expect(preset.layers).toBeGreaterThan(0);
      expect(preset.hiddenSize).toBeGreaterThan(0);
      expect(preset.attentionHeads).toBeGreaterThan(0);
      expect(preset.kvHeads).toBeGreaterThan(0);
      expect(preset.headDim).toBeGreaterThan(0);
      expect(preset.defaultContextLength).toBeGreaterThan(0);
    }
  });

  it('looks up presets by ID', () => {
    expect(getModelPresetById('generic-7b')?.paramsBillion).toBe(7);
    expect(getModelPresetById('generic-70b')?.layers).toBe(80);
    expect(getModelPresetById('missing')).toBeUndefined();
  });
});
