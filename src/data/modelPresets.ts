import { type ModelPreset } from '../types';

export const CUSTOM_MODEL_PRESET_ID = 'custom';

export const modelPresets: ModelPreset[] = [
  {
    id: 'generic-7b',
    name: '7B reference template',
    paramsBillion: 7,
    layers: 32,
    hiddenSize: 4096,
    attentionHeads: 32,
    kvHeads: 32,
    headDim: 128,
    defaultContextLength: 4096,
  },
  {
    id: 'generic-8b',
    name: '8B reference template',
    paramsBillion: 8,
    layers: 32,
    hiddenSize: 4096,
    attentionHeads: 32,
    kvHeads: 8,
    headDim: 128,
    defaultContextLength: 8192,
  },
  {
    id: 'generic-13b',
    name: '13B reference template',
    paramsBillion: 13,
    layers: 40,
    hiddenSize: 5120,
    attentionHeads: 40,
    kvHeads: 40,
    headDim: 128,
    defaultContextLength: 4096,
  },
  {
    id: 'generic-34b',
    name: '34B reference template',
    paramsBillion: 34,
    layers: 48,
    hiddenSize: 8192,
    attentionHeads: 64,
    kvHeads: 8,
    headDim: 128,
    defaultContextLength: 8192,
  },
  {
    id: 'generic-70b',
    name: '70B reference template',
    paramsBillion: 70,
    layers: 80,
    hiddenSize: 8192,
    attentionHeads: 64,
    kvHeads: 8,
    headDim: 128,
    defaultContextLength: 8192,
  },
  {
    id: 'generic-120b',
    name: '120B reference template',
    paramsBillion: 120,
    layers: 96,
    hiddenSize: 12288,
    attentionHeads: 96,
    kvHeads: 8,
    headDim: 128,
    defaultContextLength: 8192,
  },
];

export const getModelPresetById = (
  presetId: string
): ModelPreset | undefined => {
  return modelPresets.find((preset) => preset.id === presetId);
};
