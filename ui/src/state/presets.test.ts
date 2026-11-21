import { describe, expect, it } from 'vitest';
import {
  BUILTIN_PRESETS,
  DEFAULT_NAMING_TEMPLATE,
  DEFAULT_STUDIO_RACK_SETTINGS,
  MAX_PRESET_JSON_BYTES,
  PRESET_SCHEMA_ID,
  PRESET_STORAGE_KEY,
  deleteUserPreset,
  exportFormatExtension,
  exportPresetJson,
  formatBatchOutputName,
  getAllPresets,
  importPresetJson,
  loadUserPresets,
  saveUserPreset,
  validateNamingTemplate,
  validateStudioRackSettings,
  type StudioRackSettings,
} from './presets';

function makeMemoryStorage(): Storage {
  const map = new Map<string, string>();
  return {
    get length() {
      return map.size;
    },
    clear() {
      map.clear();
    },
    getItem(key: string) {
      return map.has(key) ? map.get(key)! : null;
    },
    key(index: number) {
      return Array.from(map.keys())[index] ?? null;
    },
    removeItem(key: string) {
      map.delete(key);
    },
    setItem(key: string, value: string) {
      map.set(key, String(value));
    },
  };
}

function makeThrowingStorage(): Storage {
  const boom = () => {
    throw new Error('SecurityError: storage unavailable');
  };
  return {
    get length() {
      return 0;
    },
    clear: boom,
    getItem: boom,
    key: boom,
    removeItem: boom,
    setItem: boom,
  };
}

describe('presets state & validation', () => {
  it('ships the 4 curated built-in studio presets with valid settings', () => {
    expect(BUILTIN_PRESETS.map((p) => p.name)).toEqual([
      'Broadcast Dialogue (EBU R128)',
      'Podcast Studio (-16 LUFS)',
      'Resolve Raw Stem (Match Input)',
      'Field / Phone Rescue',
    ]);

    for (const preset of BUILTIN_PRESETS) {
      expect(preset.builtin).toBe(true);
      expect(preset.id.startsWith('builtin:')).toBe(true);
      const check = validateStudioRackSettings(preset.settings);
      expect(check.ok).toBe(true);
    }
  });

  it('saves, loads, updates, and deletes user presets with an isolated Storage', () => {
    const storage = makeMemoryStorage();
    expect(loadUserPresets(storage)).toEqual([]);

    const created = saveUserPreset(
      {
        name: '  Studio Boom Mic  ',
        description: 'Warm dialogue chain',
        settings: { ...DEFAULT_STUDIO_RACK_SETTINGS, strength: 0.8, deliveryPreset: 'podcast' },
      },
      storage,
    );

    expect(created.name).toBe('Studio Boom Mic');
    expect(created.builtin).toBe(false);
    expect(created.id.startsWith('user:')).toBe(true);

    const loaded = loadUserPresets(storage);
    expect(loaded).toHaveLength(1);
    expect(loaded[0]?.id).toBe(created.id);
    expect(loaded[0]?.settings.strength).toBe(0.8);

    const updated = saveUserPreset(
      {
        id: created.id,
        name: 'Studio Boom Mic v2',
        settings: { ...created.settings, strength: 0.6 },
      },
      storage,
    );
    expect(updated.id).toBe(created.id);
    expect(updated.createdAt).toBe(created.createdAt);
    expect(updated.updatedAt).toBeDefined();
    expect(loadUserPresets(storage)).toHaveLength(1);
    expect(loadUserPresets(storage)[0]?.name).toBe('Studio Boom Mic v2');

    expect(getAllPresets(storage)).toHaveLength(BUILTIN_PRESETS.length + 1);

    // Built-in presets cannot be deleted
    expect(deleteUserPreset(BUILTIN_PRESETS[0]!.id, storage)).toBe(false);
    // Missing id returns false
    expect(deleteUserPreset('user:missing', storage)).toBe(false);
    // User preset deletes cleanly
    expect(deleteUserPreset(created.id, storage)).toBe(true);
    expect(loadUserPresets(storage)).toEqual([]);
  });

  it('never overwrites builtin ids when saving a copy from a builtin preset', () => {
    const storage = makeMemoryStorage();
    const copy = saveUserPreset(
      {
        id: BUILTIN_PRESETS[0]!.id,
        name: 'My Broadcast Tweaks',
        settings: BUILTIN_PRESETS[0]!.settings,
      },
      storage,
    );
    expect(copy.id.startsWith('user:')).toBe(true);
    expect(copy.builtin).toBe(false);
  });

  it('degrades safely when localStorage is null or throws in SSR/sandboxed environments', () => {
    expect(loadUserPresets(null)).toEqual([]);
    expect(loadUserPresets(makeThrowingStorage())).toEqual([]);
    expect(deleteUserPreset('user:abc', null)).toBe(false);
    expect(deleteUserPreset('user:abc', makeThrowingStorage())).toBe(false);

    const saved = saveUserPreset(
      { name: 'Ephemeral Preset', settings: DEFAULT_STUDIO_RACK_SETTINGS },
      null,
    );
    expect(saved.name).toBe('Ephemeral Preset');
  });

  it('drops corrupt stored records without crashing loadUserPresets', () => {
    const storage = makeMemoryStorage();
    storage.setItem(PRESET_STORAGE_KEY, '{not valid json');
    expect(loadUserPresets(storage)).toEqual([]);

    storage.setItem(
      PRESET_STORAGE_KEY,
      JSON.stringify([
        { id: 'user:bad', name: 'Corrupt', settings: { strength: 99 } },
        {
          id: 'user:good',
          name: 'Valid Preset',
          createdAt: '2026-01-01T00:00:00.000Z',
          settings: DEFAULT_STUDIO_RACK_SETTINGS,
        },
      ]),
    );
    const loaded = loadUserPresets(storage);
    expect(loaded).toHaveLength(1);
    expect(loaded[0]?.id).toBe('user:good');
  });

  it('round-trips presets cleanly through exportPresetJson and importPresetJson', () => {
    const exported = exportPresetJson(BUILTIN_PRESETS[1]!);
    const parsed = JSON.parse(exported) as { schema: string; schemaVersion: number };
    expect(parsed.schema).toBe(PRESET_SCHEMA_ID);
    expect(parsed.schemaVersion).toBe(1);

    const imported = importPresetJson(exported);
    expect(imported.ok).toBe(true);
    if (imported.ok) {
      expect(imported.preset.name).toBe('Podcast Studio (-16 LUFS)');
      expect(imported.preset.builtin).toBe(false);
      expect(imported.preset.id.startsWith('user:')).toBe(true);
      expect(imported.preset.settings).toEqual(BUILTIN_PRESETS[1]!.settings);
    }
  });

  it('rejects corrupt, out-of-range, unknown-field, oversized, and malicious JSON', () => {
    expect(importPresetJson('').ok).toBe(false);
    expect(importPresetJson('{broken').ok).toBe(false);
    expect(importPresetJson('[]').ok).toBe(false);
    expect(importPresetJson('x'.repeat(MAX_PRESET_JSON_BYTES + 1)).ok).toBe(false);

    // Prototype pollution payload
    const protoPayload = JSON.stringify({
      schema: PRESET_SCHEMA_ID,
      schemaVersion: 1,
      preset: {
        name: 'Evil',
        settings: DEFAULT_STUDIO_RACK_SETTINGS,
      },
    }).replace('"name"', '"__proto__":{"polluted":true},"name"');
    const protoRes = importPresetJson(protoPayload);
    expect(protoRes.ok).toBe(false);

    // Wrong schema / version
    expect(
      importPresetJson(
        JSON.stringify({
          schema: 'other.schema.v1',
          schemaVersion: 1,
          preset: { name: 'A', settings: DEFAULT_STUDIO_RACK_SETTINGS },
        }),
      ).ok,
    ).toBe(false);

    // Unknown property in settings
    expect(
      importPresetJson(
        JSON.stringify({
          name: 'Injected',
          settings: { ...DEFAULT_STUDIO_RACK_SETTINGS, extraFlag: true },
        }),
      ).ok,
    ).toBe(false);

    // Out-of-range strength & HPF
    expect(
      validateStudioRackSettings({ ...DEFAULT_STUDIO_RACK_SETTINGS, strength: 0.05 }).ok,
    ).toBe(false);
    expect(
      validateStudioRackSettings({ ...DEFAULT_STUDIO_RACK_SETTINGS, strength: 1.5 }).ok,
    ).toBe(false);
    expect(
      validateStudioRackSettings({ ...DEFAULT_STUDIO_RACK_SETTINGS, hpfCutoffHz: 15 }).ok,
    ).toBe(false);
    expect(
      validateStudioRackSettings({ ...DEFAULT_STUDIO_RACK_SETTINGS, hpfCutoffHz: 200 }).ok,
    ).toBe(false);
    expect(
      validateStudioRackSettings({ ...DEFAULT_STUDIO_RACK_SETTINGS, ambienceFloorDb: 5 }).ok,
    ).toBe(false);

    // Non-boolean coercion rejected
    expect(
      validateStudioRackSettings({
        ...DEFAULT_STUDIO_RACK_SETTINGS,
        deplosive: 'true' as unknown as boolean,
      }).ok,
    ).toBe(false);

    // Invalid enum values rejected
    expect(
      validateStudioRackSettings({
        ...DEFAULT_STUDIO_RACK_SETTINGS,
        deliveryPreset: 'loud' as StudioRackSettings['deliveryPreset'],
      }).ok,
    ).toBe(false);

    // Path traversal in namingTemplate rejected
    expect(validateNamingTemplate('../etc/passwd').ok).toBe(false);
    expect(validateNamingTemplate('sub/dir/{name}').ok).toBe(false);
    expect(validateNamingTemplate(DEFAULT_NAMING_TEMPLATE).ok).toBe(true);
  });

  it('formats batch output filenames safely for each export format', () => {
    expect(exportFormatExtension('wav24')).toBe('wav');
    expect(exportFormatExtension('wav32f')).toBe('wav');
    expect(exportFormatExtension('flac')).toBe('flac');
    expect(exportFormatExtension('aac')).toBe('m4a');
    expect(exportFormatExtension('video_mux')).toBe('mp4');

    expect(
      formatBatchOutputName('{name}_{profile}_{preset}', {
        name: 'interview_01',
        profile: 'studio_pro',
        preset: 'ebu_r128',
        exportFormat: 'wav32f',
      }),
    ).toBe('interview_01_studio_pro_ebu_r128.wav');

    // Falls back safely when template contains path traversal
    expect(
      formatBatchOutputName('../escape/{name}', {
        name: 'clip',
        profile: 'studio',
        preset: 'podcast',
        exportFormat: 'video_mux',
      }),
    ).toBe('clip_studio_podcast.mp4');
  });
});
