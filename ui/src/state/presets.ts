// Pro Studio Parameter Rack settings, curated built-in presets, and portable
// user preset storage with strict runtime schema validation.

export type DeliveryPresetId =
  | 'match_input'
  | 'ebu_r128'
  | 'atsc_a85'
  | 'podcast'
  | 'streaming';

export type ExportFormatId = 'wav24' | 'wav32f' | 'flac' | 'aac' | 'video_mux';

export type StereoModeId =
  | 'auto'
  | 'dual_mono_same'
  | 'split_speakers'
  | 'true_stereo_ms';

export interface StudioRackSettings {
  /** Cleaning intensity in [0.1, 1.0]. */
  strength: number;
  /** Preserved ambience floor in dB (e.g. -55, -45, -38), or null for off. */
  ambienceFloorDb: number | null;
  deplosive: boolean;
  mouthDeclick: boolean;
  declip: boolean;
  deess: boolean;
  tonalRestoration: boolean;
  levelRider: boolean;
  /** High-pass filter cutoff frequency in Hz, bounded to [20, 120]. */
  hpfCutoffHz: number;
  /** Optional explicit HPF bypass flag; omitted/true means HPF is active. */
  hpfEnabled?: boolean | undefined;
  deliveryPreset: DeliveryPresetId;
  exportFormat: ExportFormatId;
  stereoMode: StereoModeId;
  /** Optional batch output naming template (e.g. `{name}_{profile}_{preset}`). */
  namingTemplate?: string | undefined;
}

export interface UserPreset {
  id: string;
  name: string;
  description?: string | undefined;
  builtin?: boolean | undefined;
  createdAt: string;
  updatedAt?: string | undefined;
  settings: StudioRackSettings;
}

export type PresetImportResult =
  | { ok: true; preset: UserPreset }
  | { ok: false; error: string };

export interface SaveUserPresetInput {
  id?: string | undefined;
  name: string;
  description?: string | undefined;
  settings: StudioRackSettings;
}

export const PRESET_STORAGE_KEY = 'hawavoclean.studioPresets.v1';
export const PRESET_SCHEMA_ID = 'hawavoclean.preset.v1';
export const PRESET_SCHEMA_VERSION = 1;
export const DEFAULT_NAMING_TEMPLATE = '{name}_{profile}_{preset}';
export const MAX_PRESET_JSON_BYTES = 32_768;
export const MAX_PRESET_NAME_LENGTH = 80;
export const MAX_PRESET_DESCRIPTION_LENGTH = 240;
export const MAX_NAMING_TEMPLATE_LENGTH = 120;
export const MAX_USER_PRESETS = 100;

export const STRENGTH_MIN = 0.1;
export const STRENGTH_MAX = 1.0;
export const HPF_MIN_HZ = 20;
export const HPF_MAX_HZ = 120;
export const AMBIENCE_FLOOR_MIN_DB = -80;
export const AMBIENCE_FLOOR_MAX_DB = -12;

export const AMBIENCE_FLOOR_OPTIONS: readonly (number | null)[] = [null, -55, -45, -38];

const DELIVERY_PRESET_SET: ReadonlySet<string> = new Set<DeliveryPresetId>([
  'match_input',
  'ebu_r128',
  'atsc_a85',
  'podcast',
  'streaming',
]);

const EXPORT_FORMAT_SET: ReadonlySet<string> = new Set<ExportFormatId>([
  'wav24',
  'wav32f',
  'flac',
  'aac',
  'video_mux',
]);

const STEREO_MODE_SET: ReadonlySet<string> = new Set<StereoModeId>([
  'auto',
  'dual_mono_same',
  'split_speakers',
  'true_stereo_ms',
]);

const ALLOWED_SETTINGS_KEYS: ReadonlySet<string> = new Set([
  'strength',
  'ambienceFloorDb',
  'deplosive',
  'mouthDeclick',
  'declip',
  'deess',
  'tonalRestoration',
  'levelRider',
  'hpfCutoffHz',
  'hpfEnabled',
  'deliveryPreset',
  'exportFormat',
  'stereoMode',
  'namingTemplate',
]);

const ALLOWED_PRESET_KEYS: ReadonlySet<string> = new Set([
  'id',
  'name',
  'description',
  'builtin',
  'createdAt',
  'updatedAt',
  'settings',
]);

const ALLOWED_ENVELOPE_KEYS: ReadonlySet<string> = new Set([
  'schema',
  'schemaVersion',
  'preset',
]);

const FORBIDDEN_KEYS: ReadonlySet<string> = new Set([
  '__proto__',
  'constructor',
  'prototype',
]);

// eslint-disable-next-line no-control-regex
const CONTROL_CHAR_RE = /[\u0000-\u001f\u007f]/;
const UNSAFE_PATH_RE = /(?:^|[/\\])\.\.(?:[/\\]|$)|[/\\]/;

export const DEFAULT_STUDIO_RACK_SETTINGS: StudioRackSettings = Object.freeze({
  strength: 0.75,
  ambienceFloorDb: -45,
  deplosive: true,
  mouthDeclick: true,
  declip: true,
  deess: true,
  tonalRestoration: false,
  levelRider: true,
  hpfCutoffHz: 80,
  hpfEnabled: true,
  deliveryPreset: 'ebu_r128',
  exportFormat: 'wav24',
  stereoMode: 'auto',
  namingTemplate: DEFAULT_NAMING_TEMPLATE,
});

export const BUILTIN_PRESETS: readonly UserPreset[] = Object.freeze([
  Object.freeze({
    id: 'builtin:broadcast-dialogue-ebu-r128',
    name: 'Broadcast Dialogue (EBU R128)',
    description: 'Clean, natural broadcast voice calibrated to -23 LUFS with room-tone floor.',
    builtin: true,
    createdAt: '2026-01-01T00:00:00.000Z',
    settings: Object.freeze({
      strength: 0.75,
      ambienceFloorDb: -45,
      deplosive: true,
      mouthDeclick: true,
      declip: true,
      deess: true,
      tonalRestoration: false,
      levelRider: true,
      hpfCutoffHz: 80,
      hpfEnabled: true,
      deliveryPreset: 'ebu_r128',
      exportFormat: 'wav24',
      stereoMode: 'auto',
      namingTemplate: DEFAULT_NAMING_TEMPLATE,
    }),
  }),
  Object.freeze({
    id: 'builtin:podcast-studio-lufs16',
    name: 'Podcast Studio (-16 LUFS)',
    description: 'Polished close-mic vocal chain with de-click, tonal balance, and -16 LUFS target.',
    builtin: true,
    createdAt: '2026-01-01T00:00:00.000Z',
    settings: Object.freeze({
      strength: 0.85,
      ambienceFloorDb: -55,
      deplosive: true,
      mouthDeclick: true,
      declip: true,
      deess: true,
      tonalRestoration: true,
      levelRider: true,
      hpfCutoffHz: 75,
      hpfEnabled: true,
      deliveryPreset: 'podcast',
      exportFormat: 'wav24',
      stereoMode: 'auto',
      namingTemplate: DEFAULT_NAMING_TEMPLATE,
    }),
  }),
  Object.freeze({
    id: 'builtin:resolve-raw-stem-match-input',
    name: 'Resolve Raw Stem (Match Input)',
    description: 'Transparent 32-bit float stem preserving input loudness and M/S stereo image.',
    builtin: true,
    createdAt: '2026-01-01T00:00:00.000Z',
    settings: Object.freeze({
      strength: 0.65,
      ambienceFloorDb: -38,
      deplosive: false,
      mouthDeclick: false,
      declip: false,
      deess: false,
      tonalRestoration: false,
      levelRider: false,
      hpfCutoffHz: 60,
      hpfEnabled: true,
      deliveryPreset: 'match_input',
      exportFormat: 'wav32f',
      stereoMode: 'true_stereo_ms',
      namingTemplate: DEFAULT_NAMING_TEMPLATE,
    }),
  }),
  Object.freeze({
    id: 'builtin:field-phone-rescue',
    name: 'Field / Phone Rescue',
    description: 'High-strength cleanup with full restoration chain for harsh field or phone clips.',
    builtin: true,
    createdAt: '2026-01-01T00:00:00.000Z',
    settings: Object.freeze({
      strength: 0.95,
      ambienceFloorDb: null,
      deplosive: true,
      mouthDeclick: true,
      declip: true,
      deess: true,
      tonalRestoration: true,
      levelRider: true,
      hpfCutoffHz: 100,
      hpfEnabled: true,
      deliveryPreset: 'streaming',
      exportFormat: 'wav24',
      stereoMode: 'dual_mono_same',
      namingTemplate: DEFAULT_NAMING_TEMPLATE,
    }),
  }),
]);

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return false;
  }
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function hasForbiddenKeysDeep(value: unknown, depth = 0): boolean {
  if (depth > 16) return true;
  if (Array.isArray(value)) {
    return value.some((item) => hasForbiddenKeysDeep(item, depth + 1));
  }
  if (typeof value === 'object' && value !== null) {
    for (const key of Object.keys(value)) {
      if (FORBIDDEN_KEYS.has(key)) return true;
      const child = (value as Record<string, unknown>)[key];
      if (hasForbiddenKeysDeep(child, depth + 1)) return true;
    }
  }
  return false;
}

function validateSafeString(
  value: unknown,
  field: string,
  maxLength: number,
  allowEmpty = false,
): { ok: true; value: string } | { ok: false; error: string } {
  if (typeof value !== 'string') {
    return { ok: false, error: `${field} must be a string` };
  }
  if (CONTROL_CHAR_RE.test(value)) {
    return { ok: false, error: `${field} contains invalid control characters` };
  }
  const trimmed = value.trim();
  if (!allowEmpty && trimmed.length === 0) {
    return { ok: false, error: `${field} must not be empty` };
  }
  if (trimmed.length > maxLength) {
    return { ok: false, error: `${field} exceeds maximum length of ${maxLength} characters` };
  }
  return { ok: true, value: trimmed };
}

export function validateNamingTemplate(
  template: unknown,
): { ok: true; value: string } | { ok: false; error: string } {
  const base = validateSafeString(template, 'namingTemplate', MAX_NAMING_TEMPLATE_LENGTH, false);
  if (!base.ok) return base;
  if (UNSAFE_PATH_RE.test(base.value)) {
    return { ok: false, error: 'namingTemplate must not contain path separators or traversal sequences' };
  }
  return base;
}

export function validateStudioRackSettings(
  raw: unknown,
): { ok: true; settings: StudioRackSettings } | { ok: false; error: string } {
  if (!isPlainRecord(raw)) {
    return { ok: false, error: 'settings must be a JSON object' };
  }
  for (const key of Object.keys(raw)) {
    if (FORBIDDEN_KEYS.has(key) || !ALLOWED_SETTINGS_KEYS.has(key)) {
      return { ok: false, error: `Unknown or forbidden settings property: "${key}"` };
    }
  }

  const strength = raw['strength'];
  if (
    typeof strength !== 'number' ||
    !Number.isFinite(strength) ||
    strength < STRENGTH_MIN ||
    strength > STRENGTH_MAX
  ) {
    return { ok: false, error: `strength must be a finite number between ${STRENGTH_MIN} and ${STRENGTH_MAX}` };
  }

  const ambienceFloorDb = raw['ambienceFloorDb'];
  if (ambienceFloorDb !== null) {
    if (
      typeof ambienceFloorDb !== 'number' ||
      !Number.isFinite(ambienceFloorDb) ||
      ambienceFloorDb < AMBIENCE_FLOOR_MIN_DB ||
      ambienceFloorDb > AMBIENCE_FLOOR_MAX_DB
    ) {
      return {
        ok: false,
        error: `ambienceFloorDb must be null or a finite number between ${AMBIENCE_FLOOR_MIN_DB} and ${AMBIENCE_FLOOR_MAX_DB}`,
      };
    }
  }

  const boolFields = [
    'deplosive',
    'mouthDeclick',
    'declip',
    'deess',
    'tonalRestoration',
    'levelRider',
  ] as const;
  for (const field of boolFields) {
    if (typeof raw[field] !== 'boolean') {
      return { ok: false, error: `${field} must be a boolean` };
    }
  }

  const hpfCutoffHz = raw['hpfCutoffHz'];
  if (
    typeof hpfCutoffHz !== 'number' ||
    !Number.isFinite(hpfCutoffHz) ||
    hpfCutoffHz < HPF_MIN_HZ ||
    hpfCutoffHz > HPF_MAX_HZ
  ) {
    return { ok: false, error: `hpfCutoffHz must be a finite number between ${HPF_MIN_HZ} and ${HPF_MAX_HZ}` };
  }

  let hpfEnabled: boolean | undefined;
  if ('hpfEnabled' in raw && raw['hpfEnabled'] !== undefined) {
    if (typeof raw['hpfEnabled'] !== 'boolean') {
      return { ok: false, error: 'hpfEnabled must be a boolean when provided' };
    }
    hpfEnabled = raw['hpfEnabled'];
  }

  const deliveryPreset = raw['deliveryPreset'];
  if (typeof deliveryPreset !== 'string' || !DELIVERY_PRESET_SET.has(deliveryPreset)) {
    return { ok: false, error: 'deliveryPreset must be a valid DeliveryPresetId' };
  }

  const exportFormat = raw['exportFormat'];
  if (typeof exportFormat !== 'string' || !EXPORT_FORMAT_SET.has(exportFormat)) {
    return { ok: false, error: 'exportFormat must be a valid ExportFormatId' };
  }

  const stereoMode = raw['stereoMode'];
  if (typeof stereoMode !== 'string' || !STEREO_MODE_SET.has(stereoMode)) {
    return { ok: false, error: 'stereoMode must be a valid StereoModeId' };
  }

  let namingTemplate: string | undefined;
  if ('namingTemplate' in raw && raw['namingTemplate'] !== undefined) {
    const validatedTpl = validateNamingTemplate(raw['namingTemplate']);
    if (!validatedTpl.ok) return validatedTpl;
    namingTemplate = validatedTpl.value;
  }

  const normalized: StudioRackSettings = {
    strength: Math.round(strength * 1000) / 1000,
    ambienceFloorDb,
    deplosive: raw['deplosive'] as boolean,
    mouthDeclick: raw['mouthDeclick'] as boolean,
    declip: raw['declip'] as boolean,
    deess: raw['deess'] as boolean,
    tonalRestoration: raw['tonalRestoration'] as boolean,
    levelRider: raw['levelRider'] as boolean,
    hpfCutoffHz: Math.round(hpfCutoffHz),
    deliveryPreset: deliveryPreset as DeliveryPresetId,
    exportFormat: exportFormat as ExportFormatId,
    stereoMode: stereoMode as StereoModeId,
    ...(hpfEnabled !== undefined ? { hpfEnabled } : {}),
    ...(namingTemplate !== undefined ? { namingTemplate } : {}),
  };

  return { ok: true, settings: normalized };
}

function validateUserPresetObject(
  raw: unknown,
  options: { requireId: boolean; forceUserId: boolean },
): PresetImportResult {
  if (!isPlainRecord(raw)) {
    return { ok: false, error: 'Preset must be a JSON object' };
  }
  for (const key of Object.keys(raw)) {
    if (FORBIDDEN_KEYS.has(key) || !ALLOWED_PRESET_KEYS.has(key)) {
      return { ok: false, error: `Unknown or forbidden preset property: "${key}"` };
    }
  }

  const nameCheck = validateSafeString(raw['name'], 'name', MAX_PRESET_NAME_LENGTH, false);
  if (!nameCheck.ok) return nameCheck;

  let description: string | undefined;
  if ('description' in raw && raw['description'] !== undefined) {
    const descCheck = validateSafeString(
      raw['description'],
      'description',
      MAX_PRESET_DESCRIPTION_LENGTH,
      true,
    );
    if (!descCheck.ok) return descCheck;
    if (descCheck.value.length > 0) {
      description = descCheck.value;
    }
  }

  const settingsCheck = validateStudioRackSettings(raw['settings']);
  if (!settingsCheck.ok) return settingsCheck;

  let id: string;
  if ('id' in raw && raw['id'] !== undefined) {
    const idCheck = validateSafeString(raw['id'], 'id', 96, false);
    if (!idCheck.ok) return idCheck;
    if (options.forceUserId && idCheck.value.startsWith('builtin:')) {
      id = generateUserPresetId(nameCheck.value);
    } else {
      id = idCheck.value;
    }
  } else if (options.requireId) {
    return { ok: false, error: 'id is required' };
  } else {
    id = generateUserPresetId(nameCheck.value);
  }

  let createdAt: string;
  if ('createdAt' in raw && raw['createdAt'] !== undefined) {
    const createdCheck = validateSafeString(raw['createdAt'], 'createdAt', 40, false);
    if (!createdCheck.ok) return createdCheck;
    if (Number.isNaN(Date.parse(createdCheck.value))) {
      return { ok: false, error: 'createdAt must be a valid ISO timestamp' };
    }
    createdAt = createdCheck.value;
  } else {
    createdAt = new Date().toISOString();
  }

  let updatedAt: string | undefined;
  if ('updatedAt' in raw && raw['updatedAt'] !== undefined) {
    const updatedCheck = validateSafeString(raw['updatedAt'], 'updatedAt', 40, false);
    if (!updatedCheck.ok) return updatedCheck;
    if (Number.isNaN(Date.parse(updatedCheck.value))) {
      return { ok: false, error: 'updatedAt must be a valid ISO timestamp' };
    }
    updatedAt = updatedCheck.value;
  }

  const preset: UserPreset = {
    id,
    name: nameCheck.value,
    builtin: false,
    createdAt,
    settings: settingsCheck.settings,
    ...(description !== undefined ? { description } : {}),
    ...(updatedAt !== undefined ? { updatedAt } : {}),
  };

  return { ok: true, preset };
}

function generateUserPresetId(name: string): string {
  const slug =
    name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 32) || 'preset';
  const rand = Math.random().toString(36).slice(2, 8);
  return `user:${slug}-${Date.now().toString(36)}-${rand}`;
}

function resolveStorage(explicit?: Storage | null): Storage | null {
  if (explicit !== undefined) return explicit;
  try {
    if (typeof globalThis === 'undefined' || !('localStorage' in globalThis)) {
      return null;
    }
    const ls = globalThis.localStorage;
    if (
      ls &&
      typeof ls.getItem === 'function' &&
      typeof ls.setItem === 'function' &&
      typeof ls.removeItem === 'function'
    ) {
      return ls;
    }
    return null;
  } catch {
    return null;
  }
}

export function loadUserPresets(storage?: Storage | null): UserPreset[] {
  const store = resolveStorage(storage);
  if (!store) return [];

  let rawText: string | null = null;
  try {
    rawText = store.getItem(PRESET_STORAGE_KEY);
  } catch {
    return [];
  }
  if (!rawText) return [];

  let parsed: unknown;
  try {
    parsed = JSON.parse(rawText);
  } catch {
    return [];
  }
  if (!Array.isArray(parsed) || hasForbiddenKeysDeep(parsed)) {
    return [];
  }

  const valid: UserPreset[] = [];
  const seenIds = new Set<string>();
  for (const item of parsed) {
    if (valid.length >= MAX_USER_PRESETS) break;
    const res = validateUserPresetObject(item, { requireId: true, forceUserId: true });
    if (res.ok && !seenIds.has(res.preset.id)) {
      seenIds.add(res.preset.id);
      valid.push(res.preset);
    }
  }
  return valid;
}

function writeUserPresets(presets: readonly UserPreset[], storage?: Storage | null): boolean {
  const store = resolveStorage(storage);
  if (!store) return false;
  try {
    store.setItem(PRESET_STORAGE_KEY, JSON.stringify(presets));
    return true;
  } catch {
    return false;
  }
}

export function getAllPresets(storage?: Storage | null): UserPreset[] {
  return [...BUILTIN_PRESETS, ...loadUserPresets(storage)];
}

export function saveUserPreset(
  input: SaveUserPresetInput,
  storage?: Storage | null,
): UserPreset {
  const nameCheck = validateSafeString(input.name, 'name', MAX_PRESET_NAME_LENGTH, false);
  if (!nameCheck.ok) {
    throw new Error(nameCheck.error);
  }

  let description: string | undefined;
  if (input.description !== undefined) {
    const descCheck = validateSafeString(
      input.description,
      'description',
      MAX_PRESET_DESCRIPTION_LENGTH,
      true,
    );
    if (!descCheck.ok) {
      throw new Error(descCheck.error);
    }
    if (descCheck.value.length > 0) {
      description = descCheck.value;
    }
  }

  const settingsCheck = validateStudioRackSettings(input.settings);
  if (!settingsCheck.ok) {
    throw new Error(settingsCheck.error);
  }

  const now = new Date().toISOString();
  const existing = loadUserPresets(storage);

  let targetId = input.id?.trim();
  if (!targetId || targetId.startsWith('builtin:')) {
    targetId = generateUserPresetId(nameCheck.value);
  }

  const existingIndex = existing.findIndex((p) => p.id === targetId);
  let saved: UserPreset;
  let nextList: UserPreset[];

  if (existingIndex >= 0) {
    const prev = existing[existingIndex]!;
    saved = {
      id: prev.id,
      name: nameCheck.value,
      builtin: false,
      createdAt: prev.createdAt,
      updatedAt: now,
      settings: settingsCheck.settings,
      ...(description !== undefined ? { description } : {}),
    };
    nextList = [...existing];
    nextList[existingIndex] = saved;
  } else {
    saved = {
      id: targetId,
      name: nameCheck.value,
      builtin: false,
      createdAt: now,
      settings: settingsCheck.settings,
      ...(description !== undefined ? { description } : {}),
    };
    nextList = [...existing, saved].slice(-MAX_USER_PRESETS);
  }

  writeUserPresets(nextList, storage);
  return saved;
}

export function deleteUserPreset(id: string, storage?: Storage | null): boolean {
  if (!id || id.startsWith('builtin:')) return false;
  const existing = loadUserPresets(storage);
  const nextList = existing.filter((p) => p.id !== id);
  if (nextList.length === existing.length) return false;
  writeUserPresets(nextList, storage);
  return true;
}

export function exportPresetJson(preset: UserPreset): string {
  const settingsCheck = validateStudioRackSettings(preset.settings);
  if (!settingsCheck.ok) {
    throw new Error(`Cannot export invalid preset settings: ${settingsCheck.error}`);
  }
  const nameCheck = validateSafeString(preset.name, 'name', MAX_PRESET_NAME_LENGTH, false);
  if (!nameCheck.ok) {
    throw new Error(`Cannot export invalid preset name: ${nameCheck.error}`);
  }

  const envelope = {
    schema: PRESET_SCHEMA_ID,
    schemaVersion: PRESET_SCHEMA_VERSION,
    preset: {
      id: preset.id.startsWith('builtin:') ? generateUserPresetId(nameCheck.value) : preset.id,
      name: nameCheck.value,
      ...(preset.description ? { description: preset.description } : {}),
      createdAt: preset.createdAt,
      ...(preset.updatedAt ? { updatedAt: preset.updatedAt } : {}),
      settings: settingsCheck.settings,
    },
  };
  return JSON.stringify(envelope, null, 2);
}

export function importPresetJson(jsonText: string): PresetImportResult {
  if (typeof jsonText !== 'string') {
    return { ok: false, error: 'Preset JSON input must be a string' };
  }
  const trimmed = jsonText.trim();
  if (trimmed.length === 0) {
    return { ok: false, error: 'Preset JSON is empty' };
  }
  if (trimmed.length > MAX_PRESET_JSON_BYTES) {
    return {
      ok: false,
      error: `Preset JSON exceeds maximum size of ${MAX_PRESET_JSON_BYTES} bytes`,
    };
  }
  // Reject raw prototype-pollution keys before JSON.parse attaches or strips them.
  if (/"(?:__proto__|constructor|prototype)"\s*:/.test(trimmed)) {
    return { ok: false, error: 'Preset JSON contains forbidden prototype keys' };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return { ok: false, error: 'Invalid JSON syntax' };
  }

  if (!isPlainRecord(parsed) || hasForbiddenKeysDeep(parsed)) {
    return { ok: false, error: 'Preset JSON must be a plain object without forbidden keys' };
  }

  if ('preset' in parsed || 'schema' in parsed || 'schemaVersion' in parsed) {
    for (const key of Object.keys(parsed)) {
      if (!ALLOWED_ENVELOPE_KEYS.has(key)) {
        return { ok: false, error: `Unknown envelope property: "${key}"` };
      }
    }
    if (parsed['schema'] !== PRESET_SCHEMA_ID) {
      return {
        ok: false,
        error: `Unsupported preset schema: expected "${PRESET_SCHEMA_ID}"`,
      };
    }
    if (parsed['schemaVersion'] !== PRESET_SCHEMA_VERSION) {
      return {
        ok: false,
        error: `Unsupported preset schemaVersion: expected ${PRESET_SCHEMA_VERSION}`,
      };
    }
    return validateUserPresetObject(parsed['preset'], {
      requireId: false,
      forceUserId: true,
    });
  }

  return validateUserPresetObject(parsed, {
    requireId: false,
    forceUserId: true,
  });
}

export function exportFormatExtension(format: ExportFormatId): string {
  switch (format) {
    case 'wav24':
    case 'wav32f':
      return 'wav';
    case 'flac':
      return 'flac';
    case 'aac':
      return 'm4a';
    case 'video_mux':
      return 'mp4';
  }
}

export function formatBatchOutputName(
  template: string,
  tokens: { name: string; profile: string; preset: string; exportFormat: ExportFormatId },
): string {
  const safeTemplate = validateNamingTemplate(template).ok ? template.trim() : DEFAULT_NAMING_TEMPLATE;
  const sanitizeSegment = (raw: string): string =>
    raw
      .replace(/[/\\:*?"<>|]+/g, '_')
      .replace(/\.\.+/g, '_')
      .trim() || 'untitled';

  const stem = safeTemplate
    .replace(/\{name\}/g, sanitizeSegment(tokens.name))
    .replace(/\{profile\}/g, sanitizeSegment(tokens.profile))
    .replace(/\{preset\}/g, sanitizeSegment(tokens.preset));

  return `${stem}.${exportFormatExtension(tokens.exportFormat)}`;
}
