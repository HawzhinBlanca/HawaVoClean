import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

/** Ephemeral in-memory partition without persist: prefix to prevent disk caching of private media. */
export const SESSION_PARTITION = 'hawavoclean-desktop';

export const PREFERENCES_FILENAME = 'preferences.json';

export const IPC = Object.freeze({
  engineEndpoint: 'hawa:engine:endpoint',
  pickAudioFiles: 'hawa:files:pick-audio',
  registerDroppedFile: 'hawa:files:register-dropped',
  chooseExportPath: 'hawa:files:choose-export',
  reveal: 'hawa:files:reveal',
  appInfo: 'hawa:app:info',
  getPreferences: 'hawa:app:get-preferences',
  setPreferences: 'hawa:app:set-preferences',
  clearLocalData: 'hawa:session:clear-local-data',
} as const);

export const IPC_EVENTS = Object.freeze({
  openFile: 'hawa:files:opened',
} as const);

export const AUDIO_EXTENSIONS = Object.freeze([
  'wav',
  'wave',
  'aif',
  'aiff',
  'aifc',
  'flac',
  'mp3',
  'm4a',
  'mp4',
  'mov',
  'aac',
  'ogg',
  'opus',
  'webm',
  'mxf',
] as const);

const AUDIO_EXTENSION_SET: ReadonlySet<string> = new Set(AUDIO_EXTENSIONS);

export const MASTER_EXPORT_FILTERS = Object.freeze([
  Object.freeze({ name: 'WAV audio', extensions: Object.freeze(['wav']) }),
  Object.freeze({ name: 'FLAC audio', extensions: Object.freeze(['flac']) }),
  Object.freeze({ name: 'AAC audio', extensions: Object.freeze(['aac', 'm4a']) }),
  Object.freeze({ name: 'MP4 media', extensions: Object.freeze(['mp4']) }),
] as const);

export const RECORD_BUNDLE_EXPORT_FILTERS = Object.freeze([
  Object.freeze({ name: 'ZIP archive', extensions: Object.freeze(['zip']) }),
] as const);

/** Renderer-visible endpoint. Authentication never crosses preload IPC. */
export type EngineEndpoint = Readonly<{ baseUrl: string }>;
export type ExportKind = 'master' | 'record_bundle';
export type ExportRequest = Readonly<{
  kind: ExportKind;
  suggestedName?: string;
  sourcePath?: string;
  sourceId?: string;
}>;

export type AppInfo = Readonly<{
  name: string;
  version: string;
  locale: string;
  platform: NodeJS.Platform;
  arch: string;
  packaged: boolean;
}>;

export type ClearLocalDataResult = Readonly<{
  ok: boolean;
  clearedItems: readonly string[];
  retainedItems: readonly string[];
}>;

export type WorkspaceModePreference = 'simple' | 'pro';

export type WindowBounds = Readonly<{
  x?: number;
  y?: number;
  width: number;
  height: number;
}>;

export type UserPreferences = Readonly<{
  workspaceMode?: WorkspaceModePreference;
  locale?: string;
  livePreviewEnabled?: boolean;
  windowBounds?: WindowBounds;
}>;

const LOCALE_PATTERN = /^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$/;
const SOURCE_ID_PATTERN = /^[0-9a-f]{32}$/;

export function parseWindowBounds(value: unknown): WindowBounds {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new TypeError('windowBounds must be an object.');
  }
  const raw = value as Record<string, unknown>;
  if (typeof raw.width !== 'number' || !Number.isFinite(raw.width)) {
    throw new TypeError('windowBounds.width must be a finite number.');
  }
  if (typeof raw.height !== 'number' || !Number.isFinite(raw.height)) {
    throw new TypeError('windowBounds.height must be a finite number.');
  }
  const width = Math.min(7680, Math.max(960, Math.round(raw.width)));
  const height = Math.min(4320, Math.max(640, Math.round(raw.height)));
  const result: { x?: number; y?: number; width: number; height: number } = { width, height };
  if (raw.x !== undefined) {
    if (typeof raw.x !== 'number' || !Number.isFinite(raw.x)) {
      throw new TypeError('windowBounds.x must be a finite number.');
    }
    result.x = Math.min(16384, Math.max(-16384, Math.round(raw.x)));
  }
  if (raw.y !== undefined) {
    if (typeof raw.y !== 'number' || !Number.isFinite(raw.y)) {
      throw new TypeError('windowBounds.y must be a finite number.');
    }
    result.y = Math.min(16384, Math.max(-16384, Math.round(raw.y)));
  }
  return Object.freeze(result);
}

/** Strictly validate non-sensitive desktop UI preferences. */
export function parseUserPreferences(value: unknown): UserPreferences {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new TypeError('Preferences must be a JSON object.');
  }
  const candidate = value as Record<string, unknown>;
  const out: {
    workspaceMode?: WorkspaceModePreference;
    locale?: string;
    livePreviewEnabled?: boolean;
    windowBounds?: WindowBounds;
  } = {};

  if (candidate.workspaceMode !== undefined) {
    if (candidate.workspaceMode !== 'simple' && candidate.workspaceMode !== 'pro') {
      throw new TypeError('workspaceMode must be "simple" or "pro".');
    }
    out.workspaceMode = candidate.workspaceMode;
  }
  if (candidate.locale !== undefined) {
    if (
      typeof candidate.locale !== 'string' ||
      candidate.locale.length > 32 ||
      !LOCALE_PATTERN.test(candidate.locale)
    ) {
      throw new TypeError('locale must be a valid BCP-47 language tag.');
    }
    out.locale = candidate.locale;
  }
  if (candidate.livePreviewEnabled !== undefined) {
    if (typeof candidate.livePreviewEnabled !== 'boolean') {
      throw new TypeError('livePreviewEnabled must be a boolean.');
    }
    out.livePreviewEnabled = candidate.livePreviewEnabled;
  }
  if (candidate.windowBounds !== undefined) {
    out.windowBounds = parseWindowBounds(candidate.windowBounds);
  }
  return Object.freeze(out);
}

export function loadUserPreferences(userDataDir: string): UserPreferences {
  const filePath = path.join(userDataDir, PREFERENCES_FILENAME);
  if (!fs.existsSync(filePath)) return Object.freeze({});
  try {
    const raw = JSON.parse(fs.readFileSync(filePath, 'utf8')) as unknown;
    return parseUserPreferences(raw);
  } catch {
    return Object.freeze({});
  }
}

export function saveUserPreferences(userDataDir: string, patch: unknown): UserPreferences {
  const validatedPatch = parseUserPreferences(patch);
  const current = loadUserPreferences(userDataDir);
  const merged: UserPreferences = Object.freeze({
    ...current,
    ...validatedPatch,
  });
  fs.mkdirSync(userDataDir, { recursive: true, mode: 0o700 });
  const targetFile = path.join(userDataDir, PREFERENCES_FILENAME);
  const tmpFile = path.join(
    userDataDir,
    `.${PREFERENCES_FILENAME}.tmp.${process.pid}.${crypto.randomBytes(6).toString('hex')}`,
  );
  try {
    fs.writeFileSync(tmpFile, JSON.stringify(merged, null, 2) + '\n', { mode: 0o600 });
    try {
      fs.chmodSync(tmpFile, 0o600);
    } catch {
      // Best-effort on platforms without POSIX mode bits.
    }
    fs.renameSync(tmpFile, targetFile);
  } catch (error) {
    try {
      if (fs.existsSync(tmpFile)) fs.unlinkSync(tmpFile);
    } catch {
      // Best-effort cleanup of temp file.
    }
    throw error;
  }
  return merged;
}

export async function safeClearSessionStorage(
  userDataDir: string,
  sessionClearers?: {
    clearCache?: () => Promise<void>;
    clearStorageData?: () => Promise<void>;
    clearCodeCaches?: () => Promise<void>;
  },
): Promise<ClearLocalDataResult> {
  const clearedItems: string[] = [];
  if (sessionClearers?.clearCache) {
    await sessionClearers.clearCache();
    clearedItems.push('session_http_cache');
  }
  if (sessionClearers?.clearStorageData) {
    await sessionClearers.clearStorageData();
    clearedItems.push('session_storage_data');
  }
  if (sessionClearers?.clearCodeCaches) {
    await sessionClearers.clearCodeCaches();
    clearedItems.push('session_code_cache');
  }
  const partitionsDir = path.join(userDataDir, 'Partitions');
  if (fs.existsSync(partitionsDir)) {
    try {
      const entries = fs.readdirSync(partitionsDir);
      for (const entry of entries) {
        const target = path.join(partitionsDir, entry);
        fs.rmSync(target, { recursive: true, force: true });
        clearedItems.push(`legacy_partition_${entry}`);
      }
    } catch {
      // Best effort removal of legacy partition directory
    }
  }
  return Object.freeze({
    ok: true,
    clearedItems: Object.freeze(clearedItems),
    retainedItems: Object.freeze([
      'exported_wav_masters',
      'exported_processing_records',
      'user_source_media',
      'engine_jobs_database',
    ]),
  });
}

export function parseExportRequest(value: unknown): ExportRequest {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new TypeError('Export request must be an object.');
  }
  const candidate = value as Record<string, unknown>;
  if (candidate.kind !== 'master' && candidate.kind !== 'record_bundle') {
    throw new TypeError('Export kind must be master or record_bundle.');
  }
  if (candidate.suggestedName !== undefined && typeof candidate.suggestedName !== 'string') {
    throw new TypeError('Suggested export name must be a string.');
  }
  const suggestedName = candidate.suggestedName?.trim();
  if (suggestedName && suggestedName.length > 240) {
    throw new TypeError('Suggested export name is too long.');
  }
  let sourcePath: string | undefined;
  if (candidate.sourcePath !== undefined) {
    sourcePath = requireAbsolutePath(candidate.sourcePath, 'chooseExportPath source');
  }
  let sourceId: string | undefined;
  if (candidate.sourceId !== undefined) {
    if (typeof candidate.sourceId !== 'string' || !SOURCE_ID_PATTERN.test(candidate.sourceId)) {
      throw new TypeError('chooseExportPath sourceId must be a 32-character lowercase hex identifier.');
    }
    sourceId = candidate.sourceId;
  }
  return {
    kind: candidate.kind,
    ...(suggestedName ? { suggestedName } : {}),
    ...(sourcePath ? { sourcePath } : {}),
    ...(sourceId ? { sourceId } : {}),
  };
}

export function safeSuggestedExportName(request: ExportRequest): string {
  const extension = request.kind === 'master' ? '.wav' : '.zip';
  const fallback = request.kind === 'master' ? 'HawaVoClean Master.wav' : 'HawaVoClean Processing Record.zip';
  if (!request.suggestedName) return fallback;
  // Treat both separators as separators on both hosts. A name prepared on macOS
  // can later be handed to Windows, and backslash is an ordinary character to
  // path.basename() on POSIX.
  const portablePath = request.suggestedName.replace(/\\/g, '/');
  const base = path.posix
    .basename(portablePath)
    .replace(/[\u0000-\u001f\u007f<>:"/\\|?*]/g, '-')
    .replace(/[ .]+$/g, '')
    .trim();
  if (!base || base === '.' || base === '..') return fallback;
  let withoutWrongExtension = base.replace(/\.(?:wav|zip)$/i, '');
  if (/^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$/i.test(withoutWrongExtension)) {
    withoutWrongExtension = `HawaVoClean-${withoutWrongExtension}`;
  }
  return `${withoutWrongExtension || 'HawaVoClean'}${extension}`;
}

export function requireAbsolutePath(value: unknown, operation: string): string {
  if (typeof value !== 'string' || value.length === 0 || value.length > 32_768 || !path.isAbsolute(value)) {
    throw new TypeError(`${operation}: path must be an absolute path.`);
  }
  if (value.includes('\u0000')) throw new TypeError(`${operation}: path contains a NUL byte.`);
  return path.normalize(value);
}

/** Refuse a native path outside the closed production decode contract. */
export function requireSupportedMediaPath(value: unknown, operation: string): string {
  const filePath = requireAbsolutePath(value, operation);
  const extension = path.extname(filePath).slice(1).toLowerCase();
  if (!AUDIO_EXTENSION_SET.has(extension)) {
    throw new TypeError(`${operation}: path must use a supported audio or video extension.`);
  }
  return filePath;
}

/** Extract supported absolute media file paths from OS open-with / second-instance argv. */
export function extractExternalMediaPaths(argv: readonly string[]): string[] {
  const found: string[] = [];
  const seen = new Set<string>();
  for (const rawArg of argv) {
    if (typeof rawArg !== 'string') continue;
    const trimmed = rawArg.trim();
    if (!trimmed || trimmed.startsWith('-')) continue;
    try {
      const validated = requireSupportedMediaPath(trimmed, 'openWith');
      if (!seen.has(validated)) {
        seen.add(validated);
        found.push(validated);
      }
    } catch {
      // Ignore non-media arguments (executable path, working dir '.', Electron switches).
    }
  }
  return found;
}

/** Scrub bearer tokens, root secrets, and user home directories before writing logs or diagnostics. */
export function redactDiagnosticText(text: string): string {
  let scrubbed = text
    .replace(/Bearer\s+[A-Za-z0-9._~+/=-]{12,}/gi, 'Bearer [REDACTED]')
    .replace(/(X-Hawa-Token\s*[:=]\s*)[A-Za-z0-9._~+/=-]{8,}/gi, '$1[REDACTED]')
    .replace(/("?(?:token|sessionToken|authorization)"?\s*[:=]\s*"?)[A-Za-z0-9._~+/=-]{12,}"?/gi, '$1[REDACTED]')
    .replace(/\/Users\/[^/\s"'`]+/g, '/Users/[REDACTED]')
    .replace(/\/home\/[^/\s"'`]+/g, '/home/[REDACTED]')
    .replace(/[A-Za-z]:\\Users\\[^\\\s"'`]+/g, 'C:\\Users\\[REDACTED]');
  const home = os.homedir();
  if (home && home.length > 1) {
    scrubbed = scrubbed.split(home).join('[HOME]');
  }
  return scrubbed;
}
