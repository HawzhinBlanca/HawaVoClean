import fs from 'node:fs';
import path from 'node:path';

/** Ephemeral in-memory partition without persist: prefix to prevent disk caching of private media. */
export const SESSION_PARTITION = 'hawavoclean-desktop';

export const IPC = Object.freeze({
  engineEndpoint: 'hawa:engine:endpoint',
  pickAudioFiles: 'hawa:files:pick-audio',
  registerDroppedFile: 'hawa:files:register-dropped',
  pickFolder: 'hawa:files:pick-folder',
  chooseExportPath: 'hawa:files:choose-export',
  reveal: 'hawa:files:reveal',
  appInfo: 'hawa:app:info',
  diagnosticsState: 'hawa:diagnostics:state',
  updateState: 'hawa:updates:state',
  clearLocalData: 'hawa:session:clear-local-data',
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
] as const);

const AUDIO_EXTENSION_SET: ReadonlySet<string> = new Set(AUDIO_EXTENSIONS);

/** Renderer-visible endpoint. Authentication never crosses preload IPC. */
export type EngineEndpoint = Readonly<{ baseUrl: string }>;
export type ExportKind = 'master' | 'record_bundle';
export type ExportRequest = Readonly<{ kind: ExportKind; suggestedName?: string }>;

export type AppInfo = Readonly<{
  name: string;
  version: string;
  locale: string;
  platform: NodeJS.Platform;
  arch: string;
  packaged: boolean;
}>;

export type DiagnosticsState = Readonly<{
  status: 'unavailable';
  reason: 'not_implemented';
}>;

export type UpdateState = Readonly<{
  status: 'disabled';
  reason: 'release_feed_not_configured';
  canCheck: false;
}>;

export type ClearLocalDataResult = Readonly<{
  ok: boolean;
  clearedItems: readonly string[];
  retainedItems: readonly string[];
}>;

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
  return suggestedName ? { kind: candidate.kind, suggestedName } : { kind: candidate.kind };
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
