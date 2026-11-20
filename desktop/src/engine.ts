import { spawn, type ChildProcess, type SpawnOptions } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { performance } from 'node:perf_hooks';

import { type EngineEndpoint, redactDiagnosticText } from './contracts.js';

const READY_TIMEOUT_MS = 60_000;
const SHUTDOWN_HTTP_TIMEOUT_MS = 1_000;
const SHUTDOWN_GRACE_MS = 3_000;
const SESSION_HTTP_TIMEOUT_MS = 3_000;
const SESSION_RESPONSE_LIMIT = 8 * 1024;
const NATIVE_SOURCE_HTTP_TIMEOUT_MS = 3_000;
const NATIVE_SOURCE_RESPONSE_LIMIT = 8 * 1024;
const SESSION_MAX_TTL_S = 60 * 60;
const SESSION_RENEW_SKEW_MS = 60_000;
const STDERR_TAIL_LIMIT = 80;
const LINE_BUFFER_LIMIT = 65_536;
const DEFAULT_MAX_RESTART_ATTEMPTS = 3;
const DEFAULT_RESTART_BASE_DELAY_MS = 250;
const DEFAULT_HEALTHY_RESET_MS = 30_000;
const SEMVER_PATTERN = /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/;
const SHA256_LINE_PATTERN = /^([0-9a-fA-F]{64})(?:  | \*)(.+)$/;

export const ENGINE_LOG_MAX_BYTES = 5 * 1024 * 1024;
export const ENGINE_LOG_MAX_ROTATIONS = 3;

export type EngineSpec = Readonly<{
  executable: string;
  prefixArgs: readonly string[];
  cwd: string;
}>;

export type EnginePaths = Readonly<{
  packaged: boolean;
  resourcesPath: string;
  repositoryRoot: string;
  userData: string;
  temp: string;
  platform: NodeJS.Platform;
}>;

export type EngineSessionCapability = Readonly<{
  authorization: string;
  expiresAtMs: number;
  refreshAtMs: number;
}>;

export type NativeSourceRegistration = Readonly<{
  sourceId: string;
  path: string;
}>;

export type EngineIntegrityResult = Readonly<{
  valid: boolean;
  manifestSha256: string | null;
  verifiedFiles?: number;
  error?: string;
}>;

export type EngineBrokerOptions = Readonly<{
  maxRestartAttempts?: number;
  restartBaseDelayMs?: number;
  healthyResetMs?: number;
  logMaxBytes?: number;
  logMaxRotations?: number;
  onPhaseChange?: (phase: EngineBrokerPhase) => void;
}>;

/** Bounded, redacted, rotating log writer for desktop engine diagnostics. */
export class RotatingFileLogger {
  readonly #logPath: string;
  readonly #maxBytes: number;
  readonly #maxRotations: number;

  constructor(
    logPath: string,
    maxBytes: number = ENGINE_LOG_MAX_BYTES,
    maxRotations: number = ENGINE_LOG_MAX_ROTATIONS,
  ) {
    this.#logPath = logPath;
    this.#maxBytes = Math.max(256, Math.floor(maxBytes));
    this.#maxRotations = Math.max(1, Math.floor(maxRotations));
  }

  get logPath(): string {
    return this.#logPath;
  }

  append(rawLine: string): void {
    const trimmed = rawLine.replace(/[\r\n]+/g, ' ').slice(0, 4096).trim();
    if (!trimmed) return;
    const entry = `[${new Date().toISOString()}] ${redactDiagnosticText(trimmed)}\n`;
    const entryBytes = Buffer.byteLength(entry, 'utf8');
    try {
      fs.mkdirSync(path.dirname(this.#logPath), { recursive: true, mode: 0o700 });
      if (fs.existsSync(this.#logPath)) {
        const size = fs.statSync(this.#logPath).size;
        if (size > 0 && size + entryBytes > this.#maxBytes) {
          this.#rotate();
        }
      }
      fs.appendFileSync(this.#logPath, entry, { encoding: 'utf8', mode: 0o600 });
      try {
        fs.chmodSync(this.#logPath, 0o600);
      } catch {
        // Best-effort on platforms without POSIX mode bits.
      }
    } catch {
      // Logging must never crash the desktop host.
    }
  }

  readRecentLines(maxLines: number = 200): string[] {
    if (!fs.existsSync(this.#logPath)) return [];
    try {
      const content = fs.readFileSync(this.#logPath, 'utf8');
      const lines = content.split(/\r?\n/).filter(Boolean);
      return lines.slice(-Math.max(1, maxLines));
    } catch {
      return [];
    }
  }

  #rotate(): void {
    for (let index = this.#maxRotations; index >= 1; index -= 1) {
      const current = `${this.#logPath}.${index}`;
      if (index === this.#maxRotations) {
        if (fs.existsSync(current)) fs.unlinkSync(current);
      } else if (fs.existsSync(current)) {
        fs.renameSync(current, `${this.#logPath}.${index + 1}`);
      }
    }
    if (fs.existsSync(this.#logPath)) {
      fs.renameSync(this.#logPath, `${this.#logPath}.1`);
    }
  }
}

export function engineSessionNeedsRenewal(
  capability: EngineSessionCapability | null,
  nowMs: number,
): boolean {
  return (
    capability === null ||
    !Number.isFinite(nowMs) ||
    nowMs >= capability.refreshAtMs ||
    nowMs >= capability.expiresAtMs
  );
}

export function rendererEngineEndpoint(port: number): EngineEndpoint {
  if (!Number.isInteger(port) || port < 1 || port > 65_535) {
    throw new Error('Engine ready port is invalid.');
  }
  return Object.freeze({ baseUrl: `http://127.0.0.1:${port}` });
}

function exactEngineUrl(endpoint: EngineEndpoint): URL {
  const url = new URL(endpoint.baseUrl);
  if (
    url.protocol !== 'http:' ||
    url.hostname !== '127.0.0.1' ||
    url.username !== '' ||
    url.password !== '' ||
    url.pathname !== '/' ||
    url.search !== '' ||
    url.hash !== '' ||
    !/^\d+$/.test(url.port) ||
    Number(url.port) < 1 ||
    Number(url.port) > 65_535
  ) {
    throw new Error('Engine endpoint is not an exact 127.0.0.1 HTTP origin.');
  }
  return url;
}

function postLoopbackJson(
  endpoint: EngineEndpoint,
  rootToken: string,
  routePath: string,
  label: string,
  timeoutMs: number,
  maxBytes: number,
  refusedStatusMessage: (code: number) => string,
  payload?: unknown,
): Promise<Record<string, unknown>> {
  let url: URL;
  try {
    url = exactEngineUrl(endpoint);
  } catch (error) {
    return Promise.reject(error instanceof Error ? error : new Error('Engine endpoint is invalid.'));
  }
  if (!rootToken || rootToken.length > 256) {
    return Promise.reject(new Error('Engine bootstrap secret is invalid.'));
  }
  const body = payload === undefined ? null : Buffer.from(JSON.stringify(payload), 'utf8');
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (error: Error | null, value?: Record<string, unknown>): void => {
      if (settled) return;
      settled = true;
      if (error) reject(error);
      else if (value) resolve(value);
      else reject(new Error(`${label} produced no result.`));
    };
    const headers: Record<string, string> = {
      'X-Hawa-Token': rootToken,
      Accept: 'application/json',
      'Content-Length': body ? String(body.length) : '0',
    };
    if (body) headers['Content-Type'] = 'application/json';
    const request = http.request(
      { hostname: '127.0.0.1', port: Number(url.port), method: 'POST', path: routePath, headers, timeout: timeoutMs },
      (response) => {
        const chunks: Buffer[] = [];
        let bytes = 0;
        response.on('data', (chunk: Buffer) => {
          bytes += chunk.length;
          if (bytes > maxBytes) {
            response.destroy();
            finish(new Error(`${label} response exceeded its size limit.`));
            return;
          }
          chunks.push(chunk);
        });
        response.once('aborted', () => finish(new Error(`${label} response was interrupted.`)));
        response.once('error', () => finish(new Error(`${label} response failed.`)));
        response.once('end', () => {
          if (response.statusCode !== 200) {
            finish(new Error(refusedStatusMessage(response.statusCode ?? 0)));
            return;
          }
          const cacheControl = String(response.headers['cache-control'] ?? '').toLowerCase();
          if (!cacheControl.includes('no-store')) {
            finish(new Error(`${label} response was not marked no-store.`));
            return;
          }
          let parsed: unknown;
          try {
            parsed = JSON.parse(Buffer.concat(chunks).toString('utf8')) as unknown;
          } catch {
            finish(new Error(`${label} response was not valid JSON.`));
            return;
          }
          if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
            finish(new Error(`${label} response had the wrong shape.`));
            return;
          }
          finish(null, parsed as Record<string, unknown>);
        });
      },
    );
    request.once('timeout', () => {
      request.destroy();
      finish(new Error(`${label} timed out.`));
    });
    request.once('error', () => finish(new Error(`${label} failed.`)));
    request.end(body ?? undefined);
  });
}

/** Mint a bounded, short-lived renderer capability without exposing the root secret. */
export async function requestEngineSession(
  endpoint: EngineEndpoint,
  rootToken: string,
  now: () => number = () => performance.now(),
): Promise<EngineSessionCapability> {
  const record = await postLoopbackJson(
    endpoint,
    rootToken,
    '/api/session',
    'Engine session bootstrap',
    SESSION_HTTP_TIMEOUT_MS,
    SESSION_RESPONSE_LIMIT,
    (code) => `Engine refused session bootstrap (HTTP ${String(code)}).`,
  );
  const token = record.sessionToken;
  const ttlS = record.expiresInSeconds;
  if (
    record.tokenType !== 'Bearer' ||
    typeof token !== 'string' ||
    !/^[A-Za-z0-9_-]{32,256}$/.test(token) ||
    typeof ttlS !== 'number' ||
    !Number.isInteger(ttlS) ||
    ttlS < 1 ||
    ttlS > SESSION_MAX_TTL_S
  ) {
    throw new Error('Engine session response had invalid capability metadata.');
  }
  const issuedAt = now();
  const ttlMs = ttlS * 1000;
  const skewMs = Math.min(SESSION_RENEW_SKEW_MS, Math.max(1_000, ttlMs / 4));
  return {
    authorization: `Bearer ${token}`,
    expiresAtMs: issuedAt + ttlMs,
    refreshAtMs: issuedAt + Math.max(1, ttlMs - skewMs),
  };
}

/** Register one user-selected native file with the root-owned broker channel. */
export async function requestNativeSourceRegistration(
  endpoint: EngineEndpoint,
  rootToken: string,
  filePath: string,
): Promise<NativeSourceRegistration> {
  if (
    typeof filePath !== 'string' ||
    filePath.length < 1 ||
    filePath.length > 32_768 ||
    filePath.includes('\0') ||
    !path.isAbsolute(filePath)
  ) {
    throw new Error('Native source path is invalid.');
  }
  const record = await postLoopbackJson(
    endpoint,
    rootToken,
    '/api/v1/native-sources',
    'Native source registration',
    NATIVE_SOURCE_HTTP_TIMEOUT_MS,
    NATIVE_SOURCE_RESPONSE_LIMIT,
    (code) => `Engine refused the selected file (HTTP ${String(code)}).`,
    { path: filePath },
  );
  if (
    typeof record.sourceId !== 'string' ||
    !/^[0-9a-f]{32}$/.test(record.sourceId) ||
    typeof record.path !== 'string' ||
    record.path.length < 1 ||
    record.path.length > 32_768 ||
    record.path.includes('\0') ||
    !path.isAbsolute(record.path)
  ) {
    throw new Error('Native source registration response had invalid capability metadata.');
  }
  return Object.freeze({ sourceId: record.sourceId, path: record.path });
}

function executableName(platform: NodeJS.Platform): string {
  return platform === 'win32' ? 'hawavoclean-engine.exe' : 'hawavoclean-engine';
}

export function parseDevelopmentCommand(raw: string | undefined): readonly string[] | null {
  if (raw === undefined || raw.trim() === '') return null;
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    throw new Error('HAWAVOCLEAN_DESKTOP_ENGINE_COMMAND must be a JSON array of strings.');
  }
  if (
    !Array.isArray(value) ||
    value.length === 0 ||
    value.length > 32 ||
    !value.every((part) => typeof part === 'string' && part.length > 0 && part.length <= 4096)
  ) {
    throw new Error('HAWAVOCLEAN_DESKTOP_ENGINE_COMMAND must be a non-empty JSON array of bounded strings.');
  }
  return value;
}

function isMatchingManifestPlatform(manifestPlatform: unknown, hostPlatform: NodeJS.Platform): boolean {
  if (typeof manifestPlatform !== 'string') return false;
  if (hostPlatform === 'darwin') {
    return (
      manifestPlatform === 'darwin' ||
      manifestPlatform === 'macos-arm64' ||
      manifestPlatform === 'darwin-arm64'
    );
  }
  if (hostPlatform === 'win32') {
    return (
      manifestPlatform === 'win32' ||
      manifestPlatform === 'win32-x64' ||
      manifestPlatform === 'windows-x64'
    );
  }
  if (hostPlatform === 'linux') {
    return (
      manifestPlatform === 'linux' ||
      manifestPlatform === 'linux-x64' ||
      manifestPlatform === 'linux-arm64'
    );
  }
  return manifestPlatform === hostPlatform;
}

function normalizeSafeChecksumRelativePath(rawRel: string): string | null {
  if (!rawRel || rawRel.includes('\u0000') || rawRel.includes('\\')) return null;
  if (rawRel.startsWith('/') || /^[A-Za-z]:/.test(rawRel) || path.isAbsolute(rawRel)) return null;
  const trimmed = rawRel.startsWith('./') ? rawRel.slice(2) : rawRel;
  if (!trimmed) return null;
  const parts = trimmed.split('/');
  if (parts.some((part) => part === '' || part === '.' || part === '..')) return null;
  return parts.join('/');
}

export function verifyBundledEngineIntegrity(
  engineDir: string,
  platform: NodeJS.Platform,
): EngineIntegrityResult {
  const executable = executableName(platform);
  const launcher = path.join(engineDir, executable);
  const placeholder = path.join(engineDir, 'README.txt');
  if (fs.existsSync(placeholder)) {
    return {
      valid: false,
      manifestSha256: null,
      error: `Archive validation rejected placeholder engine resource: ${placeholder}`,
    };
  }
  if (!fs.existsSync(launcher)) {
    return { valid: false, manifestSha256: null, error: `Engine executable missing: ${launcher}` };
  }
  const manifestPath = path.join(engineDir, 'ENGINE-MANIFEST.json');
  const checksumsPath = path.join(engineDir, 'ENGINE-SHA256SUMS');
  if (!fs.existsSync(manifestPath) || !fs.existsSync(checksumsPath)) {
    return { valid: false, manifestSha256: null, error: 'Engine manifest or checksum inventory missing' };
  }
  try {
    const realEngineDir = fs.realpathSync(engineDir);
    if (!fs.lstatSync(launcher).isFile()) {
      return { valid: false, manifestSha256: null, error: `Engine launcher is not a regular file: ${launcher}` };
    }
    if (!fs.lstatSync(manifestPath).isFile() || !fs.lstatSync(checksumsPath).isFile()) {
      return { valid: false, manifestSha256: null, error: 'Engine manifest or checksums is not a regular file' };
    }

    const manifestBytes = fs.readFileSync(manifestPath);
    const manifestDigest = crypto.createHash('sha256').update(manifestBytes).digest('hex');

    let parsedManifest: unknown;
    try {
      parsedManifest = JSON.parse(manifestBytes.toString('utf8')) as unknown;
    } catch {
      return { valid: false, manifestSha256: null, error: 'ENGINE-MANIFEST.json is not valid JSON' };
    }
    if (typeof parsedManifest !== 'object' || parsedManifest === null || Array.isArray(parsedManifest)) {
      return { valid: false, manifestSha256: null, error: 'ENGINE-MANIFEST.json must be a JSON object' };
    }
    const manifest = parsedManifest as Record<string, unknown>;

    const versionField = typeof manifest.version === 'string' ? manifest.version : null;
    const productVersionField = typeof manifest.product_version === 'string' ? manifest.product_version : null;
    const effectiveVersion = versionField ?? productVersionField;
    if (effectiveVersion === null || !SEMVER_PATTERN.test(effectiveVersion)) {
      return { valid: false, manifestSha256: null, error: 'ENGINE-MANIFEST.json has missing or invalid version' };
    }
    if (versionField !== null && productVersionField !== null && versionField !== productVersionField) {
      return {
        valid: false,
        manifestSha256: null,
        error: 'ENGINE-MANIFEST.json version and product_version disagree',
      };
    }
    if (typeof manifest.launcher !== 'string' || manifest.launcher !== executable) {
      return {
        valid: false,
        manifestSha256: null,
        error: `ENGINE-MANIFEST.json launcher mismatch: expected ${executable}`,
      };
    }
    if (!isMatchingManifestPlatform(manifest.platform, platform)) {
      return {
        valid: false,
        manifestSha256: null,
        error: `ENGINE-MANIFEST.json platform mismatch for host ${platform}`,
      };
    }

    const lines = fs
      .readFileSync(checksumsPath, 'utf8')
      .split(/\r?\n/)
      .filter((line) => line.length > 0);
    if (lines.length === 0) {
      return { valid: false, manifestSha256: null, error: 'ENGINE-SHA256SUMS is empty' };
    }

    const verifiedPaths = new Set<string>();
    for (const line of lines) {
      const match = SHA256_LINE_PATTERN.exec(line);
      if (!match) {
        return { valid: false, manifestSha256: null, error: 'Malformed ENGINE-SHA256SUMS entry' };
      }
      const expectedHash = match[1]!.toLowerCase();
      const rawRel = match[2]!;
      const safeRel = normalizeSafeChecksumRelativePath(rawRel);
      if (safeRel === null) {
        return {
          valid: false,
          manifestSha256: null,
          error: `Unsafe or traversal path in ENGINE-SHA256SUMS: ${rawRel}`,
        };
      }
      if (verifiedPaths.has(safeRel)) {
        return {
          valid: false,
          manifestSha256: null,
          error: `Duplicate entry in ENGINE-SHA256SUMS: ${safeRel}`,
        };
      }
      const candidatePath = path.resolve(realEngineDir, safeRel);
      if (candidatePath !== realEngineDir && !candidatePath.startsWith(realEngineDir + path.sep)) {
        return {
          valid: false,
          manifestSha256: null,
          error: `Path escapes engine root in ENGINE-SHA256SUMS: ${rawRel}`,
        };
      }
      if (!fs.existsSync(candidatePath)) {
        return {
          valid: false,
          manifestSha256: null,
          error: `Checksummed engine file is missing: ${safeRel}`,
        };
      }
      const stats = fs.lstatSync(candidatePath);
      if (!stats.isFile()) {
        return {
          valid: false,
          manifestSha256: null,
          error: `Checksummed engine entry is not a regular file: ${safeRel}`,
        };
      }
      const realTarget = fs.realpathSync(candidatePath);
      if (!realTarget.startsWith(realEngineDir + path.sep)) {
        return {
          valid: false,
          manifestSha256: null,
          error: `Checksummed engine file resolves outside engine root: ${safeRel}`,
        };
      }
      const actualHash = crypto.createHash('sha256').update(fs.readFileSync(candidatePath)).digest('hex');
      if (actualHash !== expectedHash) {
        return {
          valid: false,
          manifestSha256: null,
          error: `Engine checksum mismatch for ${safeRel}`,
        };
      }
      verifiedPaths.add(safeRel);
    }

    if (!verifiedPaths.has(executable)) {
      return {
        valid: false,
        manifestSha256: null,
        error: `ENGINE-SHA256SUMS does not cover launcher ${executable}`,
      };
    }

    return { valid: true, manifestSha256: manifestDigest, verifiedFiles: verifiedPaths.size };
  } catch (err) {
    return { valid: false, manifestSha256: null, error: String(err) };
  }
}

export function resolveEngineSpec(paths: EnginePaths, override: string | undefined): EngineSpec {
  if (paths.packaged) {
    return {
      executable: path.join(paths.resourcesPath, 'engine', executableName(paths.platform)),
      prefixArgs: [],
      cwd: paths.userData,
    };
  }
  const command = parseDevelopmentCommand(override);
  if (command) {
    const executable = command[0];
    if (executable === undefined) throw new Error('Development engine command is empty.');
    return { executable, prefixArgs: command.slice(1), cwd: paths.repositoryRoot };
  }
  return {
    executable: path.join(
      paths.repositoryRoot,
      '.venv',
      paths.platform === 'win32' ? 'Scripts' : 'bin',
      paths.platform === 'win32' ? 'hawavoclean.exe' : 'hawavoclean',
    ),
    prefixArgs: [],
    cwd: paths.repositoryRoot,
  };
}

function waitForExit(child: ChildProcess, timeoutMs: number): Promise<boolean> {
  if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve(true);
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      child.removeListener('exit', onExit);
      resolve(child.exitCode !== null || child.signalCode !== null);
    }, timeoutMs);
    const onExit = (): void => {
      clearTimeout(timer);
      resolve(true);
    };
    child.once('exit', onExit);
  });
}

function postShutdown(endpoint: EngineEndpoint, rootToken: string): Promise<void> {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (): void => {
      if (settled) return;
      settled = true;
      resolve();
    };
    const url = exactEngineUrl(endpoint);
    const request = http.request(
      {
        hostname: '127.0.0.1',
        port: Number(url.port),
        method: 'POST',
        path: '/api/shutdown',
        headers: { 'X-Hawa-Token': rootToken, 'Content-Length': '0' },
        timeout: SHUTDOWN_HTTP_TIMEOUT_MS,
      },
      (response) => {
        response.resume();
        response.once('end', finish);
        response.once('error', finish);
      },
    );
    request.once('timeout', () => {
      request.destroy();
      finish();
    });
    request.once('error', finish);
    request.end();
    setTimeout(finish, SHUTDOWN_HTTP_TIMEOUT_MS + 250).unref();
  });
}

function signalProcessTree(child: ChildProcess, signal: NodeJS.Signals): void {
  if (child.pid === undefined || child.exitCode !== null || child.signalCode !== null) return;
  try {
    if (process.platform === 'win32') {
      child.kill(signal);
    } else {
      process.kill(-child.pid, signal);
    }
  } catch {
    try {
      child.kill(signal);
    } catch {
      // The process may have exited between the checks and signal delivery.
    }
  }
}

function forceKillWindowsTree(child: ChildProcess): Promise<void> {
  if (child.pid === undefined || child.exitCode !== null || child.signalCode !== null) return Promise.resolve();
  return new Promise((resolve) => {
    const killer = spawn('taskkill.exe', ['/pid', String(child.pid), '/t', '/f'], {
      stdio: 'ignore',
      windowsHide: true,
    });
    killer.once('error', () => resolve());
    killer.once('exit', () => resolve());
  });
}

export type EngineBrokerPhase =
  | 'idle'
  | 'starting'
  | 'ready'
  | 'restarting'
  | 'stopping'
  | 'stopped'
  | 'failed';

export const ALLOWED_ENGINE_BROKER_TRANSITIONS: Readonly<
  Record<EngineBrokerPhase, readonly EngineBrokerPhase[]>
> = Object.freeze({
  idle: Object.freeze(['starting', 'stopping', 'stopped'] as const),
  starting: Object.freeze(['ready', 'restarting', 'failed', 'stopping', 'stopped'] as const),
  ready: Object.freeze(['restarting', 'stopping', 'stopped', 'failed'] as const),
  restarting: Object.freeze(['starting', 'stopping', 'stopped', 'failed'] as const),
  stopping: Object.freeze(['stopped'] as const),
  stopped: Object.freeze([] as const),
  failed: Object.freeze(['stopping', 'stopped'] as const),
});

export function resolveRendezvousDir(
  userData: string,
  overrideDir: string | undefined = process.env.HAWAVOCLEAN_RENDEZVOUS_DIR,
): string {
  const trimmed = overrideDir?.trim();
  if (trimmed) return path.resolve(trimmed);
  return path.join(userData, 'rendezvous');
}

export function writeRendezvousRecord(
  rendezvousDir: string,
  record: Readonly<{
    schemaVersion: 1;
    pid: number;
    port: number;
    origin: string;
    token: string;
    createdAt: string;
  }>,
): string {
  fs.mkdirSync(rendezvousDir, { recursive: true, mode: 0o700 });
  const targetFile = path.join(rendezvousDir, 'engine.json');
  const tmpFile = path.join(
    rendezvousDir,
    `.engine.json.tmp.${process.pid}.${crypto.randomBytes(6).toString('hex')}`,
  );
  try {
    fs.writeFileSync(tmpFile, JSON.stringify(record, null, 2) + '\n', { mode: 0o600 });
    try {
      fs.chmodSync(tmpFile, 0o600);
    } catch {
      // Best-effort on platforms without POSIX mode bits.
    }
    fs.renameSync(tmpFile, targetFile);
    return targetFile;
  } catch (err) {
    try {
      if (fs.existsSync(tmpFile)) fs.unlinkSync(tmpFile);
    } catch {
      // Best-effort cleanup of temp file.
    }
    throw err;
  }
}

export function cleanupOwnedRendezvousFile(filePath: string, expectedPid: number): boolean {
  if (!Number.isInteger(expectedPid) || expectedPid <= 0) return false;
  if (!fs.existsSync(filePath)) return false;
  let parsed: unknown;
  try {
    parsed = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return false;
  }
  if (
    typeof parsed !== 'object' ||
    parsed === null ||
    Array.isArray(parsed) ||
    (parsed as Record<string, unknown>).pid !== expectedPid
  ) {
    return false;
  }
  fs.unlinkSync(filePath);
  return true;
}

export class EngineBroker {
  readonly #paths: EnginePaths;
  readonly #token = crypto.randomBytes(32).toString('hex');
  readonly #stderrTail: string[] = [];
  readonly #logger: RotatingFileLogger;
  readonly #maxRestartAttempts: number;
  readonly #restartBaseDelayMs: number;
  readonly #healthyResetMs: number;
  readonly #onPhaseChange: ((phase: EngineBrokerPhase) => void) | undefined;
  #phase: EngineBrokerPhase = 'idle';
  #child: ChildProcess | null = null;
  #endpoint: EngineEndpoint | null = null;
  #session: EngineSessionCapability | null = null;
  #sessionPromise: Promise<EngineSessionCapability> | null = null;
  #failure: Error | null = null;
  #readyPromise: Promise<EngineEndpoint> | null = null;
  #resolveReady: ((endpoint: EngineEndpoint) => void) | null = null;
  #rejectReady: ((error: Error) => void) | null = null;
  #stopPromise: Promise<void> | null = null;
  #rendezvousFile: string | null = null;
  #rendezvousPid: number | null = null;
  #hasBeenReady = false;
  #restartAttempts = 0;
  #restartTimer: ReturnType<typeof setTimeout> | null = null;
  #readyTimer: ReturnType<typeof setTimeout> | null = null;
  #healthyResetTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(paths: EnginePaths, options?: EngineBrokerOptions) {
    this.#paths = paths;
    this.#maxRestartAttempts = options?.maxRestartAttempts ?? DEFAULT_MAX_RESTART_ATTEMPTS;
    this.#restartBaseDelayMs = options?.restartBaseDelayMs ?? DEFAULT_RESTART_BASE_DELAY_MS;
    this.#healthyResetMs = options?.healthyResetMs ?? DEFAULT_HEALTHY_RESET_MS;
    this.#onPhaseChange = options?.onPhaseChange;
    this.#logger = new RotatingFileLogger(
      path.join(paths.userData, 'logs', 'engine.log'),
      options?.logMaxBytes ?? ENGINE_LOG_MAX_BYTES,
      options?.logMaxRotations ?? ENGINE_LOG_MAX_ROTATIONS,
    );
  }

  get phase(): EngineBrokerPhase {
    return this.#phase;
  }

  get origin(): string | null {
    return this.#phase === 'ready' ? (this.#endpoint?.baseUrl ?? null) : null;
  }

  get stderrTail(): readonly string[] {
    return [...this.#stderrTail];
  }

  get pid(): number | null {
    return this.#child?.pid ?? null;
  }

  get restartCount(): number {
    return this.#restartAttempts;
  }

  get logPath(): string {
    return this.#logger.logPath;
  }

  log(message: string): void {
    this.#logger.append(message);
  }

  readRecentLogLines(maxLines: number = 200): string[] {
    return this.#logger.readRecentLines(maxLines);
  }

  start(): void {
    if (this.#phase !== 'idle') return;
    this.#createReadyPromise();
    this.#spawnEngine();
  }

  #createReadyPromise(): void {
    this.#readyPromise = new Promise<EngineEndpoint>((resolve, reject) => {
      this.#resolveReady = resolve;
      this.#rejectReady = reject;
    });
    this.#readyPromise.catch(() => undefined);
  }

  #spawnEngine(): void {
    if (this.#phase !== 'idle' && this.#phase !== 'restarting') return;
    this.#transition('starting');
    this.#clearReadyTimer();

    let spec: EngineSpec;
    try {
      spec = resolveEngineSpec(this.#paths, process.env.HAWAVOCLEAN_DESKTOP_ENGINE_COMMAND);
      if (this.#paths.packaged) {
        const engineDir = path.dirname(spec.executable);
        const integrity = verifyBundledEngineIntegrity(engineDir, this.#paths.platform);
        if (!integrity.valid) {
          throw new Error(`Bundled engine integrity verification failed: ${integrity.error}`);
        }
      }
      if (path.isAbsolute(spec.executable) && !fs.existsSync(spec.executable)) {
        throw new Error(`Engine executable is missing: ${spec.executable}`);
      }
      fs.mkdirSync(spec.cwd, { recursive: true });
    } catch (error) {
      this.#fail(error instanceof Error ? error : new Error(String(error)));
      return;
    }

    const args = [
      ...spec.prefixArgs,
      'serve',
      '--host',
      '127.0.0.1',
      '--port',
      '0',
      '--token-stdin',
    ];
    const options: SpawnOptions = {
      cwd: spec.cwd,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        PYTHONDONTWRITEBYTECODE: '1',
        PYTHONNOUSERSITE: '1',
        // The broker must tear itself down even when Electron is hard-killed
        // and no before-quit/finally handler can run.  This is a private
        // direct-parent contract consumed by hawavoclean.watchdog.
        HAWAVOCLEAN_PARENT_PID: String(process.pid),
        HAWAVOCLEAN_STATE_DIR: this.#paths.userData,
        HAWAVOCLEAN_WORK_DIR: path.join(this.#paths.temp, 'HawaVoClean', 'work'),
      },
      detached: this.#paths.platform !== 'win32',
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
    };

    let child: ChildProcess;
    try {
      child = spawn(spec.executable, args, options);
    } catch (error) {
      this.#fail(new Error(`Could not start the HawaVoClean engine: ${String(error)}`));
      return;
    }
    this.#child = child;
    this.#logger.append(
      `[broker] spawned engine pid=${String(child.pid ?? 'unknown')} attempt=${String(this.#restartAttempts)}`,
    );

    // One-shot bootstrap channel: close it immediately so the secret is not
    // retained in argv, environment, renderer state, or an open IPC stream.
    child.stdin?.once('error', () => undefined);
    child.stdin?.end(`${this.#token}\n`);

    let stdoutBuffer = '';
    child.stdout?.setEncoding('utf8');
    child.stdout?.on('data', (chunk: string) => {
      if (this.#child !== child) return;
      stdoutBuffer += chunk;
      let newline = stdoutBuffer.indexOf('\n');
      while (newline >= 0) {
        const line = stdoutBuffer.slice(0, newline).replace(/\r$/, '');
        stdoutBuffer = stdoutBuffer.slice(newline + 1);
        this.#handleStdoutLine(line);
        newline = stdoutBuffer.indexOf('\n');
      }
      if (stdoutBuffer.length > LINE_BUFFER_LIMIT) stdoutBuffer = stdoutBuffer.slice(-LINE_BUFFER_LIMIT);
    });

    let stderrBuffer = '';
    child.stderr?.setEncoding('utf8');
    child.stderr?.on('data', (chunk: string) => {
      if (this.#child !== child) return;
      stderrBuffer += chunk;
      let newline = stderrBuffer.indexOf('\n');
      while (newline >= 0) {
        this.#pushStderr(stderrBuffer.slice(0, newline).replace(/\r$/, ''));
        stderrBuffer = stderrBuffer.slice(newline + 1);
        newline = stderrBuffer.indexOf('\n');
      }
      if (stderrBuffer.length > LINE_BUFFER_LIMIT) stderrBuffer = stderrBuffer.slice(-LINE_BUFFER_LIMIT);
    });

    child.once('error', (error) => {
      if (this.#child !== child) return;
      this.#logger.append(`[broker] engine process error: ${error.message}`);
      this.#fail(new Error(`Engine process error: ${error.message}`));
    });
    child.once('exit', (code, signal) => {
      if (this.#child !== child) return;
      this.#clearReadyTimer();
      this.#clearHealthyResetTimer();
      this.invalidateSession();
      this.#endpoint = null;
      this.#cleanupRendezvous();

      const exitSummary = signal ?? `code ${String(code)}`;
      if (this.#phase === 'starting') {
        this.#logger.append(`[broker] engine exited before ready (${exitSummary})`);
        if (this.#hasBeenReady) {
          this.#handleUnexpectedCrash(exitSummary);
        } else {
          this.#fail(new Error(`Engine exited before ready (${exitSummary}).`));
        }
      } else if (this.#phase === 'ready') {
        this.#logger.append(`[broker] engine exited unexpectedly while ready (${exitSummary})`);
        this.#handleUnexpectedCrash(exitSummary);
      }
    });

    this.#readyTimer = setTimeout(() => {
      if (this.#child !== child || this.#phase !== 'starting') return;
      this.#logger.append('[broker] engine did not report ready within 60 seconds');
      this.#fail(new Error('Engine did not report ready within 60 seconds.'));
      signalProcessTree(child, 'SIGTERM');
    }, READY_TIMEOUT_MS);
    this.#readyTimer.unref();
  }

  #handleUnexpectedCrash(exitSummary: string): void {
    if (this.#phase !== 'ready' && this.#phase !== 'starting') return;
    if (this.#restartAttempts >= this.#maxRestartAttempts) {
      const exhaustedError = new Error(
        `Engine crashed repeatedly (${exitSummary}) and exceeded maximum restart attempts (${this.#maxRestartAttempts}).`,
      );
      this.#logger.append(`[broker] ${exhaustedError.message}`);
      this.#fail(exhaustedError);
      return;
    }

    this.#restartAttempts += 1;
    const delayMs = this.#restartBaseDelayMs * 2 ** (this.#restartAttempts - 1);
    const wasReady = this.#phase === 'ready';
    this.#transition('restarting');
    if (wasReady || this.#resolveReady === null) {
      this.#createReadyPromise();
    }
    this.#logger.append(
      `[broker] scheduling crash recovery attempt ${this.#restartAttempts}/${this.#maxRestartAttempts} in ${delayMs}ms`,
    );
    this.#clearRestartTimer();
    this.#restartTimer = setTimeout(() => {
      this.#restartTimer = null;
      if (this.#phase !== 'restarting') return;
      this.#spawnEngine();
    }, delayMs);
    this.#restartTimer.unref();
  }

  async endpoint(): Promise<EngineEndpoint> {
    const endpoint = await this.#waitForEndpoint();
    await this.authorizationHeader();
    return endpoint;
  }

  async authorizationHeader(): Promise<string> {
    const endpoint = await this.#waitForEndpoint();
    const now = performance.now();
    if (!engineSessionNeedsRenewal(this.#session, now) && this.#session !== null) {
      return this.#session.authorization;
    }
    if (this.#sessionPromise === null) {
      this.#sessionPromise = requestEngineSession(endpoint, this.#token);
    }
    const pending = this.#sessionPromise;
    try {
      const capability = await pending;
      this.#session = capability;
      return capability.authorization;
    } finally {
      if (this.#sessionPromise === pending) this.#sessionPromise = null;
    }
  }

  async registerNativeSource(filePath: string): Promise<NativeSourceRegistration> {
    const endpoint = await this.#waitForEndpoint();
    return requestNativeSourceRegistration(endpoint, this.#token, filePath);
  }

  invalidateSession(authorization?: string): void {
    if (authorization !== undefined && this.#session?.authorization !== authorization) return;
    this.#session = null;
  }

  async #waitForEndpoint(): Promise<EngineEndpoint> {
    if (this.#phase === 'idle') this.start();
    if (this.#phase === 'stopping' || this.#phase === 'stopped') {
      throw new Error('Engine broker has been stopped.');
    }
    if (this.#failure) throw this.#failure;
    if (!this.#readyPromise) throw new Error('Engine startup did not initialize.');
    return this.#readyPromise;
  }

  async stop(): Promise<void> {
    if (this.#phase === 'stopped') return;
    if (this.#stopPromise !== null) return this.#stopPromise;
    this.#stopPromise = this.#performStop();
    try {
      await this.#stopPromise;
    } finally {
      this.#stopPromise = null;
    }
  }

  async #performStop(): Promise<void> {
    if (this.#phase === 'stopped') return;
    this.#clearRestartTimer();
    this.#clearReadyTimer();
    this.#clearHealthyResetTimer();
    const wasStartingOrRestarting = this.#phase === 'starting' || this.#phase === 'restarting';
    if (this.#phase !== 'stopping') this.#transition('stopping');
    if (wasStartingOrRestarting || this.#rejectReady !== null) {
      const rejectReady = this.#rejectReady;
      this.#resolveReady = null;
      this.#rejectReady = null;
      const stopError = new Error('Engine broker stopped before ready.');
      if (this.#failure === null) this.#failure = stopError;
      rejectReady?.(stopError);
    }
    this.#cleanupRendezvous();
    const endpoint = this.#endpoint;
    this.#endpoint = null;
    this.invalidateSession();
    const child = this.#child;
    if (child && child.exitCode === null && child.signalCode === null) {
      if (endpoint) {
        await postShutdown(endpoint, this.#token);
        if (!(await waitForExit(child, 1_000))) {
          signalProcessTree(child, 'SIGTERM');
          if (!(await waitForExit(child, SHUTDOWN_GRACE_MS))) {
            if (this.#paths.platform === 'win32') await forceKillWindowsTree(child);
            else signalProcessTree(child, 'SIGKILL');
            await waitForExit(child, 1_000);
          }
        }
      } else {
        signalProcessTree(child, 'SIGTERM');
        if (!(await waitForExit(child, SHUTDOWN_GRACE_MS))) {
          if (this.#paths.platform === 'win32') await forceKillWindowsTree(child);
          else signalProcessTree(child, 'SIGKILL');
          await waitForExit(child, 1_000);
        }
      }
    }
    if (this.#phase === 'stopping') this.#transition('stopped');
    this.#logger.append('[broker] engine stopped cleanly');
  }

  #handleStdoutLine(line: string): void {
    if (!line.trim()) return;
    if (this.#phase !== 'starting') {
      this.#pushStderr(`[stdout] ${line}`);
      return;
    }
    try {
      const parsed = JSON.parse(line) as Record<string, unknown>;
      const port = Number(parsed.port);
      if (parsed.event !== 'ready' || !Number.isInteger(port) || port < 1 || port > 65_535) {
        this.#pushStderr(`[stdout] ${line}`);
        return;
      }
      this.#clearReadyTimer();
      this.#endpoint = rendererEngineEndpoint(port);
      this.#hasBeenReady = true;
      this.#transition('ready');
      this.#writeRendezvous(port);
      this.#logger.append(`[broker] engine ready on loopback port ${String(port)}`);
      this.#armHealthyResetTimer();
      const resolveReady = this.#resolveReady;
      this.#resolveReady = null;
      this.#rejectReady = null;
      resolveReady?.(this.#endpoint);
    } catch {
      this.#pushStderr(`[stdout] ${line}`);
    }
  }

  #armHealthyResetTimer(): void {
    this.#clearHealthyResetTimer();
    if (this.#restartAttempts === 0) return;
    this.#healthyResetTimer = setTimeout(() => {
      this.#healthyResetTimer = null;
      if (this.#phase === 'ready') {
        this.#restartAttempts = 0;
        this.#logger.append('[broker] engine remained healthy; reset crash restart counter');
      }
    }, this.#healthyResetMs);
    this.#healthyResetTimer.unref();
  }

  #clearRestartTimer(): void {
    if (this.#restartTimer !== null) {
      clearTimeout(this.#restartTimer);
      this.#restartTimer = null;
    }
  }

  #clearReadyTimer(): void {
    if (this.#readyTimer !== null) {
      clearTimeout(this.#readyTimer);
      this.#readyTimer = null;
    }
  }

  #clearHealthyResetTimer(): void {
    if (this.#healthyResetTimer !== null) {
      clearTimeout(this.#healthyResetTimer);
      this.#healthyResetTimer = null;
    }
  }

  #writeRendezvous(port: number): void {
    try {
      const origin = this.#endpoint?.baseUrl;
      if (!origin) return;
      const pid = this.#child?.pid ?? process.pid;
      const rendezvousDir = resolveRendezvousDir(this.#paths.userData);
      this.#rendezvousFile = writeRendezvousRecord(rendezvousDir, {
        schemaVersion: 1,
        pid,
        port,
        origin,
        token: this.#token,
        createdAt: new Date().toISOString(),
      });
      this.#rendezvousPid = pid;
    } catch {
      // Best-effort
    }
  }

  #cleanupRendezvous(): void {
    const rFile = this.#rendezvousFile;
    const expectedPid = this.#rendezvousPid ?? this.#child?.pid ?? null;
    this.#rendezvousFile = null;
    this.#rendezvousPid = null;
    if (!rFile || expectedPid === null) return;
    try {
      cleanupOwnedRendezvousFile(rFile, expectedPid);
    } catch {
      // Best-effort
    }
  }

  #pushStderr(line: string): void {
    if (!line) return;
    const bounded = line.slice(0, 4096);
    this.#stderrTail.push(bounded);
    if (this.#stderrTail.length > STDERR_TAIL_LIMIT) {
      this.#stderrTail.splice(0, this.#stderrTail.length - STDERR_TAIL_LIMIT);
    }
    this.#logger.append(bounded);
  }

  #transition(next: EngineBrokerPhase): void {
    const allowed = ALLOWED_ENGINE_BROKER_TRANSITIONS[this.#phase];
    if (!allowed.includes(next)) {
      throw new Error(`Illegal engine broker transition: ${this.#phase} -> ${next}`);
    }
    this.#phase = next;
    try {
      this.#onPhaseChange?.(next);
    } catch {
      // Phase observer errors must not corrupt broker transitions.
    }
  }

  #fail(error: Error): void {
    if (this.#phase !== 'starting' && this.#phase !== 'ready' && this.#phase !== 'restarting') return;
    this.#clearRestartTimer();
    this.#clearReadyTimer();
    this.#clearHealthyResetTimer();
    this.#failure = error;
    this.#transition('failed');
    const rejectReady = this.#rejectReady;
    this.#resolveReady = null;
    this.#rejectReady = null;
    rejectReady?.(error);
  }
}
