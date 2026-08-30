import { spawn, type ChildProcess, type SpawnOptions } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { performance } from 'node:perf_hooks';

import type { EngineEndpoint } from './contracts.js';

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

/** Mint a bounded, short-lived renderer capability without exposing the root secret. */
export function requestEngineSession(
  endpoint: EngineEndpoint,
  rootToken: string,
  now: () => number = () => performance.now(),
): Promise<EngineSessionCapability> {
  let url: URL;
  try {
    url = exactEngineUrl(endpoint);
  } catch (error) {
    return Promise.reject(error instanceof Error ? error : new Error('Engine endpoint is invalid.'));
  }
  if (!rootToken || rootToken.length > 256) {
    return Promise.reject(new Error('Engine bootstrap secret is invalid.'));
  }
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (error: Error | null, value?: EngineSessionCapability): void => {
      if (settled) return;
      settled = true;
      if (error) reject(error);
      else if (value) resolve(value);
      else reject(new Error('Engine session bootstrap produced no result.'));
    };
    const request = http.request(
      {
        hostname: '127.0.0.1',
        port: Number(url.port),
        method: 'POST',
        path: '/api/session',
        headers: {
          'X-Hawa-Token': rootToken,
          Accept: 'application/json',
          'Content-Length': '0',
        },
        timeout: SESSION_HTTP_TIMEOUT_MS,
      },
      (response) => {
        const chunks: Buffer[] = [];
        let bytes = 0;
        response.on('data', (chunk: Buffer) => {
          bytes += chunk.length;
          if (bytes > SESSION_RESPONSE_LIMIT) {
            response.destroy();
            finish(new Error('Engine session response exceeded its size limit.'));
            return;
          }
          chunks.push(chunk);
        });
        response.once('aborted', () => finish(new Error('Engine session response was interrupted.')));
        response.once('error', () => finish(new Error('Engine session response failed.')));
        response.once('end', () => {
          if (response.statusCode !== 200) {
            finish(
              new Error(`Engine refused session bootstrap (HTTP ${String(response.statusCode ?? 0)}).`),
            );
            return;
          }
          const cacheControl = String(response.headers['cache-control'] ?? '').toLowerCase();
          if (!cacheControl.includes('no-store')) {
            finish(new Error('Engine session response was not marked no-store.'));
            return;
          }
          let body: unknown;
          try {
            body = JSON.parse(Buffer.concat(chunks).toString('utf8')) as unknown;
          } catch {
            finish(new Error('Engine session response was not valid JSON.'));
            return;
          }
          if (typeof body !== 'object' || body === null || Array.isArray(body)) {
            finish(new Error('Engine session response had the wrong shape.'));
            return;
          }
          const record = body as Record<string, unknown>;
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
            finish(new Error('Engine session response had invalid capability metadata.'));
            return;
          }
          const issuedAt = now();
          const ttlMs = ttlS * 1000;
          const skewMs = Math.min(SESSION_RENEW_SKEW_MS, Math.max(1_000, ttlMs / 4));
          finish(null, {
            authorization: `Bearer ${token}`,
            expiresAtMs: issuedAt + ttlMs,
            refreshAtMs: issuedAt + Math.max(1, ttlMs - skewMs),
          });
        });
      },
    );
    request.once('timeout', () => {
      request.destroy();
      finish(new Error('Engine session bootstrap timed out.'));
    });
    request.once('error', () => finish(new Error('Engine session bootstrap failed.')));
    request.end();
  });
}

/** Register one user-selected native file with the root-owned broker channel. */
export function requestNativeSourceRegistration(
  endpoint: EngineEndpoint,
  rootToken: string,
  filePath: string,
): Promise<NativeSourceRegistration> {
  let url: URL;
  try {
    url = exactEngineUrl(endpoint);
  } catch (error) {
    return Promise.reject(error instanceof Error ? error : new Error('Engine endpoint is invalid.'));
  }
  if (!rootToken || rootToken.length > 256) {
    return Promise.reject(new Error('Engine bootstrap secret is invalid.'));
  }
  if (
    typeof filePath !== 'string' ||
    filePath.length < 1 ||
    filePath.length > 32_768 ||
    filePath.includes('\0') ||
    !path.isAbsolute(filePath)
  ) {
    return Promise.reject(new Error('Native source path is invalid.'));
  }
  const body = Buffer.from(JSON.stringify({ path: filePath }), 'utf8');
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (error: Error | null, value?: NativeSourceRegistration): void => {
      if (settled) return;
      settled = true;
      if (error) reject(error);
      else if (value) resolve(value);
      else reject(new Error('Native source registration produced no result.'));
    };
    const request = http.request(
      {
        hostname: '127.0.0.1',
        port: Number(url.port),
        method: 'POST',
        path: '/api/v1/native-sources',
        headers: {
          'X-Hawa-Token': rootToken,
          Accept: 'application/json',
          'Content-Type': 'application/json',
          'Content-Length': String(body.length),
        },
        timeout: NATIVE_SOURCE_HTTP_TIMEOUT_MS,
      },
      (response) => {
        const chunks: Buffer[] = [];
        let bytes = 0;
        response.on('data', (chunk: Buffer) => {
          bytes += chunk.length;
          if (bytes > NATIVE_SOURCE_RESPONSE_LIMIT) {
            response.destroy();
            finish(new Error('Native source registration response exceeded its size limit.'));
            return;
          }
          chunks.push(chunk);
        });
        response.once('aborted', () => finish(new Error('Native source registration was interrupted.')));
        response.once('error', () => finish(new Error('Native source registration response failed.')));
        response.once('end', () => {
          if (response.statusCode !== 200) {
            finish(new Error(`Engine refused the selected file (HTTP ${String(response.statusCode ?? 0)}).`));
            return;
          }
          const cacheControl = String(response.headers['cache-control'] ?? '').toLowerCase();
          if (!cacheControl.includes('no-store')) {
            finish(new Error('Native source registration response was not marked no-store.'));
            return;
          }
          let parsed: unknown;
          try {
            parsed = JSON.parse(Buffer.concat(chunks).toString('utf8')) as unknown;
          } catch {
            finish(new Error('Native source registration response was not valid JSON.'));
            return;
          }
          if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
            finish(new Error('Native source registration response had the wrong shape.'));
            return;
          }
          const record = parsed as Record<string, unknown>;
          if (
            typeof record.sourceId !== 'string' ||
            !/^[0-9a-f]{32}$/.test(record.sourceId) ||
            typeof record.path !== 'string' ||
            record.path.length < 1 ||
            record.path.length > 32_768 ||
            record.path.includes('\0') ||
            !path.isAbsolute(record.path)
          ) {
            finish(new Error('Native source registration response had invalid capability metadata.'));
            return;
          }
          finish(null, Object.freeze({ sourceId: record.sourceId, path: record.path }));
        });
      },
    );
    request.once('timeout', () => {
      request.destroy();
      finish(new Error('Native source registration timed out.'));
    });
    request.once('error', () => finish(new Error('Native source registration failed.')));
    request.end(body);
  });
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

export class EngineBroker {
  readonly #paths: EnginePaths;
  readonly #token = crypto.randomBytes(32).toString('hex');
  readonly #stderrTail: string[] = [];
  #child: ChildProcess | null = null;
  #endpoint: EngineEndpoint | null = null;
  #session: EngineSessionCapability | null = null;
  #sessionPromise: Promise<EngineSessionCapability> | null = null;
  #failure: Error | null = null;
  #readyPromise: Promise<EngineEndpoint> | null = null;
  #resolveReady: ((endpoint: EngineEndpoint) => void) | null = null;
  #rejectReady: ((error: Error) => void) | null = null;

  constructor(paths: EnginePaths) {
    this.#paths = paths;
  }

  get origin(): string | null {
    return this.#endpoint?.baseUrl ?? null;
  }

  get stderrTail(): readonly string[] {
    return [...this.#stderrTail];
  }

  start(): void {
    if (this.#readyPromise !== null) return;
    this.#readyPromise = new Promise<EngineEndpoint>((resolve, reject) => {
      this.#resolveReady = resolve;
      this.#rejectReady = reject;
    });
    this.#readyPromise.catch(() => undefined);

    let spec: EngineSpec;
    try {
      spec = resolveEngineSpec(this.#paths, process.env.HAWAVOCLEAN_DESKTOP_ENGINE_COMMAND);
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
    // One-shot bootstrap channel: close it immediately so the secret is not
    // retained in argv, environment, renderer state, or an open IPC stream.
    child.stdin?.once('error', () => undefined);
    child.stdin?.end(`${this.#token}\n`);

    let stdoutBuffer = '';
    child.stdout?.setEncoding('utf8');
    child.stdout?.on('data', (chunk: string) => {
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
      stderrBuffer += chunk;
      let newline = stderrBuffer.indexOf('\n');
      while (newline >= 0) {
        this.#pushStderr(stderrBuffer.slice(0, newline).replace(/\r$/, ''));
        stderrBuffer = stderrBuffer.slice(newline + 1);
        newline = stderrBuffer.indexOf('\n');
      }
      if (stderrBuffer.length > LINE_BUFFER_LIMIT) stderrBuffer = stderrBuffer.slice(-LINE_BUFFER_LIMIT);
    });

    child.once('error', (error) => this.#fail(new Error(`Engine process error: ${error.message}`)));
    child.once('exit', (code, signal) => {
      this.invalidateSession();
      if (this.#endpoint === null && this.#failure === null) {
        this.#fail(new Error(`Engine exited before ready (${signal ?? `code ${String(code)}`}).`));
      }
    });

    const timer = setTimeout(() => {
      if (this.#endpoint === null && this.#failure === null) {
        this.#fail(new Error('Engine did not report ready within 60 seconds.'));
        signalProcessTree(child, 'SIGTERM');
      }
    }, READY_TIMEOUT_MS);
    timer.unref();
    this.#readyPromise.then(
      () => clearTimeout(timer),
      () => clearTimeout(timer),
    );
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
    this.start();
    if (this.#failure) throw this.#failure;
    if (!this.#readyPromise) throw new Error('Engine startup did not initialize.');
    return this.#readyPromise;
  }

  async stop(): Promise<void> {
    const child = this.#child;
    if (!child || child.exitCode !== null || child.signalCode !== null) return;
    if (this.#endpoint) {
      await postShutdown(this.#endpoint, this.#token);
      if (await waitForExit(child, 1_000)) return;
    }
    signalProcessTree(child, 'SIGTERM');
    if (await waitForExit(child, SHUTDOWN_GRACE_MS)) return;
    if (this.#paths.platform === 'win32') await forceKillWindowsTree(child);
    else signalProcessTree(child, 'SIGKILL');
    await waitForExit(child, 1_000);
    this.invalidateSession();
  }

  #handleStdoutLine(line: string): void {
    if (!line.trim()) return;
    if (this.#endpoint !== null || this.#failure !== null) {
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
      this.#endpoint = rendererEngineEndpoint(port);
      this.#resolveReady?.(this.#endpoint);
    } catch {
      this.#pushStderr(`[stdout] ${line}`);
    }
  }

  #pushStderr(line: string): void {
    if (!line) return;
    this.#stderrTail.push(line.slice(0, 4096));
    if (this.#stderrTail.length > STDERR_TAIL_LIMIT) {
      this.#stderrTail.splice(0, this.#stderrTail.length - STDERR_TAIL_LIMIT);
    }
  }

  #fail(error: Error): void {
    if (this.#failure !== null || this.#endpoint !== null) return;
    this.#failure = error;
    this.#rejectReady?.(error);
  }
}
