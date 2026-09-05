import os from 'node:os';

import type { DiagnosticsState } from './contracts.js';

export interface DiagnosticsOptions {
  initialOptIn?: boolean;
}

export interface SanitizedDiagnosticError {
  readonly timestampIso: string;
  readonly code: string;
  readonly message: string;
  readonly stack?: string | undefined;
}

export interface RedactedDiagnosticReport {
  readonly schemaVersion: '1.0.0';
  readonly timestampIso: string;
  readonly optInConfirmed: true;
  readonly app: Readonly<{
    name: string;
    version: string;
    platform: NodeJS.Platform;
    arch: string;
    packaged: boolean;
  }>;
  readonly system: Readonly<{
    nodeVersion: string;
    electronVersion: string;
    osType: string;
    osRelease: string;
    cpus: number;
    memoryTotalGb: number;
  }>;
  readonly engine: Readonly<{
    status: 'running' | 'stopped' | 'uninitialized';
    pid: number | null;
  }>;
  readonly recentErrors: readonly SanitizedDiagnosticError[];
}
const MAX_RECORDED_ERRORS = 50;

/** Redact local user home and personal path components from a path or string. */
export function redactPath(raw: string): string {
  if (!raw || typeof raw !== 'string') return '';
  let sanitized = raw;

  // Redact explicit home directory
  const home = os.homedir();
  if (home && home.length > 2) {
    sanitized = sanitized.replaceAll(home, '[USER_HOME]');
    // Also handle forward-slash normalized version for Windows/URIs
    const homeForward = home.replaceAll('\\', '/');
    sanitized = sanitized.replaceAll(homeForward, '[USER_HOME]');
  }

  // Redact Unix user dirs: /Users/<name>/... or /home/<name>/...
  sanitized = sanitized.replace(/(?:^|(?<=[\s"'(]))\/(?:Users|home)\/[^/\s"']+/g, '[USER_HOME]');

  // Redact Windows user dirs: C:\Users\<name>\...
  sanitized = sanitized.replace(/(?:^|(?<=[\s"'(]))[A-Za-z]:\\(?:Users)\\[^\\\s"']+/g, '[USER_HOME]');

  // Redact file:/// URLs pointing to user directories
  sanitized = sanitized.replace(/file:\/\/\/(?:Users|home)\/[^/\s"']+/g, 'file:///[USER_HOME]');

  return sanitized;
}

/** Redact bearer tokens, credentials, and authentication secrets. */
export function redactSecrets(raw: string): string {
  if (!raw || typeof raw !== 'string') return '';
  let sanitized = raw.replace(/Bearer\s+[A-Za-z0-9_-]+/gi, 'Bearer [REDACTED_SECRET]');
  sanitized = sanitized.replace(/(?:authorization|x-hawa-token|token|password|secret|key)["']?\s*[:=]\s*["']?[^"'\s,;]+/gi, (match) => {
    const separator = match.includes(':') ? ':' : '=';
    const key = match.split(separator)[0];
    return `${key}${separator} "[REDACTED_SECRET]"`;
  });
  return sanitized;
}

/** Redact potential raw transcripts, dialogue text, and spoken content. */
export function redactContent(raw: string): string {
  if (!raw || typeof raw !== 'string') return '';
  // Redact Arabic/Kurdish Unicode scripts (0600-06FF, 0750-077F, FB50-FDFF, FE70-FEFF)
  let sanitized = raw.replace(/[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+/g, '[REDACTED_TRANSCRIPT]');
  return sanitized;
}

/** Deeply sanitize an arbitrary error or message before diagnostic recording. */
export function sanitizeMessage(raw: string): string {
  return redactContent(redactSecrets(redactPath(raw)));
}

/** Sanitize an error stack trace by removing local paths and any embedded tokens. */
export function sanitizeStack(rawStack?: string): string | undefined {
  if (!rawStack || typeof rawStack !== 'string') return undefined;
  const lines = rawStack.split(/\r?\n/).map((line) => sanitizeMessage(line));
  return lines.join('\n');
}

/** Verify that a payload is fully redacted and contains no leaked tokens, personal paths, or transcripts. */
export function assertRedactionPurity(serializedJson: string): void {
  const home = os.homedir();
  if (home && home.length > 2 && serializedJson.includes(home)) {
    throw new Error('Privacy violation: diagnostic payload contains user home path.');
  }
  if (/Bearer\s+[A-Za-z0-9_-]{20,}/i.test(serializedJson)) {
    throw new Error('Privacy violation: diagnostic payload contains unredacted Bearer token.');
  }
  if (/[\u0600-\u06FF]{3,}/.test(serializedJson)) {
    throw new Error('Privacy violation: diagnostic payload contains unredacted Kurdish/Arabic transcript text.');
  }
}

export class DiagnosticsManager {
  private optIn: boolean;
  private readonly errors: SanitizedDiagnosticError[] = [];

  constructor(options: DiagnosticsOptions = {}) {
    // Explicitly opt-in only: defaults to false unless explicitly configured
    this.optIn = Boolean(options.initialOptIn);
  }

  get isOptedIn(): boolean {
    return this.optIn;
  }

  getState(): DiagnosticsState {
    return Object.freeze({
      status: this.optIn ? ('enabled' as const) : ('ready' as const),
      optIn: this.optIn,
      canExport: this.optIn && this.errors.length > 0,
      telemetryEgress: 'none' as const,
      pendingErrorCount: this.errors.length,
      ...(this.optIn ? {} : { reason: 'Opt-in required to retain diagnostics' }),
    });
  }

  getErrorCount(): number {
    return this.errors.length;
  }

  setOptIn(enabled: boolean): DiagnosticsState {
    this.optIn = Boolean(enabled);
    if (!this.optIn) {
      // Clear any stored diagnostic events when opting out
      this.errors.length = 0;
    }
    return this.getState();
  }

  recordError(code: string, error: unknown): void {
    if (!this.optIn) return; // Discard completely when not opted in

    const rawMessage = error instanceof Error ? error.message : String(error);
    const rawStack = error instanceof Error ? error.stack : undefined;

    const sanitizedError: SanitizedDiagnosticError = Object.freeze({
      timestampIso: new Date().toISOString(),
      code: sanitizeMessage(code),
      message: sanitizeMessage(rawMessage),
      ...(rawStack ? { stack: sanitizeStack(rawStack) } : {}),
    });

    this.errors.push(sanitizedError);
    if (this.errors.length > MAX_RECORDED_ERRORS) {
      this.errors.shift();
    }
  }

  generateReport(params?: Partial<{
    appName: string;
    appVersion: string;
    packaged: boolean;
    engineStatus: 'running' | 'stopped' | 'uninitialized';
    enginePid: number | null;
  }>): RedactedDiagnosticReport {
    if (!this.optIn) {
      throw new Error('Diagnostics collection is disabled: opt-in required to generate report.');
    }

    const report: RedactedDiagnosticReport = Object.freeze({
      schemaVersion: '1.0.0' as const,
      timestampIso: new Date().toISOString(),
      optInConfirmed: true as const,
      app: Object.freeze({
        name: params?.appName ?? 'HawaVoClean',
        version: params?.appVersion ?? '3.3.0',
        platform: process.platform,
        arch: process.arch,
        packaged: params?.packaged ?? false,
      }),
      system: Object.freeze({
        nodeVersion: process.versions.node,
        electronVersion: process.versions.electron ?? 'none',
        osType: os.type(),
        osRelease: os.release(),
        cpus: os.cpus()?.length ?? 1,
        memoryTotalGb: Math.round(os.totalmem() / (1024 * 1024 * 1024)),
      }),
      engine: Object.freeze({
        status: params?.engineStatus ?? ('uninitialized' as const),
        pid: params?.enginePid ?? null,
      }),
      recentErrors: Object.freeze([...this.errors]),
    });

    // Run privacy verification assertion
    assertRedactionPurity(JSON.stringify(report));

    return report;
  }
}
