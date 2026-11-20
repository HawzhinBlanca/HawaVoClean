import type {
  ApiError,
  AudioAnalysis,
  AudioAssistantStatus,
  AudioPlan,
  AudioPlanRequest,
  BatchSummary,
  CapabilitiesResponseV1,
  CreateJobRequest,
  CreateJobResponse,
  CreateV1JobResponse,
  DeliveryExportRequest,
  DeliveryExportResponse,
  DeliveryPresetId,
  ExportFormatId,
  HealthResponse,
  JobStage,
  JobState,
  JobStatus,
  ModelInstallResponseV1,
  ModelsResponseV1,
  ModelStatusV1,
  PeaksWindow,
  ProcessingRequestV1,
  Profile,
  RegionPreviewMetrics,
  RegionPreviewRequest,
  RegionPreviewResponse,
  SpectralRepairRequest,
  SpectralRepairResponse,
  SpectrogramScale,
  SpectrogramTileRequest,
  SpectrogramTileResponse,
  TrackExtractResponse,
  TrackInfo,
  TrackInspectResponse,
  UnitDecisionRecord,
  UnitOverrideDecision,
  UnitOverrideRequest,
  UnitOverrideResponse,
  UnitSummary,
  UploadResponse,
} from './types';

const V1_STATE_MAP: Readonly<Record<string, JobState>> = Object.freeze({
  queued: 'queued',
  running: 'running',
  analyzing: 'running',
  rendering: 'running',
  guarding: 'running',
  publishing: 'running',
  done: 'done',
  completed: 'done',
  failed: 'failed',
  cancelled: 'cancelled',
  interrupted: 'interrupted',
});

/**
 * Normalize either a camelCase v1 `JobStatusResponseV1` or a snake_case
 * `JobStatus` payload into the UI's canonical `JobStatus` shape.
 */
export function normalizeJobStatus(raw: Record<string, unknown>): JobStatus {
  const rawState = typeof raw.state === 'string' ? raw.state : 'failed';
  const state: JobState = V1_STATE_MAP[rawState] ?? 'failed';
  const jobId =
    typeof raw.job_id === 'string'
      ? raw.job_id
      : typeof raw.jobId === 'string'
        ? raw.jobId
        : '';
  const outputPath =
    typeof raw.output_path === 'string'
      ? raw.output_path
      : typeof raw.outputPath === 'string'
        ? raw.outputPath
        : '';
  const reportPath =
    typeof raw.report_path === 'string'
      ? raw.report_path
      : typeof raw.reportPath === 'string'
        ? raw.reportPath
        : '';
  const inputPath =
    typeof raw.input_path === 'string'
      ? raw.input_path
      : typeof raw.inputPath === 'string'
        ? raw.inputPath
        : '';
  const startedAt =
    raw.started_at !== undefined
      ? (raw.started_at as string | null)
      : raw.startedAt !== undefined
        ? (raw.startedAt as string | null)
        : null;
  const finishedAt =
    raw.finished_at !== undefined
      ? (raw.finished_at as string | null)
      : raw.finishedAt !== undefined
        ? (raw.finishedAt as string | null)
        : null;
  const fallbackStage: JobStage =
    state === 'done' ? 'done' : state === 'failed' || state === 'interrupted' ? 'error' : 'preflight';
  const stage = (typeof raw.stage === 'string' ? raw.stage : fallbackStage) as JobStage;
  const progress =
    typeof raw.progress === 'number' && Number.isFinite(raw.progress)
      ? raw.progress
      : state === 'done'
        ? 1
        : 0;
  const message = typeof raw.message === 'string' ? raw.message : '';
  const profile = (typeof raw.profile === 'string' ? raw.profile : 'production') as Profile;

  return {
    ...(raw as Partial<JobStatus>),
    job_id: jobId,
    state,
    stage,
    progress,
    message,
    input_path: inputPath,
    output_path: outputPath,
    report_path: reportPath,
    profile,
    started_at: startedAt,
    finished_at: finishedAt,
  };
}

export interface Endpoint {
  baseUrl: string;
}

/**
 * What a one-byte ranged GET of an artefact actually established.
 * See {@link EngineClient.verify} for why a HEAD cannot be trusted here.
 */
export interface ArtifactProbe {
  /** The engine's HTTP answer (206 for the ranged byte, 200 for a whole file). */
  status: number;
  /** The answer was 2xx *and* the requested byte really arrived. */
  delivered: boolean;
  /**
   * The file's full length — `Content-Range`'s denominator, or
   * `Content-Length` on a 200 — or null when the engine did not say.
   */
  size: number | null;
}

export class EngineError extends Error {
  readonly status: number;
  readonly code: string;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = 'EngineError';
    this.status = status;
    this.code = code;
  }
}

async function parseError(res: Response): Promise<EngineError> {
  let code = `http_${res.status}`;
  let message = `${res.status} ${res.statusText}`.trim();
  try {
    const body = (await res.json()) as Partial<ApiError>;
    if (body && typeof body.error === 'string') code = body.error;
    if (body && typeof body.message === 'string') message = body.message;
    else if (body && typeof body.error === 'string') message = body.error;
  } catch {
    /* non-JSON error body */
  }
  return new EngineError(res.status, code, message);
}

/**
 * C5 · a 200 is not a promise that the body parses.
 *
 * A proxy, a half-written response, or an engine of a different vintage can
 * all answer 200 with something that is not the JSON this client expects.
 * `res.json()` then throws a bare `SyntaxError: Unexpected end of JSON input`,
 * which is an exception, not a designed state — so it is converted here into
 * an `EngineError` the UI already knows how to describe.
 */
async function parseBody<T>(res: Response): Promise<T> {
  const text = await res.text();
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new EngineError(
      res.status,
      'bad_response',
      `the engine answered ${res.status} with a body this client could not parse (${text.length} bytes)`,
    );
  }
}

export class EngineClient {
  readonly baseUrl: string;

  constructor(endpoint: Endpoint) {
    this.baseUrl = endpoint.baseUrl.replace(/\/+$/, '');
  }

  /**
   * `fetch`, with a transport failure turned into an `EngineError(0, …)`.
   *
   * A dead socket makes `fetch` reject with a bare `TypeError`, and those were
   * escaping this class raw. That mattered because `state/errors.ts` had to
   * catch them somewhere, and the only handle it has on a raw TypeError is
   * `e instanceof TypeError` — which also matches every null-dereference in a
   * React event handler that reaches the app's failure net. The result was a
   * genuine app bug being published to the user as "The engine is not
   * answering… this comes back on its own when it reconnects", while the
   * header lamp said READY: a screen contradicting itself, telling the reader
   * to wait for a recovery that was never coming.
   *
   * Normalising here means every network-shaped failure arrives with
   * `status === 0`, which errors.ts already classifies as offline and
   * `probeSoon` already re-measures on. The XHR upload path in this file
   * already did exactly this (`new EngineError(0, 'network', …)`); this brings
   * the fetch paths in line. An abort is left alone — it is the user's, not
   * the network's.
   */
  private async fetchOrOffline(input: string, init: RequestInit): Promise<Response> {
    try {
      return await fetch(input, init);
    } catch (e) {
      if (e instanceof DOMException && (e.name === 'AbortError' || e.name === 'TimeoutError')) {
        throw e;
      }
      throw new EngineError(0, 'engine_offline', e instanceof Error ? e.message : String(e));
    }
  }

  private async json<T>(path: string, init: RequestInit = {}, signal?: AbortSignal): Promise<T> {
    const request: RequestInit = {
      ...init,
      // Same-origin web mode authenticates with its HttpOnly session cookie.
      // The custom-scheme desktop shell injects a short-lived Authorization
      // header in main; renderer code never receives either credential.
      credentials: 'same-origin',
      signal: signal ?? null,
    };
    if (init.body !== undefined && !(init.body instanceof FormData)) {
      request.headers = { 'Content-Type': 'application/json' };
    }
    const res = await this.fetchOrOffline(`${this.baseUrl}${path}`, request);
    if (!res.ok) throw await parseError(res);
    return await parseBody<T>(res);
  }

  health(signal?: AbortSignal): Promise<HealthResponse> {
    return this.json<HealthResponse>('/api/health', { method: 'GET' }, signal);
  }

  analyze(path: string, buckets = 1200, signal?: AbortSignal): Promise<AudioAnalysis> {
    return this.json<AudioAnalysis>(
      '/api/analyze',
      { method: 'POST', body: JSON.stringify({ path, buckets }) },
      signal,
    );
  }

  /** Windowed peaks for the visible zoom range (docs/ui-contract.md, Addendum 1). */
  peaks(
    path: string,
    startS: number,
    endS: number,
    buckets: number,
    signal?: AbortSignal,
  ): Promise<PeaksWindow> {
    return this.json<PeaksWindow>(
      '/api/peaks',
      {
        method: 'POST',
        body: JSON.stringify({ path, start_s: startS, end_s: endS, buckets }),
      },
      signal,
    );
  }

  async capabilities(signal?: AbortSignal): Promise<CapabilitiesResponseV1> {
    const res = await this.json<any>('/api/v1/capabilities', { method: 'GET' }, signal);
    const caps = (res.capabilities ?? []).map((c: any) => ({
      capability_id: c.capability_id ?? c.capabilityId,
      available: Boolean(c.available),
      maturity: c.maturity,
      reason: c.reason ?? null,
      manifest_sha256: c.manifest_sha256 ?? c.manifestSha256 ?? null,
      providers: c.providers ?? [],
    }));
    return {
      schema_version: res.schema_version ?? res.schemaVersion ?? 1,
      capabilities: caps,
    };
  }

  async createV1Jobs(req: ProcessingRequestV1, signal?: AbortSignal): Promise<CreateV1JobResponse> {
    const res = await this.json<any>(
      '/api/v1/jobs',
      { method: 'POST', body: JSON.stringify(req) },
      signal,
    );
    const jobs = (res.jobs ?? []).map((j: any) => ({
      jobId: j.jobId ?? j.job_id,
      sourceId: j.sourceId ?? j.source_id,
      outputPath: j.outputPath ?? j.output_path,
      reportPath: j.reportPath ?? j.report_path,
    }));
    const batchId = res.batchId ?? res.batch_id;
    return {
      schemaVersion: res.schemaVersion ?? res.schema_version ?? 1,
      ...(typeof batchId === 'string' && batchId ? { batchId } : {}),
      ...(typeof res.execution === 'string' ? { execution: res.execution } : {}),
      jobs,
    };
  }

  /** Planning only: the response is reviewed before a separate job request. */
  async assistantStatus(signal?: AbortSignal): Promise<AudioAssistantStatus> {
    const value = await this.json<AudioAssistantStatus>('/api/v1/assistant/status', { method: 'GET' }, signal);
    if (!value || !['auto', 'rules', 'ollama', 'openai', 'unconfigured'].includes(value.provider)) {
      throw new EngineError(502, 'bad_response', 'The assistant status is unavailable.');
    }
    return value;
  }

  /** Planning never changes the recording or starts an audio job. */
  async planAudio(req: AudioPlanRequest, signal?: AbortSignal): Promise<AudioPlan> {
    const value: unknown = await this.json<unknown>(
      '/api/v1/assistant/plan',
      { method: 'POST', body: JSON.stringify(req) },
      signal,
    );
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      throw new EngineError(502, 'bad_plan', 'The assistant returned an unreadable plan. Please try again.');
    }
    const p = value as Record<string, unknown>;
    const validStrength =
      p.strength === undefined ||
      (typeof p.strength === 'number' && Number.isFinite(p.strength) && p.strength >= 0.1 && p.strength <= 1.0);
    const validTimeRange = (() => {
      if (p.time_range === undefined || p.time_range === null) return true;
      if (typeof p.time_range !== 'object' || Array.isArray(p.time_range)) return false;
      const tr = p.time_range as Record<string, unknown>;
      return (
        typeof tr.start_s === 'number' &&
        typeof tr.end_s === 'number' &&
        Number.isFinite(tr.start_s) &&
        Number.isFinite(tr.end_s) &&
        tr.start_s >= 0 &&
        tr.end_s > tr.start_s &&
        tr.end_s <= 86400
      );
    })();
    const valid =
      ['clean', 'restore', 'extract', 'remove', 'clarify'].includes(String(p.operation)) &&
      (p.target === null || (typeof p.target === 'string' && p.target.trim().length > 0 && p.target.length <= 2000)) &&
      (p.cleanup_profile === null || ['production', 'studio', 'studio_pro', 'dereverb', 'lowband'].includes(String(p.cleanup_profile))) &&
      ['small', 'base', 'large'].includes(String(p.model_size)) &&
      p.quality === 'high' &&
      validStrength &&
      validTimeRange &&
      typeof p.explanation === 'string' && p.explanation.trim().length > 0 && p.explanation.length <= 4000 &&
      ['rules', 'ollama', 'openai'].includes(String(p.provider)) &&
      typeof p.needs_clarification === 'boolean' &&
      ((p.operation === 'clarify') === p.needs_clarification) &&
      (!['extract', 'remove'].includes(String(p.operation)) || typeof p.target === 'string') &&
      (!['clean', 'restore', 'clarify'].includes(String(p.operation)) || p.target === null) &&
      (p.operation !== 'clean' || p.cleanup_profile !== null) &&
      (p.operation !== 'restore' || p.cleanup_profile === null) &&
      (p.operation !== 'clarify' || (p.cleanup_profile === null && (p.time_range === undefined || p.time_range === null)));
    if (!valid) {
      throw new EngineError(502, 'bad_plan', 'The assistant could not make a clear plan. Please describe the sound again.');
    }
    return p as unknown as AudioPlan;
  }

  /** @deprecated Use createV1Jobs (/api/v1/jobs) with source IDs. Scheduled for removal 2026-10-01 (v1.0.0). */
  createJob(req: CreateJobRequest, signal?: AbortSignal): Promise<CreateJobResponse> {
    return this.json<CreateJobResponse>(
      '/api/jobs',
      { method: 'POST', body: JSON.stringify(req) },
      signal,
    );
  }

  /** @deprecated Use getV1Job (/api/v1/jobs/{jobId}). Scheduled for removal 2026-10-01 (v1.0.0). */
  async getJob(jobId: string, signal?: AbortSignal): Promise<JobStatus> {
    const raw = await this.json<Record<string, unknown>>(
      `/api/jobs/${encodeURIComponent(jobId)}`,
      { method: 'GET' },
      signal,
    );
    return normalizeJobStatus(raw);
  }

  async getV1Job(jobId: string, signal?: AbortSignal): Promise<JobStatus> {
    const raw = await this.json<Record<string, unknown>>(
      `/api/v1/jobs/${encodeURIComponent(jobId)}`,
      { method: 'GET' },
      signal,
    );
    return normalizeJobStatus(raw);
  }

  /** @deprecated Use cancelV1Job (/api/v1/jobs/{jobId}/cancel). Scheduled for removal 2026-10-01 (v1.0.0). */
  cancelJob(jobId: string): Promise<{ ok: boolean }> {
    return this.json<{ ok: boolean }>(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: 'POST',
    });
  }

  cancelV1Job(jobId: string): Promise<{ ok: boolean }> {
    return this.json<{ ok: boolean }>(`/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: 'POST',
    });
  }

  async upload(file: File, signal?: AbortSignal): Promise<UploadResponse> {
    const form = new FormData();
    form.append('file', file, file.name);
    return this.json<UploadResponse>('/api/upload', { method: 'POST', body: form }, signal);
  }

  /**
   * `POST /api/upload` with real byte-level progress and a real cancel.
   *
   * `fetch` cannot report request progress (a `ReadableStream` request body
   * needs HTTP/2 and full-duplex support, which loopback HTTP/1.1 does not
   * give us), so the upload path is the one place in this client that still
   * uses `XMLHttpRequest` — it is the only API that exposes `upload.onprogress`.
   * `cancel` returns a function that aborts the transfer; the promise then
   * rejects with an `AbortError`, exactly like an aborted `fetch`.
   */
  uploadWithProgress(
    file: File,
    opts: {
      onProgress?: (loaded: number, total: number) => void;
      onCancelHandle?: (cancel: () => void) => void;
    } = {},
  ): Promise<UploadResponse> {
    return new Promise<UploadResponse>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${this.baseUrl}/api/upload`, true);
      xhr.responseType = 'text';
      let cancelled = false;
      opts.onCancelHandle?.(() => {
        cancelled = true;
        xhr.abort();
      });
      xhr.upload.onprogress = (e: ProgressEvent): void => {
        opts.onProgress?.(e.loaded, e.lengthComputable ? e.total : file.size);
      };
      xhr.onerror = () => reject(new EngineError(0, 'network', 'Engine unreachable'));
      xhr.ontimeout = () => reject(new EngineError(0, 'timeout', 'Upload timed out'));
      xhr.onabort = () =>
        reject(new DOMException(cancelled ? 'Upload cancelled' : 'Aborted', 'AbortError'));
      xhr.onload = () => {
        const text = typeof xhr.response === 'string' ? xhr.response : '';
        let body: Record<string, unknown> = {};
        try {
          body = text ? (JSON.parse(text) as Record<string, unknown>) : {};
        } catch {
          /* non-JSON body */
        }
        if (xhr.status >= 200 && xhr.status < 300 && typeof body.path === 'string') {
          // The bytes are on the wire the moment `upload.onprogress` reaches
          // 100%, but the engine still has to write them out; hold the bar at
          // full until the response lands so the two never disagree.
          opts.onProgress?.(file.size, file.size);
          const sourceId = typeof body.source_id === 'string' && body.source_id ? body.source_id : undefined;
          resolve({ path: body.path, ...(sourceId ? { source_id: sourceId } : {}) });
          return;
        }
        const code = typeof body.error === 'string' ? body.error : `http_${xhr.status}`;
        const message =
          typeof body.message === 'string'
            ? body.message
            : `${xhr.status} ${xhr.statusText}`.trim() || 'Upload failed';
        reject(new EngineError(xhr.status, code, message));
      };
      const form = new FormData();
      form.append('file', file, file.name);
      xhr.send(form);
    });
  }

  /**
   * Is this file still where the run left it — *really*?
   *
   * A HEAD used to be the whole check, and a HEAD is a stat: the engine
   * answers 200 for a master that has been `chmod 000`ed (the open fails only
   * when a body is produced) and for one truncated to 100 bytes (the stat is
   * happy; the audio is gone). So the check now reads something: one byte,
   * `Range: bytes=0-0`, consumed to completion — the same discipline the
   * player's own probe uses, so no `net::ERR_ABORTED` is left in the log.
   *
   * `delivered` is the load-bearing answer: the engine committed a 2xx *and*
   * the byte actually arrived. `size` is the file's full length from
   * `Content-Range` (or `Content-Length` on a 200), which is what lets a
   * caller who recorded the master's size at load time catch a truncation the
   * status code will never admit to.
   *
   * A thrown `fetch` (the engine is not answering at all) is *not* an answer
   * and is re-raised, because "the engine is offline" and "your master was
   * deleted" are two different things and the UI says two different things
   * about them.
   */
  async verify(path: string, signal?: AbortSignal): Promise<ArtifactProbe> {
    const res = await this.fetchOrOffline(this.fileUrl(path), {
      headers: { Range: 'bytes=0-0' },
      cache: 'no-store',
      credentials: 'same-origin',
      signal: signal ?? null,
    });
    let delivered = false;
    try {
      // Read the byte (or the error body) to completion. A 2xx whose body
      // dies mid-read is the chmod-000 shape: headers committed, bytes never
      // produced. That is a fact about the *file*, not a transport error.
      await res.arrayBuffer();
      delivered = res.ok;
    } catch {
      delivered = false;
    }
    // `bytes 0-0/13624364` — the denominator is the whole file. A 200 (the
    // engine ignores ranges on an empty file) carries plain Content-Length.
    const total = res.headers.get('Content-Range')?.split('/')[1];
    let size = Number(total ?? Number.NaN);
    if (!Number.isFinite(size) && res.status === 200) {
      size = Number(res.headers.get('Content-Length') ?? Number.NaN);
    }
    return { status: res.status, delivered, size: Number.isFinite(size) ? size : null };
  }

  /** Plain text of a served file (the human-readable report sidecar). */
  async fetchText(path: string, signal?: AbortSignal): Promise<string> {
    const res = await this.fetchOrOffline(this.fileUrl(path), {
      credentials: 'same-origin',
      signal: signal ?? null,
    });
    if (!res.ok) throw await parseError(res);
    return await res.text();
  }

  shutdown(): Promise<{ ok: boolean }> {
    return this.json<{ ok: boolean }>('/api/shutdown', { method: 'POST' });
  }

  /**
   * URL for any file the engine will serve under its path policy: the audio a
   * deck plays, and equally the JSON report and its .txt sidecar (`/api/audio`
   * types the response from the file's own extension, so it serves all three).
   * Authentication is deliberately absent from this URL. Same-origin web
   * mode uses an HttpOnly cookie; the hardened shell injects Authorization at
   * its network boundary for `<audio>`, downloads and ordinary fetches.
   */
  fileUrl(path: string): string {
    const q = new URLSearchParams({ path });
    return `${this.baseUrl}/api/audio?${q.toString()}`;
  }

  /** @deprecated Use v1EventsUrl (/api/v1/jobs/{jobId}/events). Scheduled for removal 2026-10-01 (v1.0.0). */
  eventsUrl(jobId: string): string {
    return `${this.baseUrl}/api/jobs/${encodeURIComponent(jobId)}/events`;
  }

  /** Credential-free URL for the job's v1 `EventSource` stream. */
  v1EventsUrl(jobId: string): string {
    return `${this.baseUrl}/api/v1/jobs/${encodeURIComponent(jobId)}/events`;
  }

  listBatches(limit = 50): Promise<{ schemaVersion: number; batches: BatchSummary[] }> {
    return this.json<{ schemaVersion: number; batches: BatchSummary[] }>(
      `/api/v1/batches?limit=${encodeURIComponent(limit)}`,
    );
  }

  getBatch(batchId: string): Promise<BatchSummary> {
    return this.json<BatchSummary>(`/api/v1/batches/${encodeURIComponent(batchId)}`);
  }

  pauseBatch(batchId: string): Promise<{ ok: boolean; batchId: string; state: string }> {
    return this.json<{ ok: boolean; batchId: string; state: string }>(
      `/api/v1/batches/${encodeURIComponent(batchId)}/pause`,
      { method: 'POST' },
    );
  }

  resumeBatch(batchId: string): Promise<{ ok: boolean; batchId: string; state: string }> {
    return this.json<{ ok: boolean; batchId: string; state: string }>(
      `/api/v1/batches/${encodeURIComponent(batchId)}/resume`,
      { method: 'POST' },
    );
  }

  cancelBatch(
    batchId: string,
    wait = false,
  ): Promise<{ ok: boolean; batchId: string; state: string }> {
    return this.json<{ ok: boolean; batchId: string; state: string }>(
      `/api/v1/batches/${encodeURIComponent(batchId)}/cancel?wait=${wait ? 'true' : 'false'}`,
      { method: 'POST' },
    );
  }

  retryJob(jobId: string): Promise<Record<string, unknown>> {
    return this.json<Record<string, unknown>>(
      `/api/v1/jobs/${encodeURIComponent(jobId)}/retry`,
      { method: 'POST' },
    );
  }

  v1BatchEventsUrl(batchId: string): string {
    return `${this.baseUrl}/api/v1/batches/${encodeURIComponent(batchId)}/events`;
  }

  // ---- Pro Studio API methods (Wave 2) --------------------------------------

  async fetchSpectrogramTile(
    req: SpectrogramTileRequest,
    signal?: AbortSignal,
  ): Promise<SpectrogramTileResponse> {
    const body: Record<string, unknown> = {
      path: req.path,
      mode: req.mode,
      startS: req.startS,
      endS: req.endS,
    };
    if (req.cleanedPath) body.cleanedPath = req.cleanedPath;
    if (req.widthCols !== undefined) body.widthCols = req.widthCols;
    if (req.nBins !== undefined) body.nBins = req.nBins;
    if (req.nFft !== undefined) body.nFft = req.nFft;
    if (req.scale !== undefined) body.scale = req.scale;
    if (req.dbFloor !== undefined) body.dbFloor = req.dbFloor;
    if (req.dbCeil !== undefined) body.dbCeil = req.dbCeil;

    const raw = await this.json<Record<string, unknown>>(
      '/api/v1/spectrogram',
      { method: 'POST', body: JSON.stringify(body) },
      signal,
    );
    const num = (a: unknown, b: unknown, fallback = 0): number => {
      const v = a !== undefined ? a : b;
      return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
    };
    const scaleRaw = raw.scale;
    const scale: SpectrogramScale =
      scaleRaw === 'mel' || scaleRaw === 'linear' || scaleRaw === 'log'
        ? scaleRaw
        : (req.scale ?? 'log');
    return {
      start_s: num(raw.start_s, raw.startS, req.startS),
      end_s: num(raw.end_s, raw.endS, req.endS),
      sample_rate: num(raw.sample_rate, raw.sampleRate, 48000),
      f_min_hz: num(raw.f_min_hz, raw.fMinHz, 20),
      f_max_hz: num(raw.f_max_hz, raw.fMaxHz, 24000),
      n_bins: num(raw.n_bins, raw.nBins, 0),
      width_cols: num(raw.width_cols, raw.widthCols, 0),
      scale,
      db_floor: num(raw.db_floor, raw.dbFloor, -60),
      db_ceil: num(raw.db_ceil, raw.dbCeil, 0),
      data_b64: String(raw.data_b64 ?? raw.dataB64 ?? ''),
    };
  }

  async renderRegionPreview(
    req: RegionPreviewRequest,
    signal?: AbortSignal,
  ): Promise<RegionPreviewResponse> {
    const body: Record<string, unknown> = {
      path: req.path,
      startS: req.startS,
      endS: req.endS,
    };
    if (req.profile !== undefined) body.profile = req.profile;
    if (req.strength !== undefined) body.strength = req.strength;
    if (req.ambienceFloorDb !== undefined) body.ambienceFloorDb = req.ambienceFloorDb;
    if (req.hpfEnabled !== undefined) body.hpfEnabled = req.hpfEnabled;
    if (req.hpfCutoffHz !== undefined) body.hpfCutoffHz = req.hpfCutoffHz;
    if (req.deliveryPreset !== undefined) body.deliveryPreset = req.deliveryPreset;
    if (req.stereoMode !== undefined) body.stereoMode = req.stereoMode;
    if (req.proRepair !== undefined) body.proRepair = req.proRepair;

    const raw = await this.json<Record<string, unknown>>(
      '/api/v1/preview',
      { method: 'POST', body: JSON.stringify(body) },
      signal,
    );
    const normMetrics = (m: unknown): RegionPreviewMetrics => {
      const rec = (m && typeof m === 'object' ? m : {}) as Record<string, unknown>;
      const lufsVal = rec.lufs ?? rec.integrated_lufs;
      const noiseVal = rec.noiseFloorDb ?? rec.noise_floor_db;
      const mosVal = rec.mos;
      return {
        lufs: typeof lufsVal === 'number' && Number.isFinite(lufsVal) ? lufsVal : null,
        noiseFloorDb: typeof noiseVal === 'number' && Number.isFinite(noiseVal) ? noiseVal : null,
        ...(typeof mosVal === 'number' && Number.isFinite(mosVal) ? { mos: mosVal } : {}),
      };
    };
    return {
      previewPath: String(raw.previewPath ?? raw.preview_path ?? ''),
      audioUrl: String(raw.audioUrl ?? raw.audio_url ?? ''),
      renderMs: Number(raw.renderMs ?? raw.render_ms ?? 0),
      warmHit: Boolean(raw.warmHit ?? raw.warm_hit ?? false),
      metricsBefore: normMetrics(raw.metricsBefore ?? raw.metrics_before),
      metricsAfter: normMetrics(raw.metricsAfter ?? raw.metrics_after),
      durationS: Number(raw.durationS ?? raw.duration_s ?? Math.max(0, req.endS - req.startS)),
    };
  }

  async overrideUnit(
    jobId: string,
    req: UnitOverrideRequest,
    signal?: AbortSignal,
  ): Promise<UnitOverrideResponse> {
    const body: Record<string, unknown> = {
      unitId: req.unitId,
      decision: req.decision,
    };
    if (req.strength !== undefined) body.strength = req.strength;

    const raw = await this.json<Record<string, unknown>>(
      `/api/v1/jobs/${encodeURIComponent(jobId)}/override`,
      { method: 'POST', body: JSON.stringify(body) },
      signal,
    );
    const restitchMsRaw = raw.restitchMs ?? raw.restitch_ms;
    const masterShaRaw = raw.masterSha256 ?? raw.master_sha256;
    return {
      jobId: String(raw.jobId ?? raw.job_id ?? jobId),
      unitIndex: Number(raw.unitIndex ?? raw.unit_index ?? req.unitId),
      unitId: Number(raw.unitId ?? raw.unit_id ?? req.unitId),
      oldDecision: String(raw.oldDecision ?? raw.old_decision ?? ''),
      newDecision: String(raw.newDecision ?? raw.new_decision ?? ''),
      override: (raw.override ?? req.decision) as UnitOverrideDecision,
      restitched: Boolean(raw.restitched),
      ...(typeof restitchMsRaw === 'number' && Number.isFinite(restitchMsRaw)
        ? { restitchMs: restitchMsRaw }
        : {}),
      ...(typeof masterShaRaw === 'string' && masterShaRaw ? { masterSha256: masterShaRaw } : {}),
      ...(raw.summary && typeof raw.summary === 'object'
        ? { summary: raw.summary as UnitSummary }
        : {}),
      ...(Array.isArray(raw.units) ? { units: raw.units as UnitDecisionRecord[] } : {}),
    };
  }

  async exportDelivery(
    req: DeliveryExportRequest,
    signal?: AbortSignal,
  ): Promise<DeliveryExportResponse> {
    const body: Record<string, unknown> = {
      masterPath: req.masterPath,
      format: req.format,
    };
    if (req.destPath) body.destPath = req.destPath;
    if (req.deliveryPreset !== undefined) body.deliveryPreset = req.deliveryPreset;
    if (req.sourceMediaPath) body.sourceMediaPath = req.sourceMediaPath;
    if (req.preserveMetadata !== undefined) body.preserveMetadata = req.preserveMetadata;

    const raw = await this.json<Record<string, unknown>>(
      '/api/v1/export',
      { method: 'POST', body: JSON.stringify(body) },
      signal,
    );
    const optNum = (a: unknown, b: unknown): number | null => {
      const v = a !== undefined ? a : b;
      return typeof v === 'number' && Number.isFinite(v) ? v : null;
    };
    return {
      outputPath: String(raw.outputPath ?? raw.output_path ?? ''),
      format: (raw.format ?? req.format) as ExportFormatId,
      deliveryPreset: (raw.deliveryPreset ??
        raw.delivery_preset ??
        req.deliveryPreset ??
        'match_input') as DeliveryPresetId,
      lraLu: optNum(raw.lraLu, raw.lra_lu),
      integratedLufs: optNum(raw.integratedLufs, raw.integrated_lufs),
      truePeakDbtp: optNum(raw.truePeakDbtp, raw.true_peak_dbtp),
    };
  }

  async inspectTracks(path: string, signal?: AbortSignal): Promise<TrackInspectResponse> {
    const raw = await this.json<Record<string, unknown>>(
      '/api/v1/tracks/inspect',
      { method: 'POST', body: JSON.stringify({ path }) },
      signal,
    );
    const tracks = Array.isArray(raw.tracks)
      ? raw.tracks.map((item): TrackInfo => {
          const t = (item && typeof item === 'object' ? item : {}) as Record<string, unknown>;
          const idx = Number(t.index ?? 0);
          return {
            index: idx,
            name: String(t.name ?? `Track ${idx + 1}`),
            sample_rate: Number(t.sample_rate ?? t.sampleRate ?? 48000),
            samples: Number(t.samples ?? 0),
            duration_s: Number(t.duration_s ?? t.durationS ?? 0),
            rms_dbfs: Number(t.rms_dbfs ?? t.rmsDb ?? t.rms_db ?? -Infinity),
            peak_dbfs: Number(t.peak_dbfs ?? t.peakDbfs ?? t.peak_db ?? -Infinity),
            speech_activity_ratio: Number(
              t.speech_activity_ratio ?? t.speechActivity ?? t.speech_activity ?? 0,
            ),
            likely_dialogue: Boolean(t.likely_dialogue ?? t.likelyDialogue ?? false),
          };
        })
      : [];
    const trackCount = Number(raw.trackCount ?? raw.track_count ?? tracks.length);
    return {
      path: String(raw.path ?? path),
      trackCount,
      tracks,
    };
  }

  async extractTracks(
    path: string,
    trackIndices: number[],
    outDir?: string,
    signal?: AbortSignal,
  ): Promise<TrackExtractResponse> {
    const body: Record<string, unknown> = {
      path,
      trackIndices,
    };
    if (outDir !== undefined) body.outDir = outDir;

    const raw = await this.json<Record<string, unknown>>(
      '/api/v1/tracks/extract',
      { method: 'POST', body: JSON.stringify(body) },
      signal,
    );
    const extractedRaw = raw.extractedPaths ?? raw.extracted_paths;
    const extractedPaths = Array.isArray(extractedRaw) ? extractedRaw.map((p) => String(p)) : [];
    const sourceIdsRaw = raw.sourceIds ?? raw.source_ids;
    const sourceIds = Array.isArray(sourceIdsRaw)
      ? sourceIdsRaw.map((id) => (typeof id === 'string' ? id : String(id ?? '')))
      : undefined;
    const outDirRaw = raw.outDir ?? raw.out_dir;
    return {
      ...(typeof raw.path === 'string' ? { path: raw.path } : {}),
      ...(typeof outDirRaw === 'string' ? { outDir: outDirRaw } : {}),
      extractedPaths,
      ...(sourceIds ? { sourceIds } : {}),
    };
  }

  async repairSpectralBox(
    req: SpectralRepairRequest,
    signal?: AbortSignal,
  ): Promise<SpectralRepairResponse> {
    const body: Record<string, unknown> = {
      path: req.path,
      startS: req.startS,
      endS: req.endS,
      fLowHz: req.fLowHz,
      fHighHz: req.fHighHz,
    };
    if (req.attenuationDb !== undefined) body.attenuationDb = req.attenuationDb;

    const raw = await this.json<Record<string, unknown>>(
      '/api/v1/spectral-repair',
      { method: 'POST', body: JSON.stringify(body) },
      signal,
    );
    return {
      outputPath: String(raw.outputPath ?? raw.output_path ?? ''),
      audioUrl: String(raw.audioUrl ?? raw.audio_url ?? ''),
      startS: Number(raw.startS ?? raw.start_s ?? req.startS),
      endS: Number(raw.endS ?? raw.end_s ?? req.endS),
      fLowHz: Number(raw.fLowHz ?? raw.f_low_hz ?? req.fLowHz),
      fHighHz: Number(raw.fHighHz ?? raw.f_high_hz ?? req.fHighHz),
      attenuationDb: Number(raw.attenuationDb ?? raw.attenuation_db ?? req.attenuationDb ?? 24),
      renderMs: Number(raw.renderMs ?? raw.render_ms ?? 0),
    };
  }

  // ---- Hash-Verified Model Manager API (Wave E) -----------------------------

  async getModels(signal?: AbortSignal): Promise<ModelsResponseV1> {
    const raw = await this.json<Record<string, unknown>>('/api/v1/models', { method: 'GET' }, signal);
    const models = Array.isArray(raw.models)
      ? raw.models.map((item): ModelStatusV1 => {
          const m = (item && typeof item === 'object' ? item : {}) as Record<string, unknown>;
          return {
            model_id: String(m.model_id ?? m.modelId ?? ''),
            display_name: String(m.display_name ?? m.displayName ?? ''),
            description: String(m.description ?? ''),
            used_by: Array.isArray(m.used_by) ? m.used_by.map((u) => String(u)) : [],
            license: String(m.license ?? ''),
            expected_sha256: String(m.expected_sha256 ?? m.expectedSha256 ?? ''),
            approx_size_mb: Number(m.approx_size_mb ?? m.approxSizeMb ?? 0),
            bundled: Boolean(m.bundled),
            downloadable: Boolean(m.downloadable),
            installed: Boolean(m.installed),
            verified: Boolean(m.verified),
            size_bytes: Number(m.size_bytes ?? m.sizeBytes ?? 0),
            ...(typeof m.path === 'string' || m.path === null ? { path: m.path } : {}),
          };
        })
      : [];
    return {
      schema_version: 1,
      models,
    };
  }

  async installModel(modelId: string, signal?: AbortSignal): Promise<ModelInstallResponseV1> {
    const raw = await this.json<Record<string, unknown>>(
      `/api/v1/models/${encodeURIComponent(modelId)}/install`,
      { method: 'POST' },
      signal,
    );
    const resObj = (raw.result && typeof raw.result === 'object'
      ? raw.result
      : {}) as Record<string, unknown>;
    const models = Array.isArray(raw.models)
      ? raw.models.map((item): ModelStatusV1 => {
          const m = (item && typeof item === 'object' ? item : {}) as Record<string, unknown>;
          return {
            model_id: String(m.model_id ?? m.modelId ?? ''),
            display_name: String(m.display_name ?? m.displayName ?? ''),
            description: String(m.description ?? ''),
            used_by: Array.isArray(m.used_by) ? m.used_by.map((u) => String(u)) : [],
            license: String(m.license ?? ''),
            expected_sha256: String(m.expected_sha256 ?? m.expectedSha256 ?? ''),
            approx_size_mb: Number(m.approx_size_mb ?? m.approxSizeMb ?? 0),
            bundled: Boolean(m.bundled),
            downloadable: Boolean(m.downloadable),
            installed: Boolean(m.installed),
            verified: Boolean(m.verified),
            size_bytes: Number(m.size_bytes ?? m.sizeBytes ?? 0),
            ...(typeof m.path === 'string' || m.path === null ? { path: m.path } : {}),
          };
        })
      : [];
    const capabilities = Array.isArray(raw.capabilities)
      ? raw.capabilities.map((c: any) => ({
          capability_id: c.capability_id ?? c.capabilityId,
          available: Boolean(c.available),
          maturity: c.maturity,
          reason: c.reason ?? null,
          manifest_sha256: c.manifest_sha256 ?? c.manifestSha256 ?? null,
          providers: c.providers ?? [],
        }))
      : undefined;
    return {
      schema_version: 1,
      ok: Boolean(raw.ok ?? true),
      result: {
        model_id: String(resObj.model_id ?? resObj.modelId ?? modelId),
        installed: Boolean(resObj.installed),
        verified: Boolean(resObj.verified),
        sha256: String(resObj.sha256 ?? ''),
        size_bytes: Number(resObj.size_bytes ?? resObj.sizeBytes ?? 0),
        ...(typeof resObj.path === 'string' || resObj.path === null ? { path: resObj.path } : {}),
        already_installed: Boolean(resObj.already_installed ?? resObj.alreadyInstalled ?? false),
        message: String(resObj.message ?? ''),
      },
      models,
      ...(capabilities !== undefined ? { capabilities } : {}),
    };
  }
}
