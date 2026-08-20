import type {
  ApiError,
  AudioAnalysis,
  CreateJobRequest,
  CreateJobResponse,
  HealthResponse,
  JobStatus,
  PeaksWindow,
} from './types';

export interface Endpoint {
  baseUrl: string;
  token: string;
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
  readonly token: string;

  constructor(endpoint: Endpoint) {
    this.baseUrl = endpoint.baseUrl.replace(/\/+$/, '');
    this.token = endpoint.token;
  }

  private headers(extra?: Record<string, string>): HeadersInit {
    return { 'X-Hawa-Token': this.token, ...(extra ?? {}) };
  }

  private async json<T>(path: string, init: RequestInit = {}, signal?: AbortSignal): Promise<T> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: this.headers(
        init.body !== undefined && !(init.body instanceof FormData)
          ? { 'Content-Type': 'application/json' }
          : undefined,
      ),
      signal: signal ?? null,
    });
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

  createJob(req: CreateJobRequest, signal?: AbortSignal): Promise<CreateJobResponse> {
    return this.json<CreateJobResponse>(
      '/api/jobs',
      { method: 'POST', body: JSON.stringify(req) },
      signal,
    );
  }

  getJob(jobId: string, signal?: AbortSignal): Promise<JobStatus> {
    return this.json<JobStatus>(`/api/jobs/${encodeURIComponent(jobId)}`, { method: 'GET' }, signal);
  }

  cancelJob(jobId: string): Promise<{ ok: boolean }> {
    return this.json<{ ok: boolean }>(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: 'POST',
    });
  }

  async upload(file: File, signal?: AbortSignal): Promise<{ path: string }> {
    const form = new FormData();
    form.append('file', file, file.name);
    return this.json<{ path: string }>('/api/upload', { method: 'POST', body: form }, signal);
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
  ): Promise<{ path: string }> {
    return new Promise<{ path: string }>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${this.baseUrl}/api/upload`, true);
      xhr.setRequestHeader('X-Hawa-Token', this.token);
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
          resolve({ path: body.path });
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
    const res = await fetch(this.fileUrl(path), {
      headers: { Range: 'bytes=0-0' },
      cache: 'no-store',
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
    const res = await fetch(this.fileUrl(path), { signal: signal ?? null });
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
   * The token must travel as a query parameter — an `<audio src>` and a
   * download anchor cannot carry a header.
   */
  fileUrl(path: string): string {
    const q = new URLSearchParams({ path, token: this.token });
    return `${this.baseUrl}/api/audio?${q.toString()}`;
  }

  /** @deprecated name — `fileUrl` says what it actually does. */
  audioUrl(path: string): string {
    return this.fileUrl(path);
  }

  /** URL for the job's `EventSource` stream. */
  eventsUrl(jobId: string): string {
    const q = new URLSearchParams({ token: this.token });
    return `${this.baseUrl}/api/jobs/${encodeURIComponent(jobId)}/events?${q.toString()}`;
  }
}
