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
    return (await res.json()) as T;
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

  shutdown(): Promise<{ ok: boolean }> {
    return this.json<{ ok: boolean }>('/api/shutdown', { method: 'POST' });
  }

  /** URL for `<audio src>` — the token must travel as a query parameter. */
  audioUrl(path: string): string {
    const q = new URLSearchParams({ path, token: this.token });
    return `${this.baseUrl}/api/audio?${q.toString()}`;
  }

  /** URL for the job's `EventSource` stream. */
  eventsUrl(jobId: string): string {
    const q = new URLSearchParams({ token: this.token });
    return `${this.baseUrl}/api/jobs/${encodeURIComponent(jobId)}/events?${q.toString()}`;
  }
}
