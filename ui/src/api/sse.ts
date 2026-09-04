import { EngineError, type EngineClient } from './client';
import type { BatchSummary, JobState, JobStatus } from './types';

const TERMINAL: ReadonlySet<JobState> = new Set<JobState>(['done', 'failed', 'cancelled']);

export function isTerminal(state: JobState): boolean {
  return TERMINAL.has(state);
}

export interface JobStreamHandlers {
  onStatus: (status: JobStatus) => void;
  onEnd: () => void;
  /**
   * The engine answered, and it has never heard of this job (404). That is not
   * a dropped connection: the process that owned the run is gone, so the run
   * is gone with it. Without this the stream would reopen against a route that
   * 404s forever and the job would sit at "running" for the rest of the
   * session (goal box B6).
   */
  onGone?: () => void;
  onConnectionChange?: (connected: boolean) => void;
}

/**
 * Follow a job over SSE. Reconnects (with back-off) whenever the stream drops
 * before the job reaches a terminal state; on every reconnect the server
 * re-sends the current status so nothing is lost. Returns a disposer.
 */
export function followJob(
  client: EngineClient,
  jobId: string,
  handlers: JobStreamHandlers,
): () => void {
  let es: EventSource | null = null;
  let closed = false;
  let finished = false;
  let attempt = 0;
  let retryTimer: number | null = null;

  const finish = (): void => {
    if (finished) return;
    finished = true;
    cleanup();
    handlers.onEnd();
  };

  const cleanup = (): void => {
    if (es) {
      es.onopen = null;
      es.onerror = null;
      es.close();
      es = null;
    }
    if (retryTimer !== null) {
      window.clearTimeout(retryTimer);
      retryTimer = null;
    }
  };

  const scheduleReconnect = (): void => {
    if (closed || finished) return;
    cleanup();
    handlers.onConnectionChange?.(false);
    attempt += 1;
    const delay = Math.min(8000, 300 * 2 ** Math.min(attempt, 5));
    retryTimer = window.setTimeout(() => {
      retryTimer = null;
      // Poll once so a job that finished while we were offline still resolves.
      const fetchStatus = typeof client.getV1Job === 'function'
        ? client.getV1Job(jobId)
        : client.getJob(jobId);
      fetchStatus
        .then((st) => {
          if (closed || finished) return;
          handlers.onStatus(st);
          if (isTerminal(st.state)) finish();
          else connect();
        })
        .catch((err: unknown) => {
          if (closed || finished) return;
          if (err instanceof EngineError && err.status === 404) {
            handlers.onGone?.();
            finish();
            return;
          }
          connect();
        });
    }, delay);
  };

  const connect = (): void => {
    if (closed || finished) return;
    cleanup();
    const url = typeof client.v1EventsUrl === 'function' ? client.v1EventsUrl(jobId) : client.eventsUrl(jobId);
    es = new EventSource(url);
    es.onopen = () => {
      attempt = 0;
      handlers.onConnectionChange?.(true);
    };
    es.addEventListener('status', (ev) => {
      try {
        const st = JSON.parse((ev as MessageEvent<string>).data) as JobStatus;
        handlers.onStatus(st);
        if (isTerminal(st.state)) finish();
      } catch {
        /* malformed event — ignore, next status will supersede */
      }
    });
    es.addEventListener('end', () => finish());
    es.onerror = () => {
      // EventSource auto-retries, but we own the back-off + polling fallback.
      scheduleReconnect();
    };
  };

  connect();

  return () => {
    closed = true;
    cleanup();
  };
}

export interface BatchStreamHandlers {
  onBatchStatus: (status: BatchSummary) => void;
  onEnd: () => void;
  onError?: (err: unknown) => void;
}

/**
 * Follow a batch over SSE (/api/v1/batches/{batchId}/events).
 * Handles auto-reconnection and parses incoming `batch_status` frames.
 */
export function followBatch(
  client: EngineClient,
  batchId: string,
  handlers: BatchStreamHandlers,
): () => void {
  let es: EventSource | null = null;
  let closed = false;
  let finished = false;
  let retryTimer: number | null = null;

  const finish = (): void => {
    if (finished) return;
    finished = true;
    cleanup();
    handlers.onEnd();
  };

  const cleanup = (): void => {
    if (es) {
      es.onopen = null;
      es.onerror = null;
      es.close();
      es = null;
    }
    if (retryTimer !== null) {
      window.clearTimeout(retryTimer);
      retryTimer = null;
    }
  };

  const connect = (): void => {
    if (closed || finished) return;
    cleanup();
    try {
      es = new EventSource(client.v1BatchEventsUrl(batchId));
    } catch (e) {
      handlers.onError?.(e);
      return;
    }

    es.addEventListener('batch_status', (ev) => {
      try {
        const parsed = JSON.parse((ev as MessageEvent<string>).data) as BatchSummary;
        handlers.onBatchStatus(parsed);
        if (
          parsed.state === 'done' ||
          parsed.state === 'failed' ||
          parsed.state === 'cancelled'
        ) {
          finish();
        }
      } catch {
        /* malformed event */
      }
    });

    es.addEventListener('end', () => {
      finish();
    });

    es.onerror = (e) => {
      if (closed || finished) return;
      cleanup();
      handlers.onError?.(e);
      retryTimer = window.setTimeout(connect, 2000);
    };
  };

  connect();

  return () => {
    closed = true;
    cleanup();
  };
}

