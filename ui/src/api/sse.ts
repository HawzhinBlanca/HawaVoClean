import type { EngineClient } from './client';
import type { JobState, JobStatus } from './types';

const TERMINAL: ReadonlySet<JobState> = new Set<JobState>(['done', 'failed', 'cancelled']);

export function isTerminal(state: JobState): boolean {
  return TERMINAL.has(state);
}

export interface JobStreamHandlers {
  onStatus: (status: JobStatus) => void;
  onEnd: () => void;
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
      client
        .getJob(jobId)
        .then((st) => {
          if (closed || finished) return;
          handlers.onStatus(st);
          if (isTerminal(st.state)) finish();
          else connect();
        })
        .catch(() => {
          if (!closed && !finished) connect();
        });
    }, delay);
  };

  const connect = (): void => {
    if (closed || finished) return;
    cleanup();
    es = new EventSource(client.eventsUrl(jobId));
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
