// api/sse.ts — following a job over SSE. The failure modes are all timing:
// a stream that drops mid-run, an engine that comes back without the job, a
// terminal event that arrives while we are between sockets.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { EngineError, type EngineClient } from './client';
import { followJob, isTerminal } from './sse';
import type { JobState, JobStatus } from './types';

class FakeEventSource {
  static instances: FakeEventSource[] = [];
  static get live(): FakeEventSource {
    const es = FakeEventSource.instances[FakeEventSource.instances.length - 1];
    if (!es) throw new Error('no EventSource was opened');
    return es;
  }
  readonly url: string;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;
  private readonly listeners = new Map<string, Array<(e: Event) => void>>();

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }
  addEventListener(type: string, fn: (e: Event) => void): void {
    const list = this.listeners.get(type) ?? [];
    list.push(fn);
    this.listeners.set(type, list);
  }
  close(): void {
    this.closed = true;
  }
  // --- drivers -------------------------------------------------------------
  // A closed EventSource delivers nothing: the browser has torn the socket
  // down. The fake honours that, so "events after the end" means "events the
  // real thing would still have delivered", not "events we invented".
  connect(): void {
    if (!this.closed) this.onopen?.();
  }
  send(type: string, data: string): void {
    if (this.closed) return;
    for (const fn of this.listeners.get(type) ?? []) fn({ data } as unknown as Event);
  }
  drop(): void {
    if (!this.closed) this.onerror?.();
  }
}

function status(state: JobState, over: Partial<JobStatus> = {}): JobStatus {
  return {
    job_id: 'j1',
    state,
    stage: state === 'done' ? 'done' : 'enhance',
    progress: state === 'done' ? 1 : 0.4,
    message: state,
    input_path: '/a.wav',
    output_path: '/out/a.wav',
    report_path: '/out/a.hawavoclean.json',
    profile: 'studio',
    started_at: null,
    finished_at: null,
    ...over,
  };
}

function stubClient(): { client: EngineClient; getJob: ReturnType<typeof vi.fn> } {
  const getJob = vi.fn();
  const client = {
    getJob,
    eventsUrl: (id: string) => `http://127.0.0.1:8765/api/jobs/${id}/events?token=devtok`,
  } as unknown as EngineClient;
  return { client, getJob };
}

function handlers() {
  return {
    onStatus: vi.fn(),
    onEnd: vi.fn(),
    onGone: vi.fn(),
    onConnectionChange: vi.fn(),
  };
}

beforeEach(() => {
  FakeEventSource.instances = [];
  vi.stubGlobal('EventSource', FakeEventSource);
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('isTerminal', () => {
  it('names exactly the three states a run can end in', () => {
    expect(['done', 'failed', 'cancelled'].map((s) => isTerminal(s as JobState))).toEqual([
      true,
      true,
      true,
    ]);
    expect(['queued', 'running'].map((s) => isTerminal(s as JobState))).toEqual([false, false]);
  });
});

describe('the happy path', () => {
  it('opens the stream against the job’s own event URL', () => {
    const { client } = stubClient();
    followJob(client, 'j1', handlers());
    expect(FakeEventSource.instances).toHaveLength(1);
    expect(FakeEventSource.live.url).toContain('/api/jobs/j1/events');
  });

  it('reports the connection and forwards each status', () => {
    const { client } = stubClient();
    const h = handlers();
    followJob(client, 'j1', h);
    FakeEventSource.live.connect();
    expect(h.onConnectionChange).toHaveBeenCalledWith(true);
    FakeEventSource.live.send('status', JSON.stringify(status('running')));
    expect(h.onStatus).toHaveBeenCalledTimes(1);
    expect(h.onStatus.mock.calls[0]?.[0]).toMatchObject({ state: 'running', job_id: 'j1' });
    expect(h.onEnd).not.toHaveBeenCalled();
  });

  it('lands on a terminal status, ends once, and closes the socket', () => {
    const { client } = stubClient();
    const h = handlers();
    followJob(client, 'j1', h);
    FakeEventSource.live.connect();
    const es = FakeEventSource.live;
    es.send('status', JSON.stringify(status('done')));
    expect(h.onEnd).toHaveBeenCalledTimes(1);
    expect(es.closed).toBe(true);
    // The socket is down, so nothing more can arrive over it — and nothing
    // reopens it either.
    es.send('status', JSON.stringify(status('running')));
    es.drop();
    vi.advanceTimersByTime(30000);
    expect(h.onEnd).toHaveBeenCalledTimes(1);
    expect(h.onStatus).toHaveBeenCalledTimes(1);
    expect(FakeEventSource.instances).toHaveLength(1);
  });

  it('ends on the server’s explicit end event too', () => {
    const { client } = stubClient();
    const h = handlers();
    followJob(client, 'j1', h);
    FakeEventSource.live.send('end', '');
    expect(h.onEnd).toHaveBeenCalledTimes(1);
    expect(FakeEventSource.live.closed).toBe(true);
  });

  it('ignores a malformed event instead of throwing out of the handler', () => {
    const { client } = stubClient();
    const h = handlers();
    followJob(client, 'j1', h);
    expect(() => FakeEventSource.live.send('status', '{"state":')).not.toThrow();
    expect(h.onStatus).not.toHaveBeenCalled();
    expect(h.onEnd).not.toHaveBeenCalled();
  });
});

describe('reconnection', () => {
  it('marks the connection down and backs off 600 → 8000 ms', async () => {
    const { client, getJob } = stubClient();
    const h = handlers();
    getJob.mockResolvedValue(status('running'));
    followJob(client, 'j1', h);

    const expected = [600, 1200, 2400, 4800, 8000, 8000];
    for (const [i, delay] of expected.entries()) {
      FakeEventSource.live.drop();
      expect(h.onConnectionChange).toHaveBeenLastCalledWith(false);
      await vi.advanceTimersByTimeAsync(delay - 1);
      expect(getJob, `attempt ${i + 1} fired early`).toHaveBeenCalledTimes(i);
      await vi.advanceTimersByTimeAsync(1);
      expect(getJob, `attempt ${i + 1} did not fire`).toHaveBeenCalledTimes(i + 1);
    }
  });

  it('polls once on the way back, so a run that finished offline still lands', async () => {
    const { client, getJob } = stubClient();
    const h = handlers();
    getJob.mockResolvedValue(status('done'));
    followJob(client, 'j1', h);
    FakeEventSource.live.drop();
    await vi.advanceTimersByTimeAsync(600);
    expect(h.onStatus).toHaveBeenCalledTimes(1);
    expect(h.onStatus.mock.calls[0]?.[0]).toMatchObject({ state: 'done' });
    expect(h.onEnd).toHaveBeenCalledTimes(1);
    expect(FakeEventSource.instances).toHaveLength(1); // never reopened
  });

  it('reopens the stream when the poll says the run is still going', async () => {
    const { client, getJob } = stubClient();
    const h = handlers();
    getJob.mockResolvedValue(status('running'));
    followJob(client, 'j1', h);
    FakeEventSource.live.drop();
    await vi.advanceTimersByTimeAsync(600);
    expect(FakeEventSource.instances).toHaveLength(2);
    expect(h.onEnd).not.toHaveBeenCalled();
  });

  it('resets the back-off once a stream actually opens', async () => {
    const { client, getJob } = stubClient();
    const h = handlers();
    getJob.mockResolvedValue(status('running'));
    followJob(client, 'j1', h);
    FakeEventSource.live.drop();
    await vi.advanceTimersByTimeAsync(600);
    FakeEventSource.live.connect(); // a healthy socket again
    FakeEventSource.live.drop();
    await vi.advanceTimersByTimeAsync(599);
    expect(getJob).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(getJob).toHaveBeenCalledTimes(2); // 600 ms again, not 1200
  });

  it('keeps trying when the poll itself cannot reach the engine', async () => {
    const { client, getJob } = stubClient();
    const h = handlers();
    getJob.mockRejectedValue(new TypeError('Failed to fetch'));
    followJob(client, 'j1', h);
    FakeEventSource.live.drop();
    await vi.advanceTimersByTimeAsync(600);
    expect(h.onGone).not.toHaveBeenCalled();
    expect(h.onEnd).not.toHaveBeenCalled();
    expect(FakeEventSource.instances).toHaveLength(2);
  });
});

describe('a live engine that has never heard of the job (B6)', () => {
  it('treats a 404 as the run dying with the process, not as a drop', async () => {
    const { client, getJob } = stubClient();
    const h = handlers();
    getJob.mockRejectedValue(new EngineError(404, 'not_found', 'no such job'));
    followJob(client, 'j1', h);
    FakeEventSource.live.drop();
    await vi.advanceTimersByTimeAsync(600);
    expect(h.onGone).toHaveBeenCalledTimes(1);
    expect(h.onEnd).toHaveBeenCalledTimes(1);
    expect(FakeEventSource.instances).toHaveLength(1); // no reopen against a 404
    await vi.advanceTimersByTimeAsync(60000);
    expect(FakeEventSource.instances).toHaveLength(1);
  });

  it('does not read a 500 as "gone"', async () => {
    const { client, getJob } = stubClient();
    const h = handlers();
    getJob.mockRejectedValue(new EngineError(500, 'engine_fault', 'boom'));
    followJob(client, 'j1', h);
    FakeEventSource.live.drop();
    await vi.advanceTimersByTimeAsync(600);
    expect(h.onGone).not.toHaveBeenCalled();
    expect(FakeEventSource.instances).toHaveLength(2);
  });
});

describe('the disposer', () => {
  it('closes the socket and cancels a pending retry', async () => {
    const { client, getJob } = stubClient();
    const h = handlers();
    getJob.mockResolvedValue(status('running'));
    const stop = followJob(client, 'j1', h);
    FakeEventSource.live.drop();
    stop();
    await vi.advanceTimersByTimeAsync(30000);
    expect(getJob).not.toHaveBeenCalled();
    expect(FakeEventSource.instances).toHaveLength(1);
    expect(h.onEnd).not.toHaveBeenCalled();
  });

  it('is safe to call twice and after the job has ended', () => {
    const { client } = stubClient();
    const h = handlers();
    const stop = followJob(client, 'j1', h);
    FakeEventSource.live.send('status', JSON.stringify(status('failed')));
    expect(() => {
      stop();
      stop();
    }).not.toThrow();
    expect(h.onEnd).toHaveBeenCalledTimes(1);
  });

  it('leaves onEnd unfired when the caller walks away mid-run', () => {
    const { client } = stubClient();
    const h = handlers();
    const stop = followJob(client, 'j1', h);
    FakeEventSource.live.connect();
    stop();
    expect(FakeEventSource.live.closed).toBe(true);
    expect(h.onEnd).not.toHaveBeenCalled();
  });

  it('works without the optional handlers', () => {
    const { client } = stubClient();
    const onStatus = vi.fn();
    const onEnd = vi.fn();
    const stop = followJob(client, 'j1', { onStatus, onEnd });
    expect(() => {
      FakeEventSource.live.connect();
      FakeEventSource.live.drop();
    }).not.toThrow();
    stop();
  });
});
