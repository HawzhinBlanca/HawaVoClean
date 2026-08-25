// state/actions.ts — the async flows that tie the engine, the store and the
// player together. These are the paths that were previously only ever proved
// by driving a real browser against a real engine: the job lifecycle, the
// B6 reconciliation of a run whose engine went away, and the B5 promise that
// re-opening a finished run costs no /api/analyze call at all.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type {
  AudioAnalysis,
  HawaVoCleanReport,
  JobState,
  JobStatus,
  UnitDecisionRecord,
} from '../api/types';

// --- the world around actions.ts -------------------------------------------

const player = vi.hoisted(() => ({
  time: 0,
  // The deck the player says is audible. `actions` mirrors this onto the store
  // whenever a deck falls out of service, so the A/B control cannot drift.
  activeDeck: 'original' as 'original' | 'cleaned',
  pause: vi.fn(),
  load: vi.fn(),
  // "You may keep your audio only if it is already this file" — the labelled
  // half of the atomic-or-labelled run swap. See the A7 swap tests below.
  claimOnly: vi.fn(),
  setActive: vi.fn(),
  seek: vi.fn(),
  toggle: vi.fn(),
  hasDeck: vi.fn((_deck: 'original' | 'cleaned') => true),
  deckFault: vi.fn((_deck: 'original' | 'cleaned') => null as { kind: string } | null),
  onFault: vi.fn((_fn: (f: unknown) => void) => () => undefined),
  // How the player asks whether the engine is answering — the fact that tells
  // "the engine died mid-load" apart from "these bytes are junk", both of
  // which Chromium reports as MediaError.code 4.
  setLivenessProbe: vi.fn((_fn: (() => boolean) | null) => undefined),
}));
vi.mock('../audio/player', () => ({ getPlayer: () => player }));

vi.mock('../bridge', () => ({
  getBridge: () => ({
    host: 'web',
    engine: {
      getEndpoint: async () => ({ baseUrl: 'http://127.0.0.1:8765', token: 'devtok' }),
    },
    files: {
      pickAudio: async () => null,
      pathForFile: () => null,
      revealInFinder: async () => undefined,
    },
  }),
}));

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

// --- fixtures ---------------------------------------------------------------

function analysis(over: Partial<AudioAnalysis> = {}): AudioAnalysis {
  return {
    path: '/a.wav',
    duration_s: 100,
    sample_rate: 48000,
    channels: 1,
    peaks: { min: [-1], max: [1] },
    rms_db: [-20],
    spectrum: { freqs_hz: [100], db: [-30] },
    loudness: { integrated_lufs: -24.9, true_peak_dbtp: -1.2 },
    noise_floor_db: -48.5,
    ...over,
  };
}

function unit(id: number): UnitDecisionRecord {
  return {
    unit_id: id,
    channel: 0,
    start_sample: 0,
    end_sample: 1,
    start_time_s: id,
    end_time_s: id + 1,
    is_speech: true,
    input_sha256: 'i',
    output_sha256: 'o',
    guard_a_verdict: 'pass',
    final_decision: 'enhanced',
  };
}

function report(): HawaVoCleanReport {
  return {
    job_id: 'j1',
    config_hash: 'c',
    input: { path: '/a.wav', sha256: 'x', sample_rate: 48000, channels: 1, samples: 1, duration_s: 100 },
    output: { path: '/out/a.wav', sha256: 'y', sample_rate: 48000, channels: 1, samples: 1, duration_s: 100 },
    core: { id: 'studio', algorithm: 'x', params_hash: 'p' },
    guard: { id: 'g', probe_hash: 'p', calibration_id: 'c' },
    environment: {},
    summary: { units_total: 5, enhanced: 5 },
    units: [unit(0), unit(1)],
  };
}

function jobStatus(state: JobState, over: Partial<JobStatus> = {}): JobStatus {
  return {
    job_id: 'j1',
    state,
    stage: state === 'done' ? 'done' : 'enhance',
    progress: state === 'done' ? 1 : 0.5,
    message: state === 'done' ? 'finished' : 'enhancing',
    input_path: '/a.wav',
    output_path: '/out/a.wav',
    report_path: '/out/a.hawavoclean.json',
    profile: 'studio',
    started_at: '2026-08-20T10:00:00Z',
    finished_at: state === 'done' ? '2026-08-20T10:00:12Z' : null,
    ...(state === 'done' ? { report: report() } : {}),
    ...over,
  };
}

interface FakeClient {
  health: ReturnType<typeof vi.fn>;
  analyze: ReturnType<typeof vi.fn>;
  /** Assigned per-test by the upload cases; unused by the rest. */
  uploadWithProgress: ReturnType<typeof vi.fn>;
  createJob: ReturnType<typeof vi.fn>;
  getJob: ReturnType<typeof vi.fn>;
  cancelJob: ReturnType<typeof vi.fn>;
  peaks: ReturnType<typeof vi.fn>;
  verify: ReturnType<typeof vi.fn>;
  fileUrl: (p: string) => string;
  eventsUrl: (id: string) => string;
}

function makeClient(): FakeClient {
  return {
    health: vi.fn(async () => ({
      ok: true,
      version: '3.2.0',
      profiles: ['studio', 'production'],
      engine_pid: 4242,
    })),
    analyze: vi.fn(async (path: string) => analysis({ path })),
    createJob: vi.fn(async () => ({
      job_id: 'j1',
      output_path: '/out/a.wav',
      report_path: '/out/a.hawavoclean.json',
    })),
    getJob: vi.fn(),
    cancelJob: vi.fn(async () => ({ ok: true })),
    uploadWithProgress: vi.fn(),
    peaks: vi.fn(),
    // One ranged byte of `/api/audio` — the artefact verification a restore
    // makes. Everything is there, readable, 1000 B long unless a test says
    // otherwise.
    verify: vi.fn(async (_path: string) => ({ status: 206, delivered: true, size: 1000 })),
    fileUrl: (p: string) => `http://127.0.0.1:8765/api/audio?path=${encodeURIComponent(p)}`,
    eventsUrl: (id: string) => `http://127.0.0.1:8765/api/jobs/${id}/events?token=devtok`,
  };
}

type ActionsMod = typeof import('./actions');
type StoreMod = typeof import('./store');
type ClientMod = typeof import('../api/client');

interface Booted {
  actions: ActionsMod;
  store: StoreMod;
  EngineError: ClientMod['EngineError'];
  client: FakeClient;
}

/**
 * A fresh module graph per test. `actions.ts` keeps real module state (the
 * pid it last saw, the probe step, the live stream's disposer), so sharing it
 * between tests would be sharing a session.
 */
async function boot(): Promise<Booted> {
  vi.resetModules();
  FakeEventSource.instances = [];
  const store = await import('./store');
  const actions = await import('./actions');
  const { EngineError } = await import('../api/client');
  const client = makeClient();
  store.useStore.getState().setEngine('ready', client as never, '3.2.0');
  return { actions, store, EngineError, client };
}

/** Let promise chains and timers run. */
async function settle(ms = 0): Promise<void> {
  await vi.advanceTimersByTimeAsync(ms);
}

function armed(store: StoreMod): void {
  const st = store.useStore.getState();
  st.setSource({ path: '/a.wav', name: 'a.wav', origin: 'file' });
  st.setOriginal(analysis());
}

beforeEach(() => {
  vi.stubGlobal('EventSource', FakeEventSource);
  vi.useFakeTimers();
  vi.setSystemTime(new Date('2026-08-20T10:00:20Z'));
  player.time = 0;
  player.activeDeck = 'original';
  player.hasDeck.mockReturnValue(true);
  player.deckFault.mockReturnValue(null);
  player.onFault.mockClear();
});

afterEach(() => {
  vi.clearAllTimers();
  vi.useRealTimers();
});

// ---------------------------------------------------------------------------

describe('startJob', () => {
  it('posts the run, opens its stream, and clears the previous result', async () => {
    const { actions, store, client } = await boot();
    armed(store);
    store.useStore.getState().setCleaned(analysis(), '/out/old.wav');
    store.useStore.getState().setReport(report());
    await actions.startJob();
    expect(client.createJob).toHaveBeenCalledWith({
      input_path: '/a.wav',
      profile: 'studio',
      overwrite: true,
    });
    const st = store.useStore.getState();
    expect(st.job).toMatchObject({ id: 'j1', outputPath: '/out/a.wav', streamConnected: false });
    expect(st.cleaned).toBeNull();
    expect(st.cleanedPath).toBeNull();
    expect(st.report).toBeNull();
    expect(st.abMode).toBe('original');
    expect(st.statusLine).toBe('Job queued');
    expect(FakeEventSource.instances).toHaveLength(1);
    expect(FakeEventSource.live.url).toContain('/api/jobs/j1/events');
  });

  it('refuses to start a second run on top of a live one', async () => {
    const { actions, store, client } = await boot();
    armed(store);
    await actions.startJob();
    FakeEventSource.live.send('status', JSON.stringify(jobStatus('running')));
    await actions.startJob();
    expect(client.createJob).toHaveBeenCalledTimes(1);
  });

  it('does nothing without a source', async () => {
    const { actions, client } = await boot();
    await actions.startJob();
    expect(client.createJob).not.toHaveBeenCalled();
  });

  it('turns a refusal into a sentence and leaves no half-made job', async () => {
    const { actions, store, client, EngineError } = await boot();
    armed(store);
    client.createJob.mockRejectedValue(
      new EngineError(400, 'INVALID_USER_INPUT', 'Input sample rate 192000 Hz exceeds maximum supported 48000 Hz'),
    );
    await actions.startJob();
    const st = store.useStore.getState();
    expect(st.job).toBeNull();
    expect(st.error).toBeTruthy();
    expect(st.error).not.toContain('INVALID_USER_INPUT');
    expect(st.statusLine).toContain('Could not start');
  });
});

describe('the job status stream', () => {
  it('narrates progress with the unit counter', async () => {
    const { actions, store } = await boot();
    armed(store);
    await actions.startJob();
    FakeEventSource.live.connect();
    expect(store.useStore.getState().job?.streamConnected).toBe(true);
    FakeEventSource.live.send(
      'status',
      JSON.stringify(jobStatus('running', { message: 'enhancing', unit: { index: 3, total: 5 } })),
    );
    expect(store.useStore.getState().statusLine).toBe('enhancing (3/5)');
  });

  it('a finished run lands the report, the cleaned deck and a history entry', async () => {
    const { actions, store, client } = await boot();
    armed(store);
    await actions.startJob();
    client.analyze.mockClear();
    FakeEventSource.live.send('status', JSON.stringify(jobStatus('done')));
    await settle();

    const st = store.useStore.getState();
    expect(st.report?.summary.enhanced).toBe(5);
    expect(st.cleanedPath).toBe('/out/a.wav');
    expect(st.abMode).toBe('cleaned');
    // The length the run says this master holds rides with the load: a deck
    // that opens far shorter than that is a truncated file, not a deck.
    expect(player.load).toHaveBeenCalledWith('cleaned', client.fileUrl('/out/a.wav'), 100);
    expect(player.setActive).toHaveBeenCalledWith('cleaned');
    expect(client.analyze).toHaveBeenCalledTimes(1); // the cleaned master only

    expect(st.history).toHaveLength(1);
    expect(st.currentRunId).toBe('j1');
    expect(st.history[0]).toMatchObject({
      jobId: 'j1',
      outcome: 'done',
      profile: 'studio',
      inputName: 'a.wav',
      enhanced: 5,
      unitsTotal: 5,
      lufsIn: -24.9,
      durationMs: 12000, // from the engine's own timestamps
      outputPath: '/out/a.wav',
    });
    expect(st.history[0]?.cleaned).not.toBeNull();
    expect(st.statusLine).toContain('5/5 units enhanced');
    // A7 · the master's served length is recorded the moment the run lands —
    // it is the yardstick every later re-verification measures against.
    expect(st.history[0]?.masterBytes).toBe(1000);
  });

  it('a failed run is re-worded, not quoted, and is recorded as failed', async () => {
    const { actions, store } = await boot();
    armed(store);
    await actions.startJob();
    FakeEventSource.live.send(
      'status',
      JSON.stringify(
        jobStatus('failed', {
          error: {
            code: 'INVALID_USER_INPUT',
            message:
              "ffprobe failed to probe /Users/me/work/uploads/6f2/take.wav: Command '['/opt/homebrew/bin/ffprobe', '-v', 'error']' returned non-zero exit status 1.",
          },
        }),
      ),
    );
    await settle();
    const st = store.useStore.getState();
    expect(st.error).toBeTruthy();
    expect(st.error).not.toContain('Command ');
    expect(st.error).not.toContain('/Users/me/work/uploads');
    expect(st.statusLine).toContain('Processing failed');
    expect(st.history[0]).toMatchObject({ jobId: 'j1', outcome: 'failed' });
    expect(st.history[0]?.error).toBeTruthy();
    expect(st.history[0]?.error).not.toContain('Command ');
  });

  it('a cancelled run is recorded without an error bar', async () => {
    const { actions, store } = await boot();
    armed(store);
    await actions.startJob();
    FakeEventSource.live.send('status', JSON.stringify(jobStatus('cancelled')));
    await settle();
    const st = store.useStore.getState();
    expect(st.statusLine).toBe('Processing cancelled');
    expect(st.history[0]?.outcome).toBe('cancelled');
    expect(st.error).toBeNull();
  });

  it('ignores a status for a job that is no longer the current one', async () => {
    const { actions, store } = await boot();
    armed(store);
    await actions.startJob();
    const es = FakeEventSource.live;
    store.useStore.getState().setJob({
      id: 'j9',
      outputPath: '/out/b.wav',
      reportPath: '/out/b.json',
      status: null,
      streamConnected: false,
    });
    es.send('status', JSON.stringify(jobStatus('done')));
    await settle();
    expect(store.useStore.getState().history).toHaveLength(0);
    expect(store.useStore.getState().report).toBeNull();
  });
});

describe('B6 · a run whose engine went away (ENGINE_RESTARTED)', () => {
  it('a live engine that 404s the job ends the run instead of hanging on "running"', async () => {
    const { actions, store, client, EngineError } = await boot();
    armed(store);
    await actions.startJob();
    FakeEventSource.live.connect();
    FakeEventSource.live.send('status', JSON.stringify(jobStatus('running')));
    client.getJob.mockRejectedValue(new EngineError(404, 'not_found', 'no such job'));

    FakeEventSource.live.drop();
    await settle(700); // past the first back-off

    const st = store.useStore.getState();
    expect(st.job?.status?.state).toBe('failed');
    expect(st.job?.status?.error?.code).toBe('ENGINE_RESTARTED');
    expect(st.job?.status?.stage).toBe('error');
    expect(st.job?.status?.finished_at).not.toBeNull();
    expect(st.history[0]).toMatchObject({ jobId: 'j1', outcome: 'failed' });
    // And it stays ended: nothing reopens against a job that is gone.
    await settle(30000);
    expect(FakeEventSource.instances).toHaveLength(1);
  });

  it('keeps the progress the run had reached in its terminal snapshot', async () => {
    const { actions, store, client, EngineError } = await boot();
    armed(store);
    await actions.startJob();
    FakeEventSource.live.send(
      'status',
      JSON.stringify(jobStatus('running', { progress: 0.62, unit: { index: 4, total: 6 } })),
    );
    client.getJob.mockRejectedValue(new EngineError(404, 'not_found', 'gone'));
    FakeEventSource.live.drop();
    await settle(700);
    const status = store.useStore.getState().job?.status;
    expect(status?.progress).toBe(0.62);
    expect(status?.unit).toEqual({ index: 4, total: 6 });
    expect(status?.error?.message).toContain('press PROCESS to run it again');
  });

  it('reconciles through the health probe when the engine comes back without the job', async () => {
    const { actions, store, client, EngineError } = await boot();
    armed(store);
    await actions.startJob();
    FakeEventSource.live.send('status', JSON.stringify(jobStatus('running')));

    // The engine dies; the app notices and stops the stream.
    store.useStore.getState().setEngine('offline', client as never, '3.2.0');
    client.getJob.mockRejectedValue(new EngineError(404, 'not_found', 'no such job'));

    actions.retryEngineNow();
    await settle(10);

    const st = store.useStore.getState();
    expect(st.engineStatus).toBe('ready');
    // With a timeout signal: this call runs on reconnect, i.e. against an
    // engine that has just misbehaved, and an un-timed-out request here is the
    // second door into the health-probe deadlock.
    expect(client.getJob).toHaveBeenCalledWith('j1', expect.any(AbortSignal));
    expect(st.job?.status?.state).toBe('failed');
    expect(st.job?.status?.error?.code).toBe('ENGINE_RESTARTED');
    expect(st.history[0]?.outcome).toBe('failed');
  });

  it('a run that finished while we were offline lands as done, not as lost', async () => {
    const { actions, store, client } = await boot();
    armed(store);
    await actions.startJob();
    FakeEventSource.live.send('status', JSON.stringify(jobStatus('running')));
    store.useStore.getState().setEngine('offline', client as never, '3.2.0');
    client.getJob.mockResolvedValue(jobStatus('done'));

    actions.retryEngineNow();
    await settle(10);

    const st = store.useStore.getState();
    expect(st.job?.status?.state).toBe('done');
    expect(st.report?.summary.enhanced).toBe(5);
    expect(st.history[0]).toMatchObject({ jobId: 'j1', outcome: 'done' });
  });

  it('re-attaches the stream to a run that is genuinely still going', async () => {
    const { actions, store, client } = await boot();
    armed(store);
    await actions.startJob();
    store.useStore.getState().setEngine('offline', client as never, '3.2.0');
    client.getJob.mockResolvedValue(jobStatus('running'));

    actions.retryEngineNow();
    await settle(10);

    expect(store.useStore.getState().job?.status?.state).toBe('running');
    expect(FakeEventSource.instances.length).toBeGreaterThanOrEqual(2);
  });

  it('leaves a run that already ended alone when the engine returns', async () => {
    const { actions, store, client } = await boot();
    armed(store);
    await actions.startJob();
    FakeEventSource.live.send('status', JSON.stringify(jobStatus('done')));
    await settle();
    client.getJob.mockClear();
    store.useStore.getState().setEngine('offline', client as never, '3.2.0');

    actions.retryEngineNow();
    await settle(10);

    expect(client.getJob).not.toHaveBeenCalled();
    expect(store.useStore.getState().job?.status?.state).toBe('done');
  });
});

describe('B5 · re-opening a finished run', () => {
  function finished(store: StoreMod, jobId: string, over: Record<string, unknown> = {}): void {
    store.useStore.getState().pushHistory({
      jobId,
      profile: 'studio',
      inputPath: '/a.wav',
      inputName: 'a.wav',
      outcome: 'done',
      durationMs: 12000,
      at: Date.now(),
      enhanced: 5,
      unitsTotal: 5,
      lufsIn: -24.9,
      lufsOut: -21.7,
      noiseIn: -48.5,
      noiseOut: -84.7,
      outputPath: `/out/${jobId}.wav`,
      reportPath: `/out/${jobId}.hawavoclean.json`,
      report: report(),
      status: jobStatus('done'),
      original: analysis(),
      cleaned: analysis({ path: `/out/${jobId}.wav` }),
      error: null,
      ...over,
    });
  }

  it('restores everything from memory: ZERO /api/analyze calls', async () => {
    const { actions, store, client } = await boot();
    finished(store, 'j1');
    finished(store, 'j2');
    client.analyze.mockClear();
    player.load.mockClear();

    await actions.selectRun('j1');
    await settle();

    const st = store.useStore.getState();
    expect(client.analyze).not.toHaveBeenCalled();
    expect(st.currentRunId).toBe('j1');
    expect(st.source?.name).toBe('a.wav');
    expect(st.job?.id).toBe('j1');
    expect(st.report?.summary.enhanced).toBe(5);
    expect(st.original).not.toBeNull();
    expect(st.cleaned).not.toBeNull();
    expect(st.cleanedPath).toBe('/out/j1.wav');
    expect(st.abMode).toBe('cleaned');
    expect(player.load).toHaveBeenCalledWith('original', client.fileUrl('/a.wav'), 100);
    expect(player.load).toHaveBeenCalledWith('cleaned', client.fileUrl('/out/j1.wav'), 100);
    expect(st.analyzing).toBe(false);
    expect(st.statusLine).toContain('units enhanced');
  });

  it('does not re-run the restore for the run already on screen', async () => {
    const { actions, store, client } = await boot();
    finished(store, 'j1');
    player.pause.mockClear();
    // Not a no-op any more — it re-verifies the artefacts (three HEADs, see
    // the B5 block below) — but nothing of the restore itself happens again.
    await actions.selectRun('j1'); // pushHistory already made it current
    expect(player.pause).not.toHaveBeenCalled();
    expect(client.analyze).not.toHaveBeenCalled();
  });

  it('re-reads only the half it does not have', async () => {
    const { actions, store, client } = await boot();
    finished(store, 'j1', { cleaned: null });
    finished(store, 'j2');
    client.analyze.mockClear();

    await actions.selectRun('j1');
    await settle();

    expect(client.analyze).toHaveBeenCalledTimes(1);
    expect(client.analyze.mock.calls[0]?.[0]).toBe('/out/j1.wav');
    const st = store.useStore.getState();
    expect(st.analyzing).toBe(false);
    expect(st.history.find((h) => h.jobId === 'j1')?.cleaned).not.toBeNull();
  });

  it('says so when the file behind an old run has gone', async () => {
    const { actions, store, client, EngineError } = await boot();
    finished(store, 'j1', { original: null });
    finished(store, 'j2');
    client.analyze.mockRejectedValue(new EngineError(404, 'not_found', '/a.wav does not exist'));

    await actions.selectRun('j1');
    await settle();

    const st = store.useStore.getState();
    expect(st.error).toBeTruthy();
    expect(st.statusLine).toContain('Could not re-read this run');
    expect(st.analyzing).toBe(false);
  });

  it('refuses while a run is still going, and says why', async () => {
    const { actions, store, client } = await boot();
    armed(store);
    await actions.startJob();
    FakeEventSource.live.send('status', JSON.stringify(jobStatus('running')));
    finished(store, 'j5');
    store.useStore.getState().setCurrentRun(null);
    client.analyze.mockClear();

    await actions.selectRun('j5');

    expect(store.useStore.getState().error).toContain('still running');
    expect(store.useStore.getState().currentRunId).toBeNull();
  });

  it('opens a failed run with no cleaned deck', async () => {
    const { actions, store } = await boot();
    finished(store, 'j1', {
      outcome: 'failed',
      cleaned: null,
      report: null,
      error: 'The run failed',
    });
    finished(store, 'j2');
    await actions.selectRun('j1');
    await settle();
    const st = store.useStore.getState();
    expect(st.cleaned).toBeNull();
    expect(st.cleanedPath).toBeNull();
    expect(st.abMode).toBe('original');
    expect(st.statusLine).toContain('Failed');
  });

  it('does nothing for a run the session never had', async () => {
    const { actions, store } = await boot();
    await actions.selectRun('nope');
    expect(store.useStore.getState().currentRunId).toBeNull();
  });
});

describe('cancelJob', () => {
  it('asks the engine to stop the run', async () => {
    const { actions, store, client } = await boot();
    armed(store);
    await actions.startJob();
    await actions.cancelJob();
    expect(client.cancelJob).toHaveBeenCalledWith('j1');
    expect(store.useStore.getState().statusLine).toBe('Cancelling');
  });

  it('does not pretend to cancel while the engine is gone', async () => {
    const { actions, store, client } = await boot();
    armed(store);
    await actions.startJob();
    store.useStore.getState().setEngine('offline', client as never, '3.2.0');
    client.cancelJob.mockClear();
    await actions.cancelJob();
    expect(client.cancelJob).not.toHaveBeenCalled();
    expect(store.useStore.getState().statusLine).toContain('offline');
  });
});

describe('report access and helpers (B7)', () => {
  it('summaryLine drops the clauses whose numbers are missing', async () => {
    const { actions } = await boot();
    const base = {
      jobId: 'j1',
      profile: 'studio' as const,
      inputPath: '/a.wav',
      inputName: 'Flute 09.m4a.mp4',
      outcome: 'done' as const,
      durationMs: 1,
      at: 0,
      enhanced: 5,
      unitsTotal: 5,
      lufsIn: -24.9,
      lufsOut: -21.7,
      noiseIn: -48.5,
      noiseOut: -84.7,
      outputPath: '/out/a.wav',
      reportPath: '/out/a.json',
      report: null,
      status: null,
      original: null,
      cleaned: null,
      error: null,
    };
    expect(actions.summaryLine(base)).toBe(
      'Flute 09.m4a · studio · 5/5 units enhanced · -24.9 -> -21.7 LUFS · noise floor -48.5 -> -84.7 dB',
    );
    expect(actions.summaryLine({ ...base, lufsOut: null, noiseOut: null })).toBe(
      'Flute 09.m4a · studio · 5/5 units enhanced',
    );
    expect(actions.summaryLine({ ...base, outcome: 'failed', error: 'the file has no audio' })).toBe(
      'Flute 09.m4a · studio · FAILED (the file has no audio)',
    );
    expect(actions.summaryLine({ ...base, outcome: 'cancelled' })).toBe(
      'Flute 09.m4a · studio · CANCELLED',
    );
  });

  it('artifactsFor names all three files a run leaves behind', async () => {
    const { actions } = await boot();
    const art = actions.artifactsFor('/out/a.wav', '/out/a.hawavoclean.json');
    expect(art?.master.name).toBe('a.wav');
    expect(art?.json.name).toBe('a.hawavoclean.json');
    expect(art?.txt.name).toBe('a.hawavoclean.txt');
    expect(art?.txt.url).toContain('a.hawavoclean.txt');
    expect(actions.artifactsFor(null, '/out/a.json')).toBeNull();
    expect(actions.artifactsFor('/out/a.wav', null)).toBeNull();
  });

  it('rateWarning flags only the rates the pipeline will refuse', async () => {
    const { actions } = await boot();
    expect(actions.rateWarning(48000)).toBeNull();
    expect(actions.rateWarning(44100)).toBeNull();
    expect(actions.rateWarning(8000)).toBeNull();
    expect(actions.rateWarning(192000)).toContain('192 kHz');
    expect(actions.rateWarning(192000)).toContain('48 kHz');
    expect(actions.rateWarning(4000)).toContain('4 kHz');
    expect(actions.rateWarning(0)).toBeNull();
    expect(actions.rateWarning(Number.NaN)).toBeNull();
  });

  it('baseName survives both separators and a bare name', async () => {
    const { actions } = await boot();
    expect(actions.baseName('/Users/me/Flute 09.m4a.mp4')).toBe('Flute 09.m4a.mp4');
    expect(actions.baseName('C:\\takes\\a.wav')).toBe('a.wav');
    expect(actions.baseName('a.wav')).toBe('a.wav');
  });

  it('isAcceptedFile takes audio and video containers and refuses the rest', async () => {
    const { actions } = await boot();
    const f = (name: string, type = ''): File => new File(['x'], name, { type });
    expect(actions.isAcceptedFile(f('take.wav', 'audio/wav'))).toBe(true);
    expect(actions.isAcceptedFile(f('take.MP4'))).toBe(true);
    expect(actions.isAcceptedFile(f('notes.txt', 'text/plain'))).toBe(false);
    expect(actions.extensionOf('Flute 09.m4a.mp4')).toBe('mp4');
    expect(actions.extensionOf('noext')).toBe('');
  });
});

// B2 · what a drop is allowed to do. Iteration 3 of the web perfection log
// claims three designed refusals here — a folder, a drop with nothing openable,
// and a multi-file drop that takes the first audio file "and says so". None of
// them was pinned; all three are early returns that never reach `ingestFile`,
// so they are cheap to hold. The *kind* is asserted rather than the sentence,
// so rewording the copy stays free while losing the refusal does not.
describe('B2 · drop routing', () => {
  function dt(
    files: File[],
    dirs: string[] = [],
  ): DataTransfer {
    const items = [
      ...dirs.map((name) => ({
        kind: 'file' as const,
        webkitGetAsEntry: () => ({ isDirectory: true, name }),
      })),
      ...files.map(() => ({
        kind: 'file' as const,
        webkitGetAsEntry: () => ({ isDirectory: false, name: 'f' }),
      })),
    ];
    return { items, files } as unknown as DataTransfer;
  }
  const wav = (name = 'take.wav'): File => new File(['x'], name, { type: 'audio/wav' });

  it('refuses a dropped folder without trying to open it', async () => {
    const { actions, store, client } = await boot();
    await actions.ingestDataTransfer(dt([], ['Takes']));
    const r = store.useStore.getState().rejection;
    expect(r?.kind).toBe('folder');
    expect(r?.name).toBe('Takes');
    expect(client.analyze).not.toHaveBeenCalled();
  });

  it('says how many folders when several are dropped at once', async () => {
    const { actions, store } = await boot();
    await actions.ingestDataTransfer(dt([], ['A', 'B', 'C']));
    expect(store.useStore.getState().rejection?.detail).toContain('3 folders');
  });

  it('an empty drop is a designed refusal, not a silent nothing', async () => {
    const { actions, store } = await boot();
    await actions.ingestDataTransfer(dt([]));
    expect(store.useStore.getState().rejection?.kind).toBe('empty');
  });

  it('a drop with nothing openable names the type it refused', async () => {
    const { actions, store, client } = await boot();
    await actions.ingestDataTransfer(dt([new File(['x'], 'notes.txt', { type: 'text/plain' })]));
    const r = store.useStore.getState().rejection;
    expect(r?.kind).toBe('type');
    expect(r?.name).toBe('notes.txt');
    expect(r?.detail).toContain('text/plain');
    expect(client.analyze).not.toHaveBeenCalled();
  });

  it('a multi-file drop takes the first audio file and says which one', async () => {
    const { actions, store } = await boot();
    await actions.ingestDataTransfer(
      dt([new File(['x'], 'notes.txt', { type: 'text/plain' }), wav('second.wav')]),
    );
    await settle(0);
    const r = store.useStore.getState().rejection;
    expect(r?.kind).toBe('multi');
    expect(r?.name).toBe('second.wav');
  });

  it('a clean single-file drop raises no rejection at all', async () => {
    const { actions, store } = await boot();
    store.useStore.getState().setRejection({ kind: 'type', name: 'old', detail: 'stale' });
    await actions.ingestDataTransfer(dt([wav()]));
    await settle(0);
    expect(store.useStore.getState().rejection).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// The three P0 correctness defects the adversarial audit found. Each one made
// the UI state something untrue, so each one gets a test that fails if the lie
// comes back.

/** A finished run, as `recordRun` would have left it. */
function done(
  store: StoreMod,
  over: {
    jobId: string;
    inputPath?: string;
    profile?: 'studio' | 'production';
    outputPath: string;
    masterBytes?: number;
  },
): void {
  const outputPath = over.outputPath;
  store.useStore.getState().pushHistory({
    ...(over.masterBytes !== undefined ? { masterBytes: over.masterBytes } : {}),
    jobId: over.jobId,
    profile: over.profile ?? 'studio',
    inputPath: over.inputPath ?? '/a.wav',
    inputName: 'a.wav',
    outcome: 'done',
    durationMs: 8000,
    at: Date.now(),
    enhanced: 5,
    unitsTotal: 5,
    lufsIn: -24.9,
    lufsOut: -21.7,
    noiseIn: -48.5,
    noiseOut: -84.7,
    outputPath,
    reportPath: outputPath.replace(/\.wav$/, '.hawavoclean.json'),
    report: report(),
    status: jobStatus('done'),
    original: analysis(),
    cleaned: analysis({ path: outputPath }),
    error: null,
  });
}

describe('A7/B5 · the A/B control cannot claim a deck the player is not on', () => {
  /** Hand `actions` the fault listener the player would have called. */
  async function faultListener(actions: ActionsMod): Promise<(f: unknown) => void> {
    await actions.connectEngine();
    await settle();
    const fn = player.onFault.mock.calls[0]?.[0];
    if (!fn) throw new Error('actions never subscribed to deck faults');
    return fn;
  }

  it('mirrors the player’s fallback onto the store, and says what happened', async () => {
    const { actions, store, client } = await boot();
    const fire = await faultListener(actions);
    const st = store.useStore.getState();
    st.setSource({ path: '/a.wav', name: 'a.wav', origin: 'file' });
    st.setCleaned(analysis(), '/out/a.wav');
    st.setAbMode('cleaned');
    // The player has already fallen back — that is what it is reporting.
    player.activeDeck = 'original';

    fire({
      deck: 'cleaned',
      url: client.fileUrl('/out/a.wav'),
      kind: 'missing',
      status: 404,
      fellBackTo: 'original',
    });

    const after = store.useStore.getState();
    expect(after.abMode).toBe('original');
    expect(after.deckFault?.deck).toBe('cleaned');
    expect(after.deckFault?.detail).toContain('cleaned master could not be loaded');
    expect(after.deckFault?.detail).toContain('playing the original');
    // The same fact reaches the artefact row: a master that cannot be played
    // is a master that cannot be downloaded.
    expect(after.artifacts?.master).toBe(false);
    // …and the analysis goes with it, so nothing keeps drawing a cleaned deck
    // or asking /api/peaks for a path that answers 404.
    expect(after.cleaned).toBeNull();
    expect(after.cleanedPath).toBe('/out/a.wav');
  });

  it('does not blame the current clip for a fault about a file that has left the screen', async () => {
    const { actions, store, client } = await boot();
    const fire = await faultListener(actions);
    const st = store.useStore.getState();
    st.setCleaned(analysis(), '/out/current.wav');
    st.setAbMode('cleaned');
    player.activeDeck = 'original';

    fire({
      deck: 'cleaned',
      url: client.fileUrl('/out/stale.wav'),
      kind: 'missing',
      status: 404,
      fellBackTo: 'original',
    });

    const after = store.useStore.getState();
    // The switch still follows the player — that is always the truth …
    expect(after.abMode).toBe('original');
    // … but nothing is said about a clip this is not about.
    expect(after.deckFault).toBeNull();
    expect(after.cleaned).not.toBeNull();
  });

  it('an engine that could not be reached is not a file that was deleted', async () => {
    const { actions, store, client } = await boot();
    const fire = await faultListener(actions);
    const st = store.useStore.getState();
    st.setCleaned(analysis(), '/out/a.wav');
    player.activeDeck = 'original';

    fire({
      deck: 'cleaned',
      url: client.fileUrl('/out/a.wav'),
      kind: 'network',
      status: null,
      fellBackTo: 'original',
    });

    const after = store.useStore.getState();
    expect(after.deckFault?.detail).toContain('engine could not be reached');
    // Nothing is condemned: the file is still on disk, the engine is not there.
    expect(after.artifacts).toBeNull();
    expect(after.cleaned).not.toBeNull();
  });
});

describe('B5 · restoring a run whose files are gone tells the truth', () => {
  it('checks the artefacts on restore — one ranged byte each, never /api/analyze', async () => {
    const { actions, store, client } = await boot();
    done(store, { jobId: 'j1', outputPath: '/out/j1.wav' });
    done(store, { jobId: 'j2', outputPath: '/out/j2.wav' });
    client.analyze.mockClear();
    client.verify.mockClear();

    await actions.selectRun('j1');
    await settle();

    expect(client.analyze).not.toHaveBeenCalled();
    expect(client.verify.mock.calls.map((c) => c[0]).sort()).toEqual([
      '/out/j1.hawavoclean.json',
      '/out/j1.hawavoclean.txt',
      '/out/j1.wav',
    ]);
    const st = store.useStore.getState();
    expect(st.artifacts).toEqual({ master: true, json: true, txt: true, reason: '' });
    expect(st.abMode).toBe('cleaned');
    expect(st.deckFault).toBeNull();
  });

  it('a run whose master was deleted does not restore as a playable run', async () => {
    const { actions, store, client } = await boot();
    done(store, { jobId: 'j1', outputPath: '/out/j1.wav' });
    done(store, { jobId: 'j2', outputPath: '/out/j2.wav' });
    client.analyze.mockClear();
    player.load.mockClear();
    client.verify.mockImplementation(async (p: string) =>
      p === '/out/j1.wav'
        ? { status: 404, delivered: false, size: null }
        : { status: 206, delivered: true, size: 1000 },
    );

    await actions.selectRun('j1');
    await settle();

    const st = store.useStore.getState();
    // The restore itself is still free.
    expect(client.analyze).not.toHaveBeenCalled();
    // No deck is offered that cannot be played …
    expect(st.abMode).toBe('original');
    expect(st.cleaned).toBeNull();
    expect(player.load).toHaveBeenCalledWith('cleaned', null);
    // … the artefact row knows which file is gone …
    expect(st.artifacts?.master).toBe(false);
    expect(st.artifacts?.json).toBe(true);
    expect(st.artifacts?.reason).toContain('cleaned master');
    // … the answer is on the run's own row, so the list stops offering it …
    expect(st.history.find((h) => h.jobId === 'j1')?.artifacts?.master).toBe(false);
    // … and the screen says so instead of reporting a complete run.
    expect(st.deckFault?.deck).toBe('cleaned');
    expect(st.statusLine).toContain('no longer on disk');
    // The path is kept: the disabled link still has to name what it cannot give.
    expect(st.cleanedPath).toBe('/out/j1.wav');
  });

  it('artefacts that are all still there are not annotated', async () => {
    const { actions, store } = await boot();
    done(store, { jobId: 'j1', outputPath: '/out/j1.wav' });
    done(store, { jobId: 'j2', outputPath: '/out/j2.wav' });
    await actions.selectRun('j1');
    await settle();
    const a = actions.artifactsFor(
      '/out/j1.wav',
      '/out/j1.hawavoclean.json',
      store.useStore.getState().artifacts,
    );
    expect(a?.master.url).toBeTruthy();
    expect(a?.master.note).toBeNull();
  });

  it('a link whose file is gone loses its href and keeps its reason', async () => {
    const { actions, store, client } = await boot();
    done(store, { jobId: 'j1', outputPath: '/out/j1.wav' });
    done(store, { jobId: 'j2', outputPath: '/out/j2.wav' });
    client.verify.mockImplementation(async (p: string) =>
      p === '/out/j1.wav'
        ? { status: 404, delivered: false, size: null }
        : { status: 206, delivered: true, size: 1000 },
    );
    await actions.selectRun('j1');
    await settle();
    const a = actions.artifactsFor(
      '/out/j1.wav',
      '/out/j1.hawavoclean.json',
      store.useStore.getState().artifacts,
    );
    expect(a?.master.url).toBeNull();
    expect(a?.master.note).toContain('no longer on disk');
    expect(a?.master.name).toBe('j1.wav');
    expect(a?.json.url).toBeTruthy();
  });

  it('a flagged run can be opened again to look for its files once more', async () => {
    const { actions, store, client } = await boot();
    done(store, { jobId: 'j1', outputPath: '/out/j1.wav' });
    done(store, { jobId: 'j2', outputPath: '/out/j2.wav' });
    client.verify.mockImplementation(async (p: string) =>
      p === '/out/j1.wav'
        ? { status: 404, delivered: false, size: null }
        : { status: 206, delivered: true, size: 1000 },
    );
    await actions.selectRun('j1');
    await settle();
    expect(store.useStore.getState().artifacts?.master).toBe(false);

    // The file comes back — a volume remounted, an undo in Finder. Re-opening
    // the run already on screen is the gesture that means "look again", and it
    // is the only one: nothing else re-checks a run that is already showing.
    client.verify.mockImplementation(async () => ({ status: 206, delivered: true, size: 1000 }));
    await actions.selectRun('j1');
    await settle();

    const st = store.useStore.getState();
    expect(st.artifacts?.master).toBe(true);
    expect(st.abMode).toBe('cleaned');
    expect(st.deckFault).toBeNull();
    expect(client.analyze).not.toHaveBeenCalled();
  });

  it('an outage is not a deletion: nothing is condemned when the check cannot be made', async () => {
    const { actions, store, client, EngineError } = await boot();
    done(store, { jobId: 'j1', outputPath: '/out/j1.wav' });
    done(store, { jobId: 'j2', outputPath: '/out/j2.wav' });
    client.verify.mockRejectedValue(new EngineError(0, 'network', 'Engine unreachable'));

    await actions.selectRun('j1');
    await settle();

    const st = store.useStore.getState();
    expect(st.artifacts).toBeNull();
    expect(st.abMode).toBe('cleaned'); // the run restores as it always did
    expect(st.deckFault).toBeNull();
  });
});

describe('B5 · two runs of the same profile do not overwrite each other', () => {
  it('the first run of a profile keeps the engine’s own naming', async () => {
    const { actions, store, client } = await boot();
    armed(store);
    await actions.startJob();
    expect(client.createJob).toHaveBeenCalledWith({
      input_path: '/a.wav',
      profile: 'studio',
      overwrite: true,
    });
  });

  it('the second run of the same profile is given its own output path', async () => {
    const { actions, store, client } = await boot();
    armed(store);
    done(store, { jobId: 'j1', outputPath: '/dir/a_studio.wav' });
    await actions.startJob();
    expect(client.createJob).toHaveBeenCalledWith({
      input_path: '/a.wav',
      profile: 'studio',
      overwrite: true,
      output_path: '/dir/a_studio-2.wav',
    });
  });

  it('a third run counts on rather than stacking suffixes', async () => {
    const { actions, store, client } = await boot();
    armed(store);
    done(store, { jobId: 'j1', outputPath: '/dir/a_studio.wav' });
    done(store, { jobId: 'j2', outputPath: '/dir/a_studio-2.wav' });
    await actions.startJob();
    expect(client.createJob.mock.calls[0]?.[0]).toMatchObject({
      output_path: '/dir/a_studio-3.wav',
    });
  });

  it('a different profile on the same clip writes its own name, untouched', async () => {
    const { actions, store, client } = await boot();
    armed(store);
    store.useStore.getState().setProfile('production');
    done(store, { jobId: 'j1', outputPath: '/dir/a_studio.wav' });
    await actions.startJob();
    expect(client.createJob).toHaveBeenCalledWith({
      input_path: '/a.wav',
      profile: 'production',
      overwrite: true,
    });
  });

  it('a run that did not finish claims no name', async () => {
    const { actions, store, client } = await boot();
    armed(store);
    done(store, { jobId: 'j1', outputPath: '/dir/a_studio.wav' });
    store.useStore.getState().patchHistory('j1', { outcome: 'failed' });
    await actions.startJob();
    expect(client.createJob).toHaveBeenCalledWith({
      input_path: '/a.wav',
      profile: 'studio',
      overwrite: true,
    });
  });

  it('a run whose file a later run took over says so instead of serving it', async () => {
    const { actions, store, client } = await boot();
    // The net under the naming: two *different* clips whose names clean to the
    // same stem still collide, and the engine will happily overwrite.
    done(store, { jobId: 'j1', inputPath: '/a.m4a', outputPath: '/dir/a_studio.wav' });
    done(store, { jobId: 'j2', inputPath: '/a.wav', outputPath: '/dir/a_studio.wav' });
    expect(store.useStore.getState().history.find((h) => h.jobId === 'j1')?.supersededBy).toBe('j2');

    client.verify.mockClear();
    await actions.selectRun('j1');
    await settle();

    const st = store.useStore.getState();
    // Nothing is asked of the engine: the files are known not to be this run's.
    expect(client.verify).not.toHaveBeenCalled();
    expect(st.artifacts).toMatchObject({ master: false, json: false, txt: false });
    expect(st.artifacts?.reason).toContain('later run');
    expect(st.abMode).toBe('original');
    expect(st.cleaned).toBeNull();
  });
});

// B6 · the health heartbeat, which every "engine loss" claim in the log rests
// on and which nothing pinned. Iteration 3 records a deliberate decision — the
// healthy cadence was put BACK to 10 s after an agent had raised it to 5 s —
// and iteration 3's B6 entry records the 400→5000 ms offline ladder. Both were
// verified by killing a real engine and watching; here they are numbers.
describe('B6 · health cadence', () => {
  it('polls a healthy engine every 10 s, not faster', async () => {
    const { actions, client } = await boot();
    await actions.connectEngine();
    expect(client.health).toHaveBeenCalledTimes(1);

    await settle(9999);
    expect(client.health).toHaveBeenCalledTimes(1); // still one: no early beat
    await settle(1);
    expect(client.health).toHaveBeenCalledTimes(2);
    await settle(10000);
    expect(client.health).toHaveBeenCalledTimes(3);
  });

  it('chases a dead engine hard, then backs off to the same 10 s beat', async () => {
    const { actions, store, client } = await boot();
    client.health.mockRejectedValue(new TypeError('Failed to fetch'));
    await actions.connectEngine();
    expect(store.useStore.getState().engineStatus).toBe('offline');

    // 400 · 800 · 1600 · 3000 · 5000, then 5000 for as long as it stays down.
    const ladder = [400, 800, 1600, 3000, 5000, 5000];
    let calls = client.health.mock.calls.length;
    for (const delay of ladder) {
      await settle(delay - 1);
      expect(client.health.mock.calls.length, `no probe before ${delay} ms`).toBe(calls);
      await settle(1);
      calls += 1;
      expect(client.health.mock.calls.length, `probe at ${delay} ms`).toBe(calls);
    }
  });

  it('a returning engine resets the ladder to the healthy beat', async () => {
    const { actions, store, client } = await boot();
    client.health.mockRejectedValue(new TypeError('Failed to fetch'));
    await actions.connectEngine();
    await settle(400);
    await settle(800); // two rungs down the ladder
    expect(store.useStore.getState().engineStatus).toBe('offline');

    client.health.mockResolvedValue({
      ok: true,
      version: '3.2.0',
      profiles: ['studio'],
      engine_pid: 4242,
    });
    await settle(1600);
    expect(store.useStore.getState().engineStatus).toBe('ready');

    const calls = client.health.mock.calls.length;
    await settle(9999);
    expect(client.health.mock.calls.length).toBe(calls); // back on the slow beat
    await settle(1);
    expect(client.health.mock.calls.length).toBe(calls + 1);
  });

  it('a clip stranded by an outage is re-analysed when the engine returns', async () => {
    const { actions, store, client } = await boot();
    await actions.connectEngine();

    // Analyze dies the way a killed engine kills it: a bare TypeError.
    client.analyze.mockRejectedValueOnce(new TypeError('Failed to fetch'));
    await actions.loadSource({ path: '/a.wav', name: 'a.wav', origin: 'file' });
    expect(store.useStore.getState().original).toBeNull();
    const analyzeCalls = client.analyze.mock.calls.length;

    // The engine goes away and comes back under a new pid.
    client.health.mockRejectedValue(new TypeError('Failed to fetch'));
    await settle(10000);
    expect(store.useStore.getState().engineStatus).toBe('offline');
    client.health.mockResolvedValue({
      ok: true,
      version: '3.2.0',
      profiles: ['studio'],
      engine_pid: 4242,
    });
    await settle(5000);

    // The promise the failure sentence makes — "this comes back on its own
    // when it reconnects" — is actually kept.
    expect(client.analyze.mock.calls.length).toBeGreaterThan(analyzeCalls);
    expect(store.useStore.getState().original).not.toBeNull();
  });

  it('a refused clip is NOT retried on reconnect — a refusal is not an outage', async () => {
    const { actions, store, client, EngineError } = await boot();
    await actions.connectEngine();

    client.analyze.mockRejectedValueOnce(new EngineError(400, 'bad_request', 'unsupported'));
    await actions.loadSource({ path: '/bad.wav', name: 'bad.wav', origin: 'file' });
    const analyzeCalls = client.analyze.mock.calls.length;

    client.health.mockResolvedValue({
      ok: true,
      version: '3.2.0',
      profiles: ['studio'],
      engine_pid: 4242,
    });
    await settle(10000);
    expect(client.analyze.mock.calls.length).toBe(analyzeCalls);
    expect(store.useStore.getState().original).toBeNull();
  });

  it('a different pid behind the same port is a different engine', async () => {
    const { actions, store, client } = await boot();
    await actions.connectEngine();
    expect(store.useStore.getState().statusLine).toContain('Engine ready');

    client.health.mockResolvedValue({
      ok: true,
      version: '3.2.0',
      profiles: ['studio'],
      engine_pid: 9999, // restarted underneath us; nothing ever looked offline
    });
    await settle(10000);
    expect(store.useStore.getState().statusLine).toContain('Engine back');
  });
});

describe('B5 · a run keeps what its own analysis found, whoever is on screen', () => {
  it('caches the cleaned analysis on the run even when the user has moved on', async () => {
    const { actions, store, client } = await boot();
    // An older run to move to, and the run about to finish.
    done(store, { jobId: 'j0', outputPath: '/out/j0.wav' });
    armed(store);

    let land: ((a: AudioAnalysis) => void) | null = null;
    client.analyze.mockImplementation(
      () =>
        new Promise<AudioAnalysis>((res) => {
          land = res;
        }),
    );
    await actions.startJob();
    FakeEventSource.live.send('status', JSON.stringify(jobStatus('done')));
    await settle();
    expect(land).not.toBeNull(); // the cleaned analysis is in flight

    // The user opens the older run before that answer comes back.
    await actions.selectRun('j0');
    await settle();
    expect(store.useStore.getState().currentRunId).toBe('j0');

    (land as unknown as (a: AudioAnalysis) => void)(
      analysis({ path: '/out/a.wav', loudness: { integrated_lufs: -21.7, true_peak_dbtp: -1 }, noise_floor_db: -84.7 }),
    );
    await settle();

    // The row that ran keeps its numbers: it is that run's answer, not a
    // property of what happens to be on screen when it arrives. Before this,
    // the row read "— LUFS Δ" for the rest of the session and re-opening it
    // cost a full POST /api/analyze.
    const row = store.useStore.getState().history.find((h) => h.jobId === 'j1');
    expect(row?.cleaned).not.toBeNull();
    expect(row?.lufsOut).toBe(-21.7);
    expect(row?.noiseOut).toBe(-84.7);
    // …and the deck on screen is still the run the user asked for.
    expect(store.useStore.getState().currentRunId).toBe('j0');
    expect(store.useStore.getState().cleanedPath).toBe('/out/j0.wav');
  });
});

describe('B5 · restoring a run restores the run, not half of it', () => {
  it('puts the profile back, so PROCESS AGAIN means again', async () => {
    const { actions, store } = await boot();
    done(store, { jobId: 'j1', outputPath: '/out/j1.wav', profile: 'production' });
    done(store, { jobId: 'j2', outputPath: '/out/j2.wav', profile: 'studio' });
    expect(store.useStore.getState().profile).toBe('studio');

    await actions.selectRun('j1');
    await settle();

    // The screen used to assert both at once: a PRODUCTION row on screen with
    // the radiogroup reading STUDIO, and PROCESS AGAIN running studio.
    expect(store.useStore.getState().profile).toBe('production');
  });

  it('sends the restored run’s own profile when it is run again', async () => {
    const { actions, store, client } = await boot();
    done(store, { jobId: 'j1', outputPath: '/out/j1.wav', profile: 'production' });
    done(store, { jobId: 'j2', outputPath: '/out/j2.wav', profile: 'studio' });
    await actions.selectRun('j1');
    await settle();
    client.createJob.mockClear();
    await actions.startJob();
    expect(client.createJob.mock.calls[0]?.[0]).toMatchObject({ profile: 'production' });
  });
});

describe('A7 · selecting a run swaps identity and decks atomically-or-labelled', () => {
  // The measured failure this pins: with the engine hung (SIGSTOP — every
  // request stalls), clicking another run's row switched the job chip and the
  // clip name while both <audio> elements kept the previous run's 12 s blobs
  // and the A/B stayed lit CLEANED — the decks claimed audio they did not
  // hold, for as long as the hang lasted. The swap must therefore be settled
  // synchronously, before the first engine round-trip.
  it('unclaims mismatched decks and swaps the cached facts before the engine answers', async () => {
    const { actions, store, client } = await boot();
    done(store, { jobId: 'j1', inputPath: '/a.wav', outputPath: '/out/j1.wav' });
    done(store, { jobId: 'j2', inputPath: '/b.wav', outputPath: '/out/j2.wav' });
    // The engine hangs: the restore's first round-trip never resolves.
    client.verify.mockImplementation(() => new Promise(() => undefined));
    player.claimOnly.mockClear();

    const pending = actions.selectRun('j1'); // cannot finish — never awaited

    // Everything below is already true, synchronously:
    const st = store.useStore.getState();
    expect(st.currentRunId).toBe('j1');
    // the decks were told whose audio they may claim — j1's files, nothing else
    expect(player.claimOnly.mock.calls).toEqual([
      ['original', client.fileUrl('/a.wav')],
      ['cleaned', client.fileUrl('/out/j1.wav')],
    ]);
    // …and told before the engine was asked anything at all
    const firstClaim = player.claimOnly.mock.invocationCallOrder[0] ?? Infinity;
    const firstVerify = client.verify.mock.invocationCallOrder[0] ?? -Infinity;
    expect(firstClaim).toBeLessThan(firstVerify);
    // the run-local facts switched with the name, not after the engine answered
    expect(st.original?.path).toBe('/a.wav');
    expect(st.cleaned?.path).toBe('/out/j1.wav');
    expect(st.cleanedPath).toBe('/out/j1.wav');
    expect(st.statusLine).toContain('j1.wav');
    void pending;
  });

  it('a failed run’s restore claims no cleaned deck at all', async () => {
    const { actions, store } = await boot();
    store.useStore.getState().pushHistory({
      jobId: 'jf',
      profile: 'studio',
      inputPath: '/a.wav',
      inputName: 'a.wav',
      outcome: 'failed',
      durationMs: 900,
      at: Date.now(),
      enhanced: null,
      unitsTotal: null,
      lufsIn: null,
      lufsOut: null,
      noiseIn: null,
      noiseOut: null,
      outputPath: '',
      reportPath: '',
      report: null,
      status: jobStatus('failed'),
      original: analysis(),
      cleaned: null,
      error: 'it broke',
    });
    // j2 is pushed after jf, so jf is NOT the run on screen and its selection
    // takes the full restore path rather than the re-pick shortcut.
    done(store, { jobId: 'j2', outputPath: '/out/j2.wav' });
    player.claimOnly.mockClear();

    await actions.selectRun('jf');
    await settle();

    expect(player.claimOnly.mock.calls[1]).toEqual(['cleaned', null]);
    expect(store.useStore.getState().cleanedPath).toBeNull();
  });
});

describe('B5 · the run on screen is re-verified, not trusted for ever', () => {
  it('re-picking the row it is on asks the engine again — three ranged reads, no analyze', async () => {
    const { actions, store, client } = await boot();
    done(store, { jobId: 'j1', outputPath: '/out/j1.wav' });
    done(store, { jobId: 'j2', outputPath: '/out/j2.wav' });
    await actions.selectRun('j1');
    await settle();
    expect(store.useStore.getState().artifacts?.master).toBe(true);

    // The master is deleted under the live run. Nothing on screen probes it:
    // the cleaned deck plays from a blob already in memory.
    client.verify.mockImplementation(async (p: string) =>
      p === '/out/j1.wav'
        ? { status: 404, delivered: false, size: null }
        : { status: 206, delivered: true, size: 1000 },
    );
    client.verify.mockClear();
    client.analyze.mockClear();
    player.load.mockClear();

    await actions.selectRun('j1'); // the row it is already on
    await settle();

    expect(client.verify).toHaveBeenCalledTimes(3);
    expect(client.analyze).not.toHaveBeenCalled();
    const st = store.useStore.getState();
    expect(st.artifacts?.master).toBe(false);
    expect(st.history.find((h) => h.jobId === 'j1')?.artifacts?.master).toBe(false);
    expect(st.deckFault?.deck).toBe('cleaned');
    expect(st.cleaned).toBeNull();
    expect(st.abMode).toBe('original');
    expect(player.load).toHaveBeenCalledWith('cleaned', null);
  });

  it('says nothing new when the files are still there', async () => {
    const { actions, store, client } = await boot();
    done(store, { jobId: 'j1', outputPath: '/out/j1.wav' });
    done(store, { jobId: 'j2', outputPath: '/out/j2.wav' });
    await actions.selectRun('j1');
    await settle();
    player.load.mockClear();
    client.analyze.mockClear();

    await actions.selectRun('j1');
    await settle();

    const st = store.useStore.getState();
    expect(client.analyze).not.toHaveBeenCalled();
    expect(player.load).not.toHaveBeenCalled();
    expect(st.deckFault).toBeNull();
    expect(st.artifacts?.master).toBe(true);
  });

  it('an engine that cannot be asked accuses nobody', async () => {
    const { actions, store, client } = await boot();
    done(store, { jobId: 'j1', outputPath: '/out/j1.wav' });
    done(store, { jobId: 'j2', outputPath: '/out/j2.wav' });
    await actions.selectRun('j1');
    await settle();
    client.verify.mockRejectedValue(new TypeError('Failed to fetch'));

    await actions.selectRun('j1');
    await settle();

    const st = store.useStore.getState();
    expect(st.artifacts?.master).toBe(true); // the last real answer stands
    expect(st.deckFault).toBeNull();
  });
});

// A7 · the audit's two attacks on the run on screen. A HEAD answers 200 for
// both of them — a chmod-000 master (the stat is happy; only producing a body
// fails) and a 100-byte truncation (the file exists; the audio is gone) — so
// the re-verification has to *read* something, and for the master, measure it.
describe('A7 · re-verification reads the file instead of stat-ing it', () => {
  it('a master truncated under the live run is condemned by its recorded length', async () => {
    const { actions, store, client } = await boot();
    done(store, { jobId: 'j1', outputPath: '/out/j1.wav', masterBytes: 1000 });
    done(store, { jobId: 'j2', outputPath: '/out/j2.wav' });
    await actions.selectRun('j1');
    await settle();
    expect(store.useStore.getState().artifacts?.master).toBe(true);

    // The attack: `truncate -s 100`. The file is still there and still hands
    // over its ranged byte perfectly happily — only the length says it is not
    // the file this run wrote.
    client.verify.mockImplementation(async (p: string) =>
      p === '/out/j1.wav'
        ? { status: 206, delivered: true, size: 100 }
        : { status: 206, delivered: true, size: 1000 },
    );
    client.analyze.mockClear();
    await actions.selectRun('j1'); // the row it is already on: "look again"
    await settle();

    const st = store.useStore.getState();
    expect(client.analyze).not.toHaveBeenCalled();
    expect(st.artifacts?.master).toBe(false);
    expect(st.artifacts?.flag).toBe('FILE BROKEN');
    expect(st.artifacts?.reason).toContain('100 B');
    expect(st.artifacts?.reason).toContain('1,000 B');
    expect(st.deckFault?.deck).toBe('cleaned');
    expect(st.abMode).toBe('original');
    expect(st.cleaned).toBeNull();
    // The condemned sighting must not overwrite the yardstick it rests on.
    expect(st.history.find((h) => h.jobId === 'j1')?.masterBytes).toBe(1000);
  });

  it('a master the engine lists but cannot read is condemned, not served', async () => {
    const { actions, store, client } = await boot();
    done(store, { jobId: 'j1', outputPath: '/out/j1.wav' });
    done(store, { jobId: 'j2', outputPath: '/out/j2.wav' });
    await actions.selectRun('j1');
    await settle();

    // The attack: `chmod 000`. The engine commits a 2xx (the stat succeeded)
    // and then cannot produce the byte — twice, which is what separates the
    // file from an engine dying between headers and body.
    client.verify.mockImplementation(async (p: string) =>
      p === '/out/j1.wav'
        ? { status: 200, delivered: false, size: null }
        : { status: 206, delivered: true, size: 1000 },
    );
    await actions.selectRun('j1');
    await settle();

    const st = store.useStore.getState();
    expect(st.artifacts?.master).toBe(false);
    expect(st.artifacts?.flag).toBe('NO ACCESS');
    expect(st.artifacts?.reason).toContain('cannot read');
    expect(st.deckFault?.deck).toBe('cleaned');
    expect(st.abMode).toBe('original');
  });

  it('an undelivered byte followed by silence is an outage, not a condemnation', async () => {
    const { actions, store, client } = await boot();
    done(store, { jobId: 'j1', outputPath: '/out/j1.wav' });
    done(store, { jobId: 'j2', outputPath: '/out/j2.wav' });
    await actions.selectRun('j1');
    await settle();

    // Headers came, the byte did not, and the second ask got nothing at all:
    // the engine died between the two. Nothing true is known about the file.
    client.verify
      .mockImplementationOnce(async () => ({ status: 200, delivered: false, size: null }))
      .mockRejectedValue(new TypeError('Failed to fetch'));
    await actions.selectRun('j1');
    await settle();

    const st = store.useStore.getState();
    expect(st.artifacts?.master).toBe(true); // the last real answer stands
    expect(st.deckFault).toBeNull();
  });

  it('a first healthy verification records the length it saw as the yardstick', async () => {
    const { actions, store } = await boot();
    done(store, { jobId: 'j1', outputPath: '/out/j1.wav' });
    done(store, { jobId: 'j2', outputPath: '/out/j2.wav' });
    expect(store.useStore.getState().history.find((h) => h.jobId === 'j1')?.masterBytes).toBe(
      undefined,
    );
    await actions.selectRun('j1');
    await settle();
    expect(store.useStore.getState().history.find((h) => h.jobId === 'j1')?.masterBytes).toBe(1000);
  });
});

describe('A7 · a master that is there but unplayable is not a master', () => {
  async function faultListener(actions: ActionsMod): Promise<(f: unknown) => void> {
    await actions.connectEngine();
    await settle();
    const fn = player.onFault.mock.calls[0]?.[0];
    if (!fn) throw new Error('actions never subscribed to deck faults');
    return fn;
  }

  for (const [kind, flag, says] of [
    ['truncated', 'FILE BROKEN', 'empty or truncated'],
    ['unreadable', 'FILE BROKEN', 'nothing here can decode it'],
  ] as const) {
    it(`a ${kind} cleaned deck takes the download with it`, async () => {
      const { actions, store, client } = await boot();
      const fire = await faultListener(actions);
      const st = store.useStore.getState();
      st.setCleaned(analysis(), '/out/a.wav');
      st.setAbMode('cleaned');
      player.activeDeck = 'original';

      fire({
        deck: 'cleaned',
        url: client.fileUrl('/out/a.wav'),
        kind,
        status: null,
        fellBackTo: 'original',
        ...(kind === 'truncated' ? { duration: { got: 0.000396, expected: 100 } } : {}),
      });

      const after = store.useStore.getState();
      // The defect: `markArtifacts` used to run only for 'missing'/'forbidden',
      // so `Master WAV` stayed an enabled <a download> for a file the app had
      // just declared unplayable.
      expect(after.artifacts?.master).toBe(false);
      expect(after.artifacts?.reason).toContain(says);
      expect(after.artifacts?.flag).toBe(flag);
      expect(after.cleaned).toBeNull();
      expect(after.deckFault?.deck).toBe('cleaned');
    });
  }

  it('names the length a truncated deck actually opened with', async () => {
    const { actions, store, client } = await boot();
    const fire = await faultListener(actions);
    store.useStore.getState().setCleaned(analysis(), '/out/a.wav');
    player.activeDeck = 'original';
    fire({
      deck: 'cleaned',
      url: client.fileUrl('/out/a.wav'),
      kind: 'truncated',
      status: null,
      fellBackTo: 'original',
      duration: { got: 0.000396, expected: 94.6 },
    });
    const said = store.useStore.getState().deckFault?.detail ?? '';
    expect(said).toContain('less than a millisecond of audio');
    expect(said).toContain('94.6 s long');
    expect(said).toContain('playing the original');
  });

  it('a deck lost to an outage still condemns nothing', async () => {
    const { actions, store, client } = await boot();
    const fire = await faultListener(actions);
    store.useStore.getState().setCleaned(analysis(), '/out/a.wav');
    player.activeDeck = 'original';
    fire({
      deck: 'cleaned',
      url: client.fileUrl('/out/a.wav'),
      kind: 'network',
      status: null,
      fellBackTo: 'original',
    });
    expect(store.useStore.getState().artifacts).toBeNull();
    expect(store.useStore.getState().cleaned).not.toBeNull();
  });
});

describe('B6 · a deck lost to an outage comes back with the engine', () => {
  it('retries a cleaned deck whose analysis died in the same outage', async () => {
    const { actions, store, client } = await boot();
    await actions.connectEngine();
    const st = store.useStore.getState();
    st.setSource({ path: '/a.wav', name: 'a.wav', origin: 'file' });
    st.setOriginal(analysis());
    // The run finished, then the engine went away before its analysis landed:
    // a path with no cached analysis behind it, which the old retry skipped.
    st.setCleaned(null, '/out/a.wav');
    player.deckFault.mockImplementation((deck) => (deck === 'cleaned' ? { kind: 'network' } : null));

    client.health.mockRejectedValue(new TypeError('Failed to fetch'));
    await settle(10000);
    expect(store.useStore.getState().engineStatus).toBe('offline');
    player.load.mockClear();
    client.health.mockResolvedValue({ ok: true, version: '3.2.0', profiles: ['studio'], engine_pid: 4242 });
    await settle(5000);

    expect(player.load).toHaveBeenCalledWith('cleaned', client.fileUrl('/out/a.wav'), 100);
    expect(store.useStore.getState().deckFault).toBeNull();
  });
});

describe('restore mode (contract addendum 2)', () => {
  it('the health probe lands the engine’s speakers and the restore flag', async () => {
    const { actions, store, client } = await boot();
    store.useStore.getState().setEngine('offline', client as never, '3.2.0');
    client.health.mockResolvedValue({
      ok: true,
      version: '3.3.0',
      profiles: ['studio', 'lowband', 'production'],
      speakers: ['character_01', 'character_02'],
      restore_available: true,
      engine_pid: 4242,
    });
    actions.retryEngineNow();
    await settle(10);
    const st = store.useStore.getState();
    expect(st.speakers).toEqual(['character_01', 'character_02']);
    expect(st.restoreAvailable).toBe(true);
    expect(st.speakerId).toBe('character_01');
  });

  it('a revision-1 health answer (no fields) reads as no restore', async () => {
    const { actions, store, client } = await boot();
    store.useStore.getState().setCapabilities(['character_01'], true);
    store.useStore.getState().setEngine('offline', client as never, '3.2.0');
    // makeClient()’s default health has neither field — an older engine.
    actions.retryEngineNow();
    await settle(10);
    const st = store.useStore.getState();
    expect(st.restoreAvailable).toBe(false);
    expect(st.speakers).toEqual([]);
    expect(st.mode).toBe('natural');
    void client;
  });

  it('a restore submit carries mode and speaker — and only those when the cutoff is auto', async () => {
    const { actions, store, client } = await boot();
    armed(store);
    store.useStore.getState().setCapabilities(['character_01'], true);
    store.useStore.getState().setMode('restore');
    await actions.startJob();
    expect(client.createJob).toHaveBeenCalledWith({
      input_path: '/a.wav',
      profile: 'studio',
      overwrite: true,
      mode: 'restore',
      speaker_id: 'character_01',
    });
  });

  it('a manual cutoff rides the restore submit as cutoff_hz', async () => {
    const { actions, store, client } = await boot();
    armed(store);
    const st = store.useStore.getState();
    st.setCapabilities(['character_01', 'character_02'], true);
    st.setMode('restore');
    st.setSpeakerId('character_02');
    st.setCutoffHz(7800);
    await actions.startJob();
    expect(client.createJob).toHaveBeenCalledWith({
      input_path: '/a.wav',
      profile: 'studio',
      overwrite: true,
      mode: 'restore',
      speaker_id: 'character_02',
      cutoff_hz: 7800,
    });
  });

  it('a natural submit sends none of the three — the engine forbids extras', async () => {
    const { actions, store, client } = await boot();
    armed(store);
    const st = store.useStore.getState();
    // Restore was configured and then switched off; nothing may leak through,
    // not even as nulls (the engine’s extra="forbid" would 422 the run).
    st.setCapabilities(['character_01'], true);
    st.setCutoffHz(7800);
    st.setMode('natural');
    await actions.startJob();
    const req = client.createJob.mock.calls[0]?.[0] as Record<string, unknown>;
    expect(req).toEqual({ input_path: '/a.wav', profile: 'studio', overwrite: true });
    expect('mode' in req).toBe(false);
    expect('speaker_id' in req).toBe(false);
    expect('cutoff_hz' in req).toBe(false);
  });
});

describe('an engine that accepts the socket and never answers', () => {
  it('times the health probe out instead of deadlocking the heartbeat', async () => {
    const { actions, store, client } = await boot();

    // `probeInFlight` is cleared in a `finally`. With no timeout on the
    // request, an engine that opens the connection and then goes silent — hung,
    // paused, wedged behind a stuck filesystem call, or a proxy holding the
    // socket — leaves that await pending forever and the flag set forever.
    // Every later beat then returns at the `if (probeInFlight) return` guard,
    // so the heartbeat stops, `Retry now` becomes inert, and no offline banner
    // appears at all, because engineOfflineSince is only set on a *failed*
    // probe. The app sits looking connected to an engine that is gone.
    client.health.mockImplementation(
      (signal?: AbortSignal) =>
        new Promise((_resolve, reject) => {
          signal?.addEventListener('abort', () => reject(signal.reason));
        }),
    );

    // retryEngineNow is a no-op while the app believes it is connected.
    store.useStore.getState().setEngine('offline', client as never, '3.3.0');
    actions.retryEngineNow();
    await settle(10);
    expect(client.health).toHaveBeenCalledTimes(1);

    // While that request is outstanding `probeInFlight` is held, so nothing
    // else can probe. This is the state the bug made permanent.
    actions.retryEngineNow();
    await settle(10);
    expect(client.health).toHaveBeenCalledTimes(1);

    // Past the 4s ceiling the request aborts itself and the `finally` runs.
    // The engine is now recorded as unreachable, which is what raises the
    // offline banner — the bug's worst symptom was that this never happened,
    // so the app sat looking connected to an engine that was gone.
    await settle(5000);
    expect(store.useStore.getState().engineOfflineSince).not.toBeNull();

    // And the heartbeat is alive again: a later probe really is issued, and
    // the app can recover. Without the timeout this call is swallowed forever.
    client.health.mockImplementation(async () => ({
      ok: true,
      version: '3.3.0',
      profiles: ['studio'],
      engine_pid: 4242,
    }));
    actions.retryEngineNow();
    await settle(2000);
    // The claim that matters, and the one the bug broke: the heartbeat is not
    // wedged. A further probe really is issued. (Whether *this* probe flips the
    // lamp back to ready depends on the backoff schedule, which the B6 tests
    // above already cover; asserting it here would be asserting the scheduler,
    // not the deadlock.)
    expect(client.health.mock.calls.length).toBeGreaterThan(1);
  });

  it('passes a real timeout signal, not an already-settled one', async () => {
    const { actions, store, client } = await boot();
    store.useStore.getState().setEngine('offline', client as never, '3.3.0');
    actions.retryEngineNow();
    await settle(10);
    const signal = client.health.mock.calls[0]?.[0] as AbortSignal | undefined;
    expect(signal).toBeInstanceOf(AbortSignal);
    expect(signal?.aborted).toBe(false);
  });
});

describe('selectRun with an analysis still in flight', () => {
  it('cancels it, so the old clip cannot land on top of the restored run', async () => {
    const { actions, store, client } = await boot();
    const st = () => store.useStore.getState();

    // A finished run to go back to.
    armed(store);
    await actions.startJob();
    FakeEventSource.live.send('status', JSON.stringify(jobStatus('done')));
    await settle(10);
    const jobId = st().job?.id;
    expect(jobId).toBe('j1');
    expect(st().history.length).toBeGreaterThan(0);

    // Now the user loads a different clip. Hold its analysis open.
    let landAnalysis: ((a: unknown) => void) | undefined;
    client.analyze.mockImplementation(
      (path: string, _b?: number, signal?: AbortSignal) =>
        new Promise((resolve, reject) => {
          landAnalysis = (a) => resolve(a as never);
          signal?.addEventListener('abort', () => reject(signal.reason));
          void path;
        }),
    );
    void actions.loadSource({ path: '/b.wav', name: 'b.wav', origin: 'file' });
    await settle(10);
    expect(st().analyzing).toBe(true);

    // …and, before it lands, goes back to the finished run.
    await actions.selectRun(jobId as string);
    await settle(10);

    // The analysis must have been abandoned, exactly as `stopFollow` is.
    expect(st().analyzing).toBe(false);
    expect(st().source?.name).toBe('a.wav');

    // If clip B's analysis were still live, this would write its peaks,
    // duration and loudness over run A — B's audio and timecode under A's
    // name, with nothing on screen saying they disagree.
    landAnalysis?.(analysis({ path: '/b.wav', duration_s: 999 }));
    await settle(50);
    expect(st().source?.name).toBe('a.wav');
    expect(st().original?.duration_s).not.toBe(999);
  });
});

describe('a run that is refused must not destroy the one on screen', () => {
  it('keeps the finished run when createJob fails', async () => {
    const { actions, store, client, EngineError } = await boot();
    const st = () => store.useStore.getState();

    armed(store);
    await actions.startJob();
    FakeEventSource.live.send('status', JSON.stringify(jobStatus('done')));
    await settle(10);
    expect(st().report).not.toBeNull();
    expect(st().cleanedPath).toBe('/out/a.wav');
    const keptReport = st().report;

    // The engine refuses the next one — a path it will not take, a 4xx, a dead
    // socket. The prelude used to clear report, cleaned deck, artefacts and
    // the A/B *before* this call, so a refusal left an empty screen with the
    // plate still claiming the run it had just erased was complete.
    client.createJob.mockRejectedValueOnce(new EngineError(422, 'bad_input', 'refused'));
    await actions.startJob();
    await settle(10);

    expect(st().report).toBe(keptReport);
    expect(st().cleanedPath).toBe('/out/a.wav');
    expect(st().error).not.toBeNull();
  });

  it('refuses to start at all while the engine is gone, and says so', async () => {
    const { actions, store, client } = await boot();
    const st = () => store.useStore.getState();

    armed(store);
    await actions.startJob();
    FakeEventSource.live.send('status', JSON.stringify(jobStatus('done')));
    await settle(10);
    const keptReport = st().report;

    store.useStore.getState().setEngine('offline', client as never, '3.3.0');
    client.createJob.mockClear();
    await actions.startJob();
    await settle(10);

    expect(client.createJob).not.toHaveBeenCalled();
    expect(st().report).toBe(keptReport);
    expect(st().statusLine).toContain('engine is offline');
  });
});

describe('a job the engine has accepted but not yet reported on', () => {
  it('counts as in flight, so a second start cannot orphan it', async () => {
    const { actions, store, client } = await boot();
    const st = () => store.useStore.getState();

    armed(store);
    await actions.startJob();
    // The window between createJob returning an id and the first status
    // arriving. Written by hand as `job && job.status && !isTerminal(...)`,
    // this reads as *idle*.
    expect(st().job?.id).toBe('j1');
    expect(st().job?.status).toBeNull();

    client.createJob.mockClear();
    await actions.startJob();
    await settle(10);
    expect(client.createJob).not.toHaveBeenCalled();
  });

  it('counts as in flight for loadSource, so a drop cannot orphan it', async () => {
    const { actions, store, client } = await boot();
    const st = () => store.useStore.getState();

    armed(store);
    await actions.startJob();
    expect(st().job?.status).toBeNull();

    client.analyze.mockClear();
    await actions.loadSource({ path: '/b.wav', name: 'b.wav', origin: 'file' });
    await settle(10);

    // The running job keeps the screen; the new clip is refused with a reason.
    expect(client.analyze).not.toHaveBeenCalled();
    expect(st().source?.name).toBe('a.wav');
    expect(st().job?.id).toBe('j1');
  });
});

describe('loading a clip whose analysis never lands', () => {
  it('does not leave the previous clip playing under the new name', async () => {
    const { actions, store, client, EngineError } = await boot();
    const st = () => store.useStore.getState();

    // Clip A is loaded and its deck is live.
    armed(store);
    player.load.mockClear();
    player.claimOnly.mockClear();
    player.seek.mockClear();

    // Clip B is refused — a rate the engine will not take, a codec it cannot
    // read. `analyzing` clears, the source name becomes B, and the original
    // deck used to still hold A's blob, so the transport played A under B's
    // name. `player.time` reads currentTime ungated, so A's timecode came too.
    client.analyze.mockRejectedValueOnce(new EngineError(415, 'bad_rate', 'refused'));
    await actions.loadSource({ path: '/b.wav', name: 'b.wav', origin: 'file' });
    await settle(20);

    expect(st().source?.name).toBe('b.wav');
    expect(st().original).toBeNull();
    // Both decks let go, and the playhead reset before they did.
    expect(player.seek).toHaveBeenCalledWith(0);
    expect(player.claimOnly).toHaveBeenCalledWith('original', null);
    expect(player.load).toHaveBeenCalledWith('cleaned', null);
  });
});

describe('a stream that ends without a terminal status', () => {
  it('does not leave the run stuck on running when the reconciling getJob fails', async () => {
    const { actions, store, client, EngineError } = await boot();
    const st = () => store.useStore.getState();

    armed(store);
    await actions.startJob();
    FakeEventSource.live.send('status', JSON.stringify(jobStatus('running')));
    await settle(10);
    expect(st().job?.status?.state).toBe('running');

    // The stream ends while the engine is up, and every attempt to ask what
    // happened fails — a 500, or the bad_response an unparseable 200 raises.
    // The `.catch` used to swallow this: PROCESS stuck as CANCEL, the drop well
    // shut on "Busy", no history row, and nothing on screen saying why.
    client.getJob.mockRejectedValue(new EngineError(500, 'boom', 'engine exploded'));
    // Each failed reconciliation re-arms the follow, so the retry budget is
    // spent one *stream* at a time — the new EventSource has to end too. Three
    // is RECONCILE_LIMIT; the fourth end is the one that gives up.
    for (let i = 0; i < 4; i++) {
      FakeEventSource.live.send('end', '{}');
      await settle(200);
    }

    const state = st().job?.status?.state;
    expect(state).not.toBe('running');
    expect(state).toBe('failed');
    expect(st().job?.status?.error?.code).toBe('LOST_TRACK');
    // The busy lock is released — that is what actually unsticks the screen.
    expect(st().history.length).toBeGreaterThan(0);
  });

  it('retries a live engine rather than condemning a run on one transient 500', async () => {
    const { actions, store, client, EngineError } = await boot();
    const st = () => store.useStore.getState();

    armed(store);
    await actions.startJob();
    FakeEventSource.live.send('status', JSON.stringify(jobStatus('running')));
    await settle(10);

    // One failure, then the engine answers properly.
    client.getJob.mockRejectedValueOnce(new EngineError(500, 'boom', 'transient'));
    FakeEventSource.live.send('end', '{}');
    await settle(200);

    // Not condemned: a single 500 from an engine that is plainly still up says
    // nothing about whether the run is still going.
    expect(st().job?.status?.error?.code).not.toBe('LOST_TRACK');
    FakeEventSource.live.send('status', JSON.stringify(jobStatus('done')));
    await settle(20);
    expect(st().job?.status?.state).toBe('done');
  });

  it('treats a 404 as the engine having lost the job, not as an unknown outcome', async () => {
    const { actions, store, client, EngineError } = await boot();
    const st = () => store.useStore.getState();

    armed(store);
    await actions.startJob();
    FakeEventSource.live.send('status', JSON.stringify(jobStatus('running')));
    await settle(10);

    client.getJob.mockRejectedValue(new EngineError(404, 'not_found', 'gone'));
    FakeEventSource.live.send('end', '{}');
    await settle(200);

    // ENGINE_RESTARTED can promise nothing was written; LOST_TRACK cannot, and
    // saying the weaker thing here would send the user to check a folder for
    // a file we know does not exist.
    expect(st().job?.status?.error?.code).toBe('ENGINE_RESTARTED');
  });
});

describe('retryFaultedDecks and the store’s single deckFault', () => {
  it('does not erase an explanation about the deck it did not retry', async () => {
    const { actions, store, client } = await boot();
    const st = () => store.useStore.getState();

    // A first successful probe, so the app has "ever connected". Without it the
    // next probe takes the first-connection branch and never reaches the
    // engine-came-back branch that calls retryFaultedDecks at all.
    st().setEngine('offline', client as never, '3.3.0');
    actions.retryEngineNow();
    await settle(100);
    expect(st().engineStatus).toBe('ready');

    armed(store);
    // Run on screen whose cleaned master has been deleted: the plate carries
    // the sentence, and artifacts.master stays false so CLEANED stays greyed.
    st().setCleaned(null, '/out/a.wav');
    st().setDeckFault({
      deck: 'cleaned',
      headline: 'CLEANED DECK UNAVAILABLE',
      detail: 'The cleaned master is no longer on disk.',
    });
    // The engine goes and comes back — the real path, through a health probe
    // that finds it up again after it was down. `retryFaultedDecks` runs only
    // on that transition, so setting the store to 'ready' by hand would have
    // exercised nothing.
    player.deckFault.mockImplementation((d: string) =>
      d === 'original' ? { kind: 'network', detail: 'unreachable' } : null,
    );
    st().setEngine('offline', client as never, '3.3.0');
    actions.retryEngineNow();
    await settle(100);
    expect(st().engineStatus).toBe('ready');

    // Guard against a vacuous pass: the original deck must really have been
    // retried, or nothing in retryFaultedDecks ran and the assertion below
    // proves only that the store was left alone.
    expect(player.load).toHaveBeenCalledWith(
      'original',
      expect.stringContaining('%2Fa.wav'),
      100,
    );

    // The cleaned deck's sentence must survive: nothing re-states it, because
    // reverifyCurrentRun returns early when the master was already known bad,
    // so clearing it here loses it for the rest of the session.
    expect(st().deckFault?.deck).toBe('cleaned');
    expect(st().deckFault?.detail).toContain('no longer on disk');
  });
});

describe('the web-mode upload state machine', () => {
  /** A file the bridge cannot give a local path for, so it must be uploaded. */
  const wav = (name = 'take-01.wav', size = 1024): File => {
    const f = new File([new Uint8Array(size)], name, { type: 'audio/wav' });
    Object.defineProperty(f, 'size', { value: size });
    return f;
  };

  it('publishes progress so the bar can move, and flips to "finishing" at 100%', async () => {
    const { actions, store, client } = await boot();
    const st = () => store.useStore.getState();
    let report: ((l: number, t: number) => void) | undefined;
    client.uploadWithProgress = vi.fn(
      (_f: File, o: { onProgress: (l: number, t: number) => void; onCancelHandle: (c: () => void) => void }) =>
        new Promise(() => {
          report = o.onProgress;
          o.onCancelHandle(() => undefined);
        }),
    );

    void actions.ingestFile(wav());
    await settle(10);
    expect(st().upload?.name).toBe('take-01.wav');
    expect(st().upload?.phase).toBe('sending');

    report?.(512, 1024);
    await settle(10);
    expect(st().upload?.loaded).toBe(512);
    expect(st().upload?.phase).toBe('sending');

    // The engine is still writing the file to the work dir at this point, so
    // 100% is "sent", not "done" — the phase is what the strip renders as
    // "writing to the work dir".
    report?.(1024, 1024);
    await settle(10);
    expect(st().upload?.phase).toBe('finishing');
  });

  it('a cancelled upload is not an error', async () => {
    const { actions, store, client } = await boot();
    const st = () => store.useStore.getState();
    let abort: (() => void) | undefined;
    client.uploadWithProgress = vi.fn(
      (_f: File, o: { onCancelHandle: (c: () => void) => void }) =>
        new Promise((_res, rej) => {
          o.onCancelHandle(() => {
            abort = () => undefined;
            rej(new DOMException('aborted', 'AbortError'));
          });
        }),
    );

    void actions.ingestFile(wav());
    await settle(10);
    expect(actions.isUploading()).toBe(true);

    actions.cancelUpload();
    await settle(20);

    // The branch this test exists for: the user stopping a transfer is not a
    // failure, so no red bar — just a status line saying what happened.
    expect(st().error).toBeNull();
    expect(st().statusLine).toContain('Upload cancelled');
    expect(st().statusLine).toContain('take-01.wav');
    // …and the machine is fully unwound, or the next drop is refused as busy.
    expect(st().upload).toBeNull();
    expect(actions.isUploading()).toBe(false);
    void abort;
  });

  it('a failed upload IS an error, and names the file', async () => {
    const { actions, store, client, EngineError } = await boot();
    const st = () => store.useStore.getState();
    client.uploadWithProgress = vi.fn(() =>
      Promise.reject(new EngineError(413, 'too_large', 'file exceeds the limit')),
    );

    await actions.ingestFile(wav());
    await settle(20);

    expect(st().error).not.toBeNull();
    expect(st().upload).toBeNull();
    expect(actions.isUploading()).toBe(false);
  });

  it('hands the uploaded path straight to loadSource', async () => {
    const { actions, store, client } = await boot();
    const st = () => store.useStore.getState();
    client.uploadWithProgress = vi.fn(() => Promise.resolve({ path: '/work/up/take-01.wav' }));

    await actions.ingestFile(wav());
    await settle(20);

    expect(client.analyze).toHaveBeenCalledWith('/work/up/take-01.wav', expect.any(Number), expect.anything());
    expect(st().source?.path).toBe('/work/up/take-01.wav');
    // The *user's* name for the file, not the work-dir spelling.
    expect(st().source?.name).toBe('take-01.wav');
    expect(st().upload).toBeNull();
  });

  it('refuses a file it cannot open before uploading a byte of it', async () => {
    const { actions, store, client } = await boot();
    const st = () => store.useStore.getState();
    client.uploadWithProgress = vi.fn();
    await actions.ingestFile(new File(['x'], 'notes.txt', { type: 'text/plain' }));
    await settle(10);
    expect(client.uploadWithProgress).not.toHaveBeenCalled();
    expect(st().rejection?.kind).toBe('type');
  });

  it('refuses an empty file, which would upload fine and then fail to decode', async () => {
    const { actions, store, client } = await boot();
    const st = () => store.useStore.getState();
    client.uploadWithProgress = vi.fn();
    await actions.ingestFile(wav('empty.wav', 0));
    await settle(10);
    expect(client.uploadWithProgress).not.toHaveBeenCalled();
    expect(st().rejection?.kind).toBe('empty');
  });
});
