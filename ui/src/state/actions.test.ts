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
  pause: vi.fn(),
  load: vi.fn(),
  setActive: vi.fn(),
  seek: vi.fn(),
  toggle: vi.fn(),
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
  createJob: ReturnType<typeof vi.fn>;
  getJob: ReturnType<typeof vi.fn>;
  cancelJob: ReturnType<typeof vi.fn>;
  peaks: ReturnType<typeof vi.fn>;
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
    peaks: vi.fn(),
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
    expect(player.load).toHaveBeenCalledWith('cleaned', client.fileUrl('/out/a.wav'));
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
    expect(client.getJob).toHaveBeenCalledWith('j1');
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
    expect(player.load).toHaveBeenCalledWith('original', client.fileUrl('/a.wav'));
    expect(player.load).toHaveBeenCalledWith('cleaned', client.fileUrl('/out/j1.wav'));
    expect(st.analyzing).toBe(false);
    expect(st.statusLine).toContain('units enhanced');
  });

  it('is a no-op for the run already on screen', async () => {
    const { actions, store, client } = await boot();
    finished(store, 'j1');
    player.pause.mockClear();
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
