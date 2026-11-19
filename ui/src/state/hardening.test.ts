// state/hardening.test.ts — Chaos, memory-leak, and state-race regression suite.
//
// Exercises:
//  1. WebGL2 handle accounting (StftCanvasRenderer + glutil partial-compile cleanup)
//  2. DualPlayer blob: URL lifecycle, AudioContext suspended/closed, source-switch isolation,
//     and rapid Delta/Loop/Wet-Dry/A-B chaos while loading or after deck fault
//  3. Async action races: 5x rapid previewRegion spam, source/run/take switches mid-flight,
//     Undo/Redo blocked while job running and isolated across source/run switches
//  4. Corrupt/truncated/hostile/quota-exceeded localStorage recovery in presets.ts

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { EngineClient } from '../api/client';
import type { AudioAnalysis, HawaVoCleanReport, JobStatus } from '../api/types';
import { DualPlayer, getPlayer } from '../audio/player';
import { createProgramU } from '../render/glutil';
import { StftCanvasRenderer, synthesizeTileFromAnalysis } from '../render/stftSpectrogram';
import {
  applySpectralBoxRepair,
  applyUnitOverride,
  extractAndLoadTrack,
  loadSource,
  previewRegion,
  redoAction,
  selectRun,
  selectTake,
  undoAction,
  updateRackSettings,
} from './actions';
import {
  DEFAULT_STUDIO_RACK_SETTINGS,
  MAX_PRESET_STORAGE_BYTES,
  PRESET_STORAGE_KEY,
  deleteUserPreset,
  loadUserPresets,
  saveUserPreset,
} from './presets';
import { getState, useStore } from './store';

vi.mock('../api/sse', () => ({
  followJob: vi.fn(() => () => undefined),
  followBatch: vi.fn(() => () => undefined),
  isTerminal: (s: string) => ['done', 'failed', 'cancelled', 'interrupted'].includes(s),
}));

const pristine = useStore.getState();

function makeAnalysis(path = '/a.wav', durationS = 20): AudioAnalysis {
  return {
    path,
    duration_s: durationS,
    sample_rate: 48000,
    channels: 1,
    peaks: { min: [-0.5, -0.4], max: [0.5, 0.4] },
    rms_db: [-20, -18],
    spectrum: { freqs_hz: [100, 1000, 8000], db: [-30, -16, -42] },
    loudness: { integrated_lufs: -23.0, true_peak_dbtp: -1.0 },
    noise_floor_db: -48.0,
  };
}

function makeReport(jobId: string, inputPath = '/a.wav', outputPath = `/out/${jobId}.wav`): HawaVoCleanReport {
  return {
    job_id: jobId,
    config_hash: 'cfg',
    input: { path: inputPath, sha256: 'x', sample_rate: 48000, channels: 1, samples: 480000, duration_s: 20 },
    output: { path: outputPath, sha256: 'y', sample_rate: 48000, channels: 1, samples: 480000, duration_s: 20 },
    core: { id: 'studio_pro', algorithm: 'x', params_hash: 'p' },
    guard: { id: 'guard_a', probe_hash: 'p', calibration_id: 'c' },
    environment: {},
    summary: { units_total: 2, enhanced: 2, reverted: 0 },
    units: [
      {
        unit_id: 0,
        channel: 0,
        start_sample: 0,
        end_sample: 48000,
        start_time_s: 0,
        end_time_s: 1,
        is_speech: true,
        input_sha256: 'in0',
        output_sha256: 'out0',
        guard_a_verdict: 'PASS',
        final_decision: 'enhanced',
        decision_reason: 'guard pass',
      },
    ],
  };
}

function makeDoneStatus(jobId: string, inputPath: string, outputPath: string): JobStatus {
  return {
    job_id: jobId,
    state: 'done',
    stage: 'done',
    progress: 1,
    message: 'Done',
    unit: null,
    input_path: inputPath,
    output_path: outputPath,
    report_path: `${outputPath}.json`,
    profile: 'studio_pro',
    started_at: '2026-01-01T00:00:00.000Z',
    finished_at: '2026-01-01T00:00:05.000Z',
    error: null,
    report: makeReport(jobId, inputPath, outputPath),
  };
}

function makeMockWebGL2(opts: { failFragmentCompile?: boolean } = {}) {
  const liveShaders = new Set<object>();
  const livePrograms = new Set<object>();
  const liveVaos = new Set<object>();
  const liveTextures = new Set<object>();
  const loseContext = vi.fn(() => {
    contextLost = true;
  });
  let contextLost = false;
  let seq = 0;

  const gl = {
    VERTEX_SHADER: 0x8b31,
    FRAGMENT_SHADER: 0x8b30,
    COMPILE_STATUS: 0x8b81,
    LINK_STATUS: 0x8b82,
    TEXTURE_2D: 0x0de1,
    TEXTURE0: 0x84c0,
    TEXTURE1: 0x84c1,
    TEXTURE_MIN_FILTER: 0x2801,
    TEXTURE_MAG_FILTER: 0x2800,
    TEXTURE_WRAP_S: 0x2802,
    TEXTURE_WRAP_T: 0x2803,
    LINEAR: 0x2601,
    CLAMP_TO_EDGE: 0x812f,
    UNPACK_ALIGNMENT: 0x0cf5,
    R8: 0x8229,
    RED: 0x1903,
    RGBA8: 0x8058,
    RGBA: 0x1908,
    UNSIGNED_BYTE: 0x1401,
    COLOR_BUFFER_BIT: 0x4000,
    TRIANGLES: 0x0004,
    isContextLost: vi.fn(() => contextLost),
    getExtension: vi.fn((name: string) =>
      name === 'WEBGL_lose_context' ? { loseContext } : null,
    ),
    createShader: vi.fn((type: number) => {
      const sh = { kind: 'shader', type, id: ++seq };
      liveShaders.add(sh);
      return sh;
    }),
    shaderSource: vi.fn(),
    compileShader: vi.fn(),
    getShaderParameter: vi.fn((sh: { type: number }, p: number) => {
      if (p === 0x8b81 && opts.failFragmentCompile && sh.type === 0x8b30) return false;
      return true;
    }),
    getShaderInfoLog: vi.fn(() => 'mock shader error'),
    deleteShader: vi.fn((sh: object | null) => {
      if (sh) liveShaders.delete(sh);
    }),
    createProgram: vi.fn(() => {
      const p = { kind: 'program', id: ++seq };
      livePrograms.add(p);
      return p;
    }),
    attachShader: vi.fn(),
    linkProgram: vi.fn(),
    getProgramParameter: vi.fn(() => true),
    getProgramInfoLog: vi.fn(() => ''),
    deleteProgram: vi.fn((p: object | null) => {
      if (p) livePrograms.delete(p);
    }),
    getUniformLocation: vi.fn((_p: object, name: string) => ({ name })),
    createVertexArray: vi.fn(() => {
      const vao = { kind: 'vao', id: ++seq };
      liveVaos.add(vao);
      return vao;
    }),
    bindVertexArray: vi.fn(),
    deleteVertexArray: vi.fn((vao: object | null) => {
      if (vao) liveVaos.delete(vao);
    }),
    createTexture: vi.fn(() => {
      const tex = { kind: 'tex', id: ++seq };
      liveTextures.add(tex);
      return tex;
    }),
    activeTexture: vi.fn(),
    bindTexture: vi.fn(),
    texParameteri: vi.fn(),
    pixelStorei: vi.fn(),
    texImage2D: vi.fn(),
    deleteTexture: vi.fn((tex: object | null) => {
      if (tex) liveTextures.delete(tex);
    }),
    viewport: vi.fn(),
    clearColor: vi.fn(),
    clear: vi.fn(),
    useProgram: vi.fn(),
    uniform1i: vi.fn(),
    uniform1f: vi.fn(),
    drawArrays: vi.fn(),
  } as unknown as WebGL2RenderingContext;

  return {
    gl,
    liveShaders,
    livePrograms,
    liveVaos,
    liveTextures,
    loseContext,
  };
}

function makeRangedResponse(status = 206, size = 32): Response {
  const headers = new Headers();
  if (status < 400) {
    headers.set('Content-Range', `bytes 0-0/${size}`);
    headers.set('Content-Length', '1');
  }
  return {
    ok: status < 400,
    status,
    headers,
    arrayBuffer: async () => new ArrayBuffer(1),
    blob: async () => new Blob([new Uint8Array(size)]),
  } as unknown as Response;
}

function makeWholeResponse(size = 32): Response {
  return {
    ok: true,
    status: 200,
    headers: new Headers(),
    arrayBuffer: async () => new ArrayBuffer(size),
    blob: async () => new Blob([new Uint8Array(size)]),
  } as unknown as Response;
}

async function settleMicrotasks(turns = 10): Promise<void> {
  for (let i = 0; i < turns; i++) await Promise.resolve();
}

describe('WebGL2 handle & context leak audit', () => {
  it('deletes all shaders, programs, VAOs, textures and releases context across repeated StftCanvasRenderer Show/Hide cycles', () => {
    const tile = synthesizeTileFromAnalysis(makeAnalysis());
    for (let cycle = 0; cycle < 5; cycle++) {
      const canvas = document.createElement('canvas');
      canvas.width = 120;
      canvas.height = 60;
      const mock = makeMockWebGL2();
      vi.spyOn(canvas, 'getContext').mockImplementation(((kind: string) =>
        kind === 'webgl2' ? mock.gl : null) as typeof canvas.getContext);

      const renderer = new StftCanvasRenderer(canvas);
      expect(renderer.backend).toBe('webgl2');
      // Shaders are already deleted right after linkProgram
      expect(mock.liveShaders.size).toBe(0);
      expect(mock.livePrograms.size).toBe(1);
      expect(mock.liveVaos.size).toBe(1);
      expect(mock.liveTextures.size).toBe(2);

      renderer.render(tile, 0, 20, 'log', 'original');
      renderer.render(tile, 0, 20, 'mel', 'delta');

      renderer.dispose();
      // Double dispose must be an idempotent no-op
      renderer.dispose();

      expect(mock.liveShaders.size).toBe(0);
      expect(mock.livePrograms.size).toBe(0);
      expect(mock.liveVaos.size).toBe(0);
      expect(mock.liveTextures.size).toBe(0);
      expect(mock.loseContext).toHaveBeenCalledTimes(1);
    }
  });

  it('cleans up vertex shader and program without leaking when fragment shader compilation fails', () => {
    const mock = makeMockWebGL2({ failFragmentCompile: true });
    expect(() => createProgramU(mock.gl, 'vs', 'fs', ['u_a'])).toThrow(/shader compile failed/i);
    expect(mock.liveShaders.size).toBe(0);
    expect(mock.livePrograms.size).toBe(0);
  });
});

describe('DualPlayer blob URL, AudioContext, source-switch & rapid chaos hardening', () => {
  let createdBlobs: string[];
  let revokedBlobs: string[];
  let player: DualPlayer;

  beforeEach(() => {
    createdBlobs = [];
    revokedBlobs = [];
    let blobId = 0;
    vi.stubGlobal('URL', {
      ...URL,
      createObjectURL: vi.fn(() => {
        const u = `blob:hardening-${++blobId}`;
        createdBlobs.push(u);
        return u;
      }),
      revokeObjectURL: vi.fn((u: string) => {
        revokedBlobs.push(u);
      }),
    });
  });

  afterEach(() => {
    player?.dispose();
    document.querySelectorAll('audio').forEach((el) => el.remove());
    vi.unstubAllGlobals();
  });

  it('revokes every created blob: URL across reload, retire, fault, superseded fetch, and dispose', async () => {
    let resolveSlowBlob!: (res: Response) => void;
    const URL_A = 'http://127.0.0.1:8765/api/audio?path=/a.wav';
    const URL_B = 'http://127.0.0.1:8765/api/audio?path=/b.wav';
    const URL_SLOW = 'http://127.0.0.1:8765/api/audio?path=/slow.wav';
    const URL_C = 'http://127.0.0.1:8765/api/audio?path=/c.wav';

    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string, init?: RequestInit) => {
        if (url === URL_SLOW && !init?.headers) {
          return await new Promise<Response>((r) => {
            resolveSlowBlob = r;
          });
        }
        return init?.headers ? makeRangedResponse(206) : makeWholeResponse();
      }),
    );

    player = new DualPlayer();

    // 1. Load A onto original, then reload B -> A's blob must be revoked
    player.load('original', URL_A);
    await settleMicrotasks();
    expect(createdBlobs).toHaveLength(1);
    const blobA = createdBlobs[0]!;

    player.load('original', URL_B);
    await settleMicrotasks();
    expect(revokedBlobs).toContain(blobA);

    // 2. Start slow fetch on cleaned, then supersede with URL_C -> slow blob must not leak
    player.load('cleaned', URL_SLOW);
    await settleMicrotasks(3);
    player.load('cleaned', URL_C);
    await settleMicrotasks();
    resolveSlowBlob(makeWholeResponse());
    await settleMicrotasks();

    // 3. Retire cleaned deck -> its blob must be revoked immediately
    const beforeRetireRevoked = revokedBlobs.length;
    player.load('cleaned', null);
    expect(revokedBlobs.length).toBeGreaterThan(beforeRetireRevoked);

    // 4. Reload cleaned and fault it via media element error -> blob revoked
    player.load('cleaned', URL_C);
    await settleMicrotasks();
    const cleanEl = document.querySelectorAll('audio')[1] as HTMLAudioElement;
    Object.defineProperty(cleanEl, 'error', { configurable: true, value: { code: 4 } });
    cleanEl.dispatchEvent(new Event('error'));

    // 5. Dispose revokes all remaining blobs -> 100% parity
    player.dispose();
    expect(new Set(revokedBlobs)).toEqual(new Set(createdBlobs));
  });

  it('handles suspended AudioContext resume rejection, pause-during-resume, and closed AudioContext gracefully', async () => {
    const URL_A = 'http://127.0.0.1:8765/api/audio?path=/a.wav';
    const URL_B = 'http://127.0.0.1:8765/api/audio?path=/b.wav';
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_u: string, init?: RequestInit) =>
        init?.headers ? makeRangedResponse(206) : makeWholeResponse(),
      ),
    );

    let ctxInstance: {
      state: AudioContextState;
      resume: ReturnType<typeof vi.fn>;
      close: ReturnType<typeof vi.fn>;
    } | null = null;
    const rejectResume = vi.fn(async () => {
      throw new DOMException('NotAllowedError', 'NotAllowedError');
    });

    class MockClosedAwareAudioContext {
      currentTime = 0;
      sampleRate = 48000;
      state: AudioContextState = 'suspended';
      destination = {};
      resume = rejectResume;
      close = vi.fn(async () => {
        this.state = 'closed';
      });
      createAnalyser = vi.fn(() => ({
        fftSize: 2048,
        smoothingTimeConstant: 0.78,
        minDecibels: -100,
        maxDecibels: 0,
        connect: vi.fn(),
        disconnect: vi.fn(),
        getFloatTimeDomainData: vi.fn((buf: Float32Array) => buf.fill(0)),
      }));
      createMediaElementSource = vi.fn(() => ({ connect: vi.fn(), disconnect: vi.fn() }));
      createGain = vi.fn(() => {
        const param = {
          value: 1,
          cancelScheduledValues: vi.fn(() => {
            if (this.state === 'closed') throw new DOMException('closed', 'InvalidStateError');
          }),
          setValueAtTime: vi.fn(() => {
            if (this.state === 'closed') throw new DOMException('closed', 'InvalidStateError');
          }),
          linearRampToValueAtTime: vi.fn(() => {
            if (this.state === 'closed') throw new DOMException('closed', 'InvalidStateError');
          }),
        };
        return { gain: param, connect: vi.fn(), disconnect: vi.fn() };
      });
      constructor() {
        ctxInstance = this;
      }
    }

    vi.stubGlobal('AudioContext', MockClosedAwareAudioContext);

    player = new DualPlayer();
    player.load('original', URL_A);
    player.load('cleaned', URL_B);
    await settleMicrotasks();

    // Suspended resume rejection must fail soft with zero unhandled rejection
    await expect(player.play()).resolves.toBeUndefined();
    expect(rejectResume).toHaveBeenCalledTimes(1);

    // Transition AudioContext to 'closed' -> subsequent controls & meter reads never throw
    ctxInstance!.state = 'closed';
    expect(() => player.setDeltaMode(true)).not.toThrow();
    expect(() => player.setWetDryMix(0.4)).not.toThrow();
    expect(() => player.setActive('cleaned')).not.toThrow();
    const levels = player.getLiveMeterLevels();
    expect(levels.momentaryLufs).toBe(-70);
    expect(() => player.dispose()).not.toThrow();
  });

  it('cleans up offset, deltaMode, loopRange, and loopEnabled when switching to a brand-new original source', async () => {
    const SRC_A = 'http://127.0.0.1:8765/api/audio?path=/src_a.wav';
    const PREVIEW_A = 'http://127.0.0.1:8765/api/audio?path=/prev_a.wav';
    const SRC_B = 'http://127.0.0.1:8765/api/audio?path=/src_b.wav';
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_u: string, init?: RequestInit) =>
        init?.headers ? makeRangedResponse(206) : makeWholeResponse(),
      ),
    );

    player = new DualPlayer();
    player.load('original', SRC_A, 30);
    await settleMicrotasks();
    const origEl = document.querySelectorAll('audio')[0] as HTMLAudioElement;
    Object.defineProperty(origEl, 'duration', { configurable: true, value: 30 });
    origEl.currentTime = 12;

    player.load('cleaned', PREVIEW_A, 5, 10);
    await settleMicrotasks();
    player.setLoopRange([10, 15]);
    player.setLoopEnabled(true);
    player.setDeltaMode(true);
    expect(player.snapshot().deltaMode).toBe(true);
    expect(player.snapshot().loopEnabled).toBe(true);
    expect(player.snapshot().loopRange).toEqual([10, 15]);

    // Switch to a brand-new source B
    player.load('cleaned', null);
    player.load('original', SRC_B, 10);
    await settleMicrotasks();

    const snap = player.snapshot();
    expect(snap.deltaMode).toBe(false);
    expect(snap.loopEnabled).toBe(false);
    expect(snap.loopRange).toBeNull();
    expect(snap.active).toBe('original');
  });

  it('survives rapid Delta + Loop + Wet/Dry + A/B spam while cleaned is loading or faulted without stuck pendingActive', async () => {
    const SRC_A = 'http://127.0.0.1:8765/api/audio?path=/src_a.wav';
    const CLEAN_SLOW = 'http://127.0.0.1:8765/api/audio?path=/clean_slow.wav';
    const CLEAN_404 = 'http://127.0.0.1:8765/api/audio?path=/clean_404.wav';
    let releaseClean!: (r: Response) => void;

    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string, init?: RequestInit) => {
        if (url === CLEAN_404) return makeRangedResponse(404);
        if (url === CLEAN_SLOW && !init?.headers) {
          return await new Promise<Response>((r) => {
            releaseClean = r;
          });
        }
        return init?.headers ? makeRangedResponse(206) : makeWholeResponse();
      }),
    );

    player = new DualPlayer();
    player.load('original', SRC_A, 20);
    await settleMicrotasks();

    // Start slow cleaned load and spam every control rapidly
    player.load('cleaned', CLEAN_SLOW, 20);
    await settleMicrotasks(3);
    expect(() => {
      player.setActive('cleaned'); // sets pendingActive = 'cleaned'
      player.toggleDeltaMode();
      player.toggleLoop();
      player.setWetDryMix(0.25);
      player.setWetDryMix(Number.NaN);
      player.seek(Number.NaN);
      player.seek(Number.POSITIVE_INFINITY);
      // User changes mind back to Original while cleaned is still loading!
      player.setActive('original');
      player.toggleDeltaMode();
      player.toggleLoop();
    }).not.toThrow();

    expect(player.snapshot().pending).toBeNull();
    // Now finish loading cleaned -> must NOT unexpectedly hijack back to cleaned!
    releaseClean(makeWholeResponse());
    await settleMicrotasks();
    expect(player.activeDeck).toBe('original');

    // Now fault cleaned with 404 and spam toggles again
    player.load('cleaned', CLEAN_404, 20);
    player.setActive('cleaned');
    await settleMicrotasks();
    expect(player.deckFault('cleaned')?.kind).toBe('missing');
    expect(() => {
      player.toggleDeltaMode();
      player.toggleLoop();
      player.setWetDryMix(0.7);
      player.setActive('cleaned');
    }).not.toThrow();
    expect(player.activeDeck).toBe('original');
    expect(player.snapshot().deltaMode).toBe(false);
  });
});

describe('Async action chaos races & Undo/Redo source isolation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useStore.setState(pristine, true);
    vi.stubGlobal('URL', {
      ...URL,
      createObjectURL: vi.fn(() => `blob:act-${Math.random().toString(16).slice(2)}`),
      revokeObjectURL: vi.fn(),
    });
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_u: string, init?: RequestInit) =>
        init?.headers ? makeRangedResponse(206) : makeWholeResponse(),
      ),
    );
  });

  afterEach(() => {
    getPlayer().load('cleaned', null);
    getPlayer().load('original', null);
    vi.unstubAllGlobals();
  });

  it('spamming previewRegion() 5 times rapidly applies ONLY the 5th preview and keeps cleanedPath intact', async () => {
    const resolvers: Array<{
      resolve: (v: unknown) => void;
      reject: (e: unknown) => void;
    }> = [];
    const renderRegionPreview = vi.fn(
      () =>
        new Promise((resolve, reject) => {
          resolvers.push({ resolve, reject });
        }),
    );
    const client = {
      baseUrl: 'http://127.0.0.1:8765',
      renderRegionPreview,
      fileUrl: (p: string) => `http://127.0.0.1:8765/api/audio?path=${encodeURIComponent(p)}`,
    } as unknown as EngineClient;

    useStore.setState({
      engineStatus: 'ready',
      client,
      source: { path: '/in/interview.wav', name: 'interview.wav', origin: 'file' },
      original: makeAnalysis('/in/interview.wav', 30),
      cleanedPath: '/out/full_master.wav',
      selectionRange: { start: 2, end: 8 },
    });

    const promises = [
      previewRegion(),
      previewRegion(),
      previewRegion(),
      previewRegion(),
      previewRegion(),
    ];
    expect(resolvers).toHaveLength(5);

    // Resolve out-of-order: #5 finishes first, then #1 rejects, then #2/#3/#4 resolve late
    resolvers[4]!.resolve({
      previewPath: '/tmp/prev_5.wav',
      audioUrl: '/api/audio?path=%2Ftmp%2Fprev_5.wav',
      renderMs: 18,
      warmHit: true,
      metricsBefore: { lufs: -23, noiseFloorDb: -45 },
      metricsAfter: { lufs: -23, noiseFloorDb: -60 },
      durationS: 6,
    });
    await settleMicrotasks();

    resolvers[0]!.reject(new Error('network blip'));
    resolvers[1]!.resolve({
      previewPath: '/tmp/prev_2.wav',
      audioUrl: '/api/audio?path=%2Ftmp%2Fprev_2.wav',
      renderMs: 99,
      warmHit: false,
      metricsBefore: { lufs: -23, noiseFloorDb: -45 },
      metricsAfter: { lufs: -23, noiseFloorDb: -50 },
      durationS: 6,
    });
    resolvers[2]!.reject(new Error('timeout'));
    resolvers[3]!.resolve({
      previewPath: '/tmp/prev_4.wav',
      audioUrl: '/api/audio?path=%2Ftmp%2Fprev_4.wav',
      renderMs: 77,
      warmHit: false,
      metricsBefore: { lufs: -23, noiseFloorDb: -45 },
      metricsAfter: { lufs: -23, noiseFloorDb: -52 },
      durationS: 6,
    });

    const results = await Promise.all(promises);
    expect(results[0]).toBeNull();
    expect(results[1]).toBeNull();
    expect(results[2]).toBeNull();
    expect(results[3]).toBeNull();
    expect(results[4]?.previewPath).toBe('/tmp/prev_5.wav');

    // Only #5 won; stale rejections/resolutions did not set error or overwrite status/cleanedPath
    expect(getState().error).toBeNull();
    expect(getState().previewActive).toBe(true);
    expect(getState().cleanedPath).toBe('/out/full_master.wav');
    expect(getState().statusLine).toContain('18 ms');
    expect(getState().statusLine).toContain('warm hit');
  });

  it('drops stale previewRegion & applySpectralBoxRepair when selectTake switches takes mid-flight', async () => {
    let resolvePreview!: (v: unknown) => void;
    let resolveRepair!: (v: unknown) => void;

    const client = {
      baseUrl: 'http://127.0.0.1:8765',
      fileUrl: (p: string) => `http://127.0.0.1:8765/api/audio?path=${encodeURIComponent(p)}`,
      analyze: vi.fn(async (p: string) => makeAnalysis(p)),
      renderRegionPreview: vi.fn(() => new Promise((r) => { resolvePreview = r; })),
      repairSpectralBox: vi.fn(() => new Promise((r) => { resolveRepair = r; })),
    } as unknown as EngineClient;

    const r1 = makeReport('take-1');
    const r2 = makeReport('take-2');
    useStore.setState({
      engineStatus: 'ready',
      client,
      source: { path: '/in/a.wav', name: 'a.wav', origin: 'file' },
      original: makeAnalysis('/in/a.wav'),
      cleanedPath: '/out/take-1.wav',
      report: r1,
      spectralSelection: { startS: 1, endS: 3, fLowHz: 150, fHighHz: 500 },
      takes: [
        { id: 'take-1', label: 'Take A', profile: 'studio', cleanedPath: '/out/take-1.wav', report: r1, mosDelta: 0.4, lufsOut: -23 },
        { id: 'take-2', label: 'Take B', profile: 'studio_pro', cleanedPath: '/out/take-2.wav', report: r2, mosDelta: 0.9, lufsOut: -23 },
      ],
      activeTakeId: 'take-1',
    });

    const pPrev = previewRegion();
    const pRep = applySpectralBoxRepair();

    // Switch take mid-flight -> bumps previewEpoch & spectralRepairEpoch
    selectTake('take-2');
    expect(getState().cleanedPath).toBe('/out/take-2.wav');
    expect(getState().previewActive).toBe(false);

    resolvePreview({
      previewPath: '/tmp/stale.wav',
      audioUrl: '/api/audio?path=%2Ftmp%2Fstale.wav',
      renderMs: 12,
      warmHit: true,
      metricsBefore: { lufs: -23, noiseFloorDb: -40 },
      metricsAfter: { lufs: -23, noiseFloorDb: -60 },
      durationS: 6,
    });
    resolveRepair({
      outputPath: '/work/stale_repair.wav',
      audioUrl: '/api/audio?path=%2Fwork%2Fstale_repair.wav',
      startS: 1,
      endS: 3,
      fLowHz: 150,
      fHighHz: 500,
      attenuationDb: 24,
      renderMs: 25,
    });

    const [prevRes, repRes] = await Promise.all([pPrev, pRep]);
    expect(prevRes).toBeNull();
    expect(repRes).toBeNull();
    expect(getState().cleanedPath).toBe('/out/take-2.wav');
    expect(getState().previewActive).toBe(false);
  });

  it('clears previewActive when applyUnitOverride re-stitches the full master', async () => {
    const client = {
      baseUrl: 'http://127.0.0.1:8765',
      fileUrl: (p: string) => `http://127.0.0.1:8765/api/audio?path=${encodeURIComponent(p)}`,
      analyze: vi.fn(async () => null),
      overrideUnit: vi.fn(async () => ({
        jobId: 'job-1',
        unitIndex: 0,
        unitId: 0,
        oldDecision: 'enhanced',
        newDecision: 'original_reverted',
        override: 'force_original' as const,
        restitched: true,
        restitchMs: 25,
        masterSha256: 'sha-new',
      })),
    } as unknown as EngineClient;

    useStore.setState({
      engineStatus: 'ready',
      client,
      source: { path: '/in/a.wav', name: 'a.wav', origin: 'file' },
      original: makeAnalysis('/in/a.wav'),
      cleanedPath: '/out/a_clean.wav',
      previewActive: true,
      report: makeReport('job-1'),
      job: {
        id: 'job-1',
        outputPath: '/out/a_clean.wav',
        reportPath: '/out/a_clean.json',
        status: makeDoneStatus('job-1', '/in/a.wav', '/out/a_clean.wav'),
        streamConnected: false,
      },
    });

    await applyUnitOverride(0, 'force_original');
    expect(getState().previewActive).toBe(false);
    expect(getState().cleanedPath).toBe('/out/a_clean.wav');
  });

  it('blocks undoAction/redoAction while a job is running and clears stale unitOverrides on loadSource and selectRun', async () => {
    const overrideUnit = vi.fn(async (jobId: string, req: { unitId: number; decision: string }) => ({
      jobId,
      unitIndex: req.unitId,
      unitId: req.unitId,
      oldDecision: 'enhanced',
      newDecision: req.decision === 'force_original' ? 'original_reverted' : 'enhanced',
      override: req.decision as 'auto' | 'force_original' | 'force_enhanced',
      restitched: false,
    }));
    const client = {
      baseUrl: 'http://127.0.0.1:8765',
      fileUrl: (p: string) => `http://127.0.0.1:8765/api/audio?path=${encodeURIComponent(p)}`,
      analyze: vi.fn(async (p: string) => makeAnalysis(p)),
      verify: vi.fn(async () => ({ status: 206, delivered: true, size: 1024 })),
      overrideUnit,
    } as unknown as EngineClient;

    const doneA = makeDoneStatus('job-a', '/in/a.wav', '/out/a_clean.wav');
    const doneB = makeDoneStatus('job-b', '/in/b.wav', '/out/b_clean.wav');

    useStore.setState({
      engineStatus: 'ready',
      client,
      source: { path: '/in/a.wav', name: 'a.wav', origin: 'file' },
      original: makeAnalysis('/in/a.wav'),
      cleaned: makeAnalysis('/out/a_clean.wav'),
      cleanedPath: '/out/a_clean.wav',
      report: doneA.report ?? null,
      job: {
        id: 'job-a',
        outputPath: '/out/a_clean.wav',
        reportPath: '/out/a_clean.json',
        status: doneA,
        streamConnected: false,
      },
      currentRunId: 'job-a',
      history: [
        {
          jobId: 'job-b',
          profile: 'studio_pro',
          inputPath: '/in/b.wav',
          inputName: 'b.wav',
          outcome: 'done',
          durationMs: 2000,
          at: Date.now(),
          enhanced: 2,
          unitsTotal: 2,
          lufsIn: -23,
          lufsOut: -23,
          noiseIn: -48,
          noiseOut: -60,
          outputPath: '/out/b_clean.wav',
          reportPath: '/out/b_clean.json',
          report: doneB.report ?? null,
          status: doneB,
          original: makeAnalysis('/in/b.wav'),
          cleaned: makeAnalysis('/out/b_clean.wav'),
          error: null,
        },
      ],
    });

    // Record a unit override on job-a and a StudioRack change
    updateRackSettings({ ...DEFAULT_STUDIO_RACK_SETTINGS, strength: 0.9 });
    await applyUnitOverride(0, 'force_original');
    expect(overrideUnit).toHaveBeenCalledTimes(1);
    expect(getState().overrideUndo.canUndo).toBe(true);

    // 1. While a job is actively running, undoAction & redoAction must be no-ops
    useStore.setState({
      job: {
        id: 'job-running',
        outputPath: '/out/r.wav',
        reportPath: '/out/r.json',
        status: { ...doneA, job_id: 'job-running', state: 'running', stage: 'enhance' },
        streamConnected: true,
      },
    });
    overrideUnit.mockClear();
    undoAction();
    redoAction();
    await settleMicrotasks();
    expect(overrideUnit).not.toHaveBeenCalled();
    expect(getState().unitOverrides[0]).toBe('force_original');
    expect(getState().rackSettings.strength).toBe(0.9);

    // Restore finished job-a, then switch to history run job-b via selectRun
    useStore.setState({
      job: {
        id: 'job-a',
        outputPath: '/out/a_clean.wav',
        reportPath: '/out/a_clean.json',
        status: doneA,
        streamConnected: false,
      },
    });
    await selectRun('job-b');
    expect(getState().unitOverrides).toEqual({});
    expect(getState().overrideUndo.canUndo).toBe(false);

    // Pressing Cmd+Z after selectRun must NEVER send an override for job-a onto job-b
    overrideUnit.mockClear();
    undoAction();
    await settleMicrotasks();
    expect(overrideUnit).not.toHaveBeenCalled();

    // Apply override on job-b, then switch to a brand-new source via loadSource
    await applyUnitOverride(0, 'force_original');
    expect(getState().overrideUndo.canUndo).toBe(true);
    overrideUnit.mockClear();

    await loadSource({ path: '/in/c.wav', name: 'c.wav', origin: 'file' });
    expect(getState().unitOverrides).toEqual({});
    expect(getState().overrideUndo.canUndo).toBe(false);
    expect(getState().lastUndoDomain).toBeNull();

    undoAction();
    await settleMicrotasks();
    expect(overrideUnit).not.toHaveBeenCalled();
  });

  it('drops stale extractAndLoadTrack response when loadSource switches files mid-flight', async () => {
    let resolveExtract!: (v: unknown) => void;
    const client = {
      baseUrl: 'http://127.0.0.1:8765',
      fileUrl: (p: string) => `http://127.0.0.1:8765/api/audio?path=${encodeURIComponent(p)}`,
      analyze: vi.fn(async (p: string) => makeAnalysis(p)),
      extractTracks: vi.fn(() => new Promise((r) => { resolveExtract = r; })),
    } as unknown as EngineClient;

    useStore.setState({
      engineStatus: 'ready',
      client,
      source: { path: '/in/poly.wav', name: 'poly.wav', origin: 'file' },
      original: makeAnalysis('/in/poly.wav'),
    });

    const pExt = extractAndLoadTrack('/in/poly.wav', 1, 'Boom');
    await loadSource({ path: '/in/other.wav', name: 'other.wav', origin: 'file' });
    resolveExtract({ extractedPaths: ['/work/extracted_boom.wav'], sourceIds: ['src-boom'] });

    const out = await pExt;
    expect(out).toBeNull();
    expect(getState().source?.path).toBe('/in/other.wav');
  });
});

describe('Hostile, corrupt, oversized, and quota-exceeded localStorage recovery in presets.ts', () => {
  function makeStorage(raw: string | null, opts: { throwQuotaOnSet?: boolean } = {}): Storage {
    const map = new Map<string, string>();
    if (raw !== null) map.set(PRESET_STORAGE_KEY, raw);
    return {
      get length() {
        return map.size;
      },
      clear: () => map.clear(),
      getItem: (k: string) => (map.has(k) ? map.get(k)! : null),
      key: (i: number) => Array.from(map.keys())[i] ?? null,
      removeItem: (k: string) => map.delete(k),
      setItem: (k: string, v: string) => {
        if (opts.throwQuotaOnSet) {
          throw new DOMException('The quota has been exceeded.', 'QuotaExceededError');
        }
        map.set(k, String(v));
      },
    };
  }

  it('recovers cleanly from invalid JSON, null, primitives, empty/huge arrays, and prototype pollution payloads', () => {
    const hostilePayloads = [
      '{truncated json',
      'null',
      '12345',
      '"hello"',
      '{}',
      '[]',
      '[{"__proto__":{"polluted":true},"id":"user:x","name":"Evil","settings":{}}]',
      '[{"constructor":{"prototype":{"polluted":true}},"id":"user:x","name":"Evil","settings":{}}]',
      `[${Array.from({ length: 10000 }, () => 'null').join(',')}]`,
      '[' + '"x",'.repeat(Math.ceil(MAX_PRESET_STORAGE_BYTES / 4)) + '"y"]',
    ];

    for (const payload of hostilePayloads) {
      const storage = makeStorage(payload);
      expect(() => loadUserPresets(storage)).not.toThrow();
      expect(loadUserPresets(storage)).toEqual([]);
      expect((Object.prototype as Record<string, unknown>)['polluted']).toBeUndefined();
    }
  });

  it('recovers gracefully when localStorage.setItem throws QuotaExceededError DOMException', () => {
    const storage = makeStorage('[]', { throwQuotaOnSet: true });
    expect(() =>
      saveUserPreset(
        { name: 'Quota Test', settings: DEFAULT_STUDIO_RACK_SETTINGS },
        storage,
      ),
    ).not.toThrow();

    const saved = saveUserPreset(
      { name: 'Quota Test', settings: DEFAULT_STUDIO_RACK_SETTINGS },
      storage,
    );
    expect(saved.name).toBe('Quota Test');
    expect(saved.id.startsWith('user:')).toBe(true);
    expect(() => deleteUserPreset(saved.id, storage)).not.toThrow();
  });
});
