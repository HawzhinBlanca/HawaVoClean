// state/proActions.test.ts — Wave 2 StudioRack undo/redo, manual payload wiring, takes, delivery export.

import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { EngineClient } from '../api/client';
import type { AudioAnalysis, HawaVoCleanReport } from '../api/types';
import { DEFAULT_STUDIO_RACK_SETTINGS } from './presets';
import { getState, useStore } from './store';
import {
  applySpectralBoxRepair,
  applyUnitOverride,
  cleanTrackWithMarkersInResolve,
  exportWithDelivery,
  extractAndLoadTrack,
  pinTake,
  previewRegion,
  redoAction,
  removeTake,
  selectTake,
  startJob,
  undoAction,
  updateRackSettings,
} from './actions';

const mockPlayer = vi.hoisted(() => ({
  time: 0,
  playing: false,
  duration: 12,
  load: vi.fn(),
  claimOnly: vi.fn(),
  seek: vi.fn(),
  play: vi.fn(async () => undefined),
  pause: vi.fn(),
  setActive: vi.fn(),
  setLoudnessMatch: vi.fn(),
  setLoopRange: vi.fn(),
  setLoopEnabled: vi.fn(),
  hasDeck: vi.fn(() => true),
  gainOffsetDb: 0,
}));

const mockBridge = vi.hoisted(() => ({
  current: {
    host: 'web' as 'web' | 'resolve' | 'electron',
    engine: { getEndpoint: async () => ({ baseUrl: '' }) },
    resolve: undefined as
      | {
          batchCleanTrack?: ReturnType<typeof vi.fn>;
        }
      | undefined,
  },
}));

vi.mock('../audio/player', () => ({ getPlayer: () => mockPlayer }));
vi.mock('../bridge', () => ({
  getBridge: () => mockBridge.current,
}));
vi.mock('../api/sse', () => ({
  followJob: vi.fn(() => () => undefined),
  followBatch: vi.fn(() => () => undefined),
  isTerminal: (s: string) => ['done', 'failed', 'cancelled', 'interrupted'].includes(s),
}));

const pristine = useStore.getState();

function makeAnalysis(path = '/a.wav'): AudioAnalysis {
  return {
    path,
    duration_s: 12,
    sample_rate: 48000,
    channels: 1,
    peaks: { min: [-0.5], max: [0.5] },
    rms_db: [-20],
    spectrum: { freqs_hz: [100], db: [-30] },
    loudness: { integrated_lufs: -23.0, true_peak_dbtp: -1.0 },
    noise_floor_db: -48.0,
  };
}

function makeReport(jobId: string): HawaVoCleanReport {
  return {
    job_id: jobId,
    config_hash: 'c',
    input: { path: '/a.wav', sha256: 'x', sample_rate: 48000, channels: 1, samples: 1, duration_s: 12 },
    output: { path: `/out/${jobId}.wav`, sha256: 'y', sample_rate: 48000, channels: 1, samples: 1, duration_s: 12 },
    core: { id: 'studio_pro', algorithm: 'x', params_hash: 'p' },
    guard: { id: 'g', probe_hash: 'p', calibration_id: 'c' },
    environment: {},
    summary: { units_total: 4, enhanced: 4 },
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockPlayer.time = 0;
  mockPlayer.playing = false;
  mockBridge.current = {
    host: 'web',
    engine: { getEndpoint: async () => ({ baseUrl: '' }) },
    resolve: undefined,
  };
  useStore.setState(pristine, true);
});

describe('Wave 2 StudioRack actions & undo/redo', () => {
  it('updates rackSettings with Undo/Redo support and syncs batchNamingTemplate', () => {
    const next = {
      ...DEFAULT_STUDIO_RACK_SETTINGS,
      strength: 0.9,
      ambienceFloorDb: -55,
      namingTemplate: '{name}_broadcast',
    };
    updateRackSettings(next);

    expect(getState().rackSettings.strength).toBe(0.9);
    expect(getState().rackSettings.ambienceFloorDb).toBe(-55);
    expect(getState().batchNamingTemplate).toBe('{name}_broadcast');
    expect(getState().rackUndo.canUndo).toBe(true);

    undoAction();
    expect(getState().rackSettings.strength).toBe(DEFAULT_STUDIO_RACK_SETTINGS.strength);
    expect(getState().rackUndo.canRedo).toBe(true);

    redoAction();
    expect(getState().rackSettings.strength).toBe(0.9);
  });

  it('attaches optional StudioRack parameters to manual startJob only when modified', async () => {
    const createV1Jobs = vi.fn(async () => ({
      schemaVersion: 1 as const,
      jobs: [{ jobId: 'j1', sourceId: 'src1', outputPath: '/out/a.wav', reportPath: '/out/a.json' }],
    }));
    const client = {
      createV1Jobs,
      fileUrl: (p: string) => `/api/audio?path=${encodeURIComponent(p)}`,
    } as unknown as EngineClient;

    useStore.setState({
      engineStatus: 'ready',
      client,
      source: { path: '/a.wav', name: 'a.wav', sourceId: 'src1', origin: 'file' },
      original: makeAnalysis(),
      profile: 'studio',
      mode: 'natural',
    });

    // Untouched rack -> clean legacy manual payload
    await startJob();
    expect(createV1Jobs).toHaveBeenCalledWith(
      expect.objectContaining({
        strategy: { kind: 'manual', route: 'studio', allow_generative_reconstruction: false },
      }),
    );

    // Clear job and modify rack settings -> attaches strength, ambienceFloorDb, deliveryPreset, stereoMode, proRepair
    useStore.setState({ job: null });
    updateRackSettings({
      ...DEFAULT_STUDIO_RACK_SETTINGS,
      strength: 0.85,
      ambienceFloorDb: -38,
      deliveryPreset: 'podcast',
      stereoMode: 'true_stereo_ms',
      deplosive: true,
      mouthDeclick: false,
    });
    await startJob();
    expect(createV1Jobs).toHaveBeenLastCalledWith(
      expect.objectContaining({
        strategy: expect.objectContaining({
          kind: 'manual',
          route: 'studio',
          allow_generative_reconstruction: false,
          strength: 0.85,
          ambienceFloorDb: -38,
          deliveryPreset: 'podcast',
          stereoMode: 'true_stereo_ms',
          proRepair: expect.objectContaining({
            deplosive: true,
            mouth_declick: false,
          }),
        }),
      }),
    );
  });
});

describe('Wave 2 Multi-Take Switcher actions', () => {
  it('selects, pins, and removes takes and loads selected take onto cleaned deck', () => {
    const client = {
      fileUrl: (p: string) => `http://127.0.0.1:8765/api/audio?path=${encodeURIComponent(p)}`,
      analyze: vi.fn(async () => makeAnalysis('/out/t2.wav')),
    } as unknown as EngineClient;

    const r1 = makeReport('t1');
    const r2 = makeReport('t2');
    useStore.setState({
      engineStatus: 'ready',
      client,
      original: makeAnalysis(),
      takes: [
        { id: 't1', label: 'Take A', profile: 'studio', cleanedPath: '/out/t1.wav', report: r1, mosDelta: 0.5, lufsOut: -23 },
        { id: 't2', label: 'Take B', profile: 'studio_pro', cleanedPath: '/out/t2.wav', report: r2, mosDelta: 1.2, lufsOut: -22.5 },
      ],
      activeTakeId: 't1',
    });

    selectTake('t2');
    expect(getState().activeTakeId).toBe('t2');
    expect(getState().cleanedPath).toBe('/out/t2.wav');
    expect(getState().report?.job_id).toBe('t2');
    expect(mockPlayer.load).toHaveBeenCalledWith(
      'cleaned',
      expect.stringContaining('t2.wav'),
      12,
    );
    expect(mockPlayer.setActive).toHaveBeenCalledWith('cleaned');

    // Pin protects from removal
    pinTake('t2');
    expect(getState().takes.find((t) => t.id === 't2')?.pinned).toBe(true);
    removeTake('t2');
    expect(getState().takes).toHaveLength(2);

    // Unpin allows removal
    pinTake('t2');
    removeTake('t2');
    expect(getState().takes).toHaveLength(1);
    expect(getState().activeTakeId).toBe('t1');
  });
});

describe('Wave 2 exportWithDelivery action', () => {
  it('calls client.exportDelivery when format or deliveryPreset is non-default', async () => {
    const exportDelivery = vi.fn(async () => ({
      outputPath: '/out/clip_mux.mp4',
      format: 'video_mux' as const,
      deliveryPreset: 'ebu_r128' as const,
      lraLu: 4.5,
      integratedLufs: -23.0,
      truePeakDbtp: -1.0,
    }));
    const client = {
      exportDelivery,
      fileUrl: (p: string) => `http://127.0.0.1:8765/api/audio?path=${encodeURIComponent(p)}`,
    } as unknown as EngineClient;

    useStore.setState({
      engineStatus: 'ready',
      client,
      source: { path: '/in/interview.mp4', name: 'interview.mp4', origin: 'file' },
      cleanedPath: '/out/interview_clean.wav',
    });

    const res = await exportWithDelivery('video_mux');
    expect(exportDelivery).toHaveBeenCalledWith(
      expect.objectContaining({
        masterPath: '/out/interview_clean.wav',
        format: 'video_mux',
        deliveryPreset: 'ebu_r128',
        sourceMediaPath: '/in/interview.mp4',
        preserveMetadata: true,
      }),
    );
    expect(res?.outputPath).toBe('/out/clip_mux.mp4');
    expect(getState().statusLine).toContain('clip_mux.mp4');
  });
});

describe('Last-Mile Pro Bridges actions', () => {
  it('previewRegion uses selectionRange (clamped to <=15s), loops cleaned deck, and reports warmHit + renderMs', async () => {
    const renderRegionPreview = vi.fn(async () => ({
      previewPath: '/tmp/prev_1.wav',
      audioUrl: '/api/audio?path=%2Ftmp%2Fprev_1.wav',
      renderMs: 42.6,
      warmHit: true,
      metricsBefore: { lufs: -22, noiseFloorDb: -40 },
      metricsAfter: { lufs: -22, noiseFloorDb: -58 },
      durationS: 15,
    }));
    const client = {
      baseUrl: 'http://127.0.0.1:8765',
      renderRegionPreview,
      fileUrl: (p: string) => `http://127.0.0.1:8765/api/audio?path=${encodeURIComponent(p)}`,
    } as unknown as EngineClient;

    useStore.setState({
      engineStatus: 'ready',
      client,
      source: { path: '/in/clip.wav', name: 'clip.wav', origin: 'file' },
      original: makeAnalysis('/in/clip.wav'),
      profile: 'studio_pro',
      selectionRange: { start: 2, end: 25 }, // 23s > 15s cap, and clipped to duration 12s
      rackSettings: {
        ...DEFAULT_STUDIO_RACK_SETTINGS,
        strength: 0.85,
        ambienceFloorDb: -45,
      },
    });
    // Increase clip duration to 40s to test the 15s clamp and preserve existing master cleanedPath
    useStore.setState({
      original: { ...makeAnalysis('/in/clip.wav'), duration_s: 40 },
      cleanedPath: '/out/full_master.wav',
    });

    const res = await previewRegion();
    expect(renderRegionPreview).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/in/clip.wav',
        startS: 2,
        endS: 17,
        profile: 'studio_pro',
        strength: 0.85,
        ambienceFloorDb: -45,
        hpfEnabled: true,
        hpfCutoffHz: 80,
        deliveryPreset: 'ebu_r128',
        proRepair: expect.objectContaining({
          deplosive: true,
          mouth_declick: true,
        }),
      }),
    );
    expect(mockPlayer.load).toHaveBeenCalledWith(
      'cleaned',
      'http://127.0.0.1:8765/api/audio?path=%2Ftmp%2Fprev_1.wav',
      15,
      2,
    );
    expect(mockPlayer.setLoopRange).toHaveBeenCalledWith([2, 17]);
    expect(mockPlayer.setLoopEnabled).toHaveBeenCalledWith(true);
    expect(mockPlayer.seek).toHaveBeenCalledWith(2);
    expect(mockPlayer.setActive).toHaveBeenCalledWith('cleaned');
    // Must NEVER clobber the full-master cleanedPath with a short region preview
    expect(getState().cleanedPath).toBe('/out/full_master.wav');
    expect(getState().previewActive).toBe(true);
    expect(getState().abMode).toBe('cleaned');
    expect(getState().statusLine).toContain('43 ms');
    expect(getState().statusLine).toContain('warm hit');
    expect(res?.warmHit).toBe(true);
  });

  it('previewRegion falls back to [player.time, min(duration, player.time + 6)] when no selectionRange is set and fails soft when offline', async () => {
    const renderRegionPreview = vi.fn(async () => ({
      previewPath: '/tmp/prev_2.wav',
      audioUrl: '/api/audio?path=%2Ftmp%2Fprev_2.wav',
      renderMs: 118,
      warmHit: false,
      metricsBefore: { lufs: -22, noiseFloorDb: -40 },
      metricsAfter: { lufs: -22, noiseFloorDb: -56 },
      durationS: 3,
    }));
    const client = {
      baseUrl: '',
      renderRegionPreview,
      fileUrl: (p: string) => `/api/audio?path=${encodeURIComponent(p)}`,
    } as unknown as EngineClient;

    mockPlayer.time = 9;
    useStore.setState({
      engineStatus: 'ready',
      client,
      source: { path: '/in/clip.wav', name: 'clip.wav', origin: 'file' },
      original: makeAnalysis('/in/clip.wav'), // duration_s = 12
      selectionRange: null,
    });

    await previewRegion();
    expect(renderRegionPreview).toHaveBeenCalledWith(
      expect.objectContaining({ path: '/in/clip.wav', startS: 9, endS: 12 }),
    );
    expect(mockPlayer.setLoopRange).toHaveBeenCalledWith([9, 12]);
    expect(getState().statusLine).toContain('cold render');

    // Offline soft handling
    useStore.setState({ engineStatus: 'offline' });
    const offlineRes = await previewRegion();
    expect(offlineRes).toBeNull();
    expect(getState().statusLine.toLowerCase()).toContain('offline');
  });

  it('applySpectralBoxRepair repairs the active audio path and loads repaired audio onto cleaned deck', async () => {
    const repairSpectralBox = vi.fn(async () => ({
      outputPath: '/work/repaired.wav',
      audioUrl: '/api/audio?path=%2Fwork%2Frepaired.wav',
      startS: 4.2,
      endS: 6.1,
      fLowHz: 140,
      fHighHz: 220,
      attenuationDb: 24,
      renderMs: 55.2,
    }));
    const client = {
      baseUrl: 'http://127.0.0.1:8765',
      repairSpectralBox,
      fileUrl: (p: string) => `http://127.0.0.1:8765/api/audio?path=${encodeURIComponent(p)}`,
    } as unknown as EngineClient;

    useStore.setState({
      engineStatus: 'ready',
      client,
      source: { path: '/in/clip.wav', name: 'clip.wav', origin: 'file' },
      original: makeAnalysis('/in/clip.wav'),
      abMode: 'original',
      spectralSelection: { startS: 4.2, endS: 6.1, fLowHz: 140, fHighHz: 220 },
    });

    const res = await applySpectralBoxRepair();
    expect(repairSpectralBox).toHaveBeenCalledWith({
      path: '/in/clip.wav',
      startS: 4.2,
      endS: 6.1,
      fLowHz: 140,
      fHighHz: 220,
      attenuationDb: 24,
    });
    expect(mockPlayer.load).toHaveBeenCalledWith(
      'cleaned',
      'http://127.0.0.1:8765/api/audio?path=%2Fwork%2Frepaired.wav',
      12,
    );
    expect(mockPlayer.setActive).toHaveBeenCalledWith('cleaned');
    expect(getState().abMode).toBe('cleaned');
    expect(getState().cleanedPath).toBe('/work/repaired.wav');
    expect(getState().statusLine).toContain('140–220 Hz');
    expect(res?.outputPath).toBe('/work/repaired.wav');
  });

  it('extractAndLoadTrack calls extractTracks, forwards sourceId, and loads the extracted mono WAV onto the deck', async () => {
    const extractTracks = vi.fn(async () => ({
      extractedPaths: ['/work/extracted_tracks/poly/ch02_Lav.wav'],
      sourceIds: ['src-extracted-lav-2'],
    }));
    const analyze = vi.fn(async () => makeAnalysis('/work/extracted_tracks/poly/ch02_Lav.wav'));
    const client = {
      baseUrl: '',
      extractTracks,
      analyze,
      fileUrl: (p: string) => `/api/audio?path=${encodeURIComponent(p)}`,
    } as unknown as EngineClient;

    useStore.setState({
      engineStatus: 'ready',
      client,
      source: { path: '/field/poly.wav', name: 'poly.wav', origin: 'file' },
      original: { ...makeAnalysis('/field/poly.wav'), channels: 6 },
    });

    const out = await extractAndLoadTrack('/field/poly.wav', 1, 'Lav');
    expect(extractTracks).toHaveBeenCalledWith('/field/poly.wav', [1]);
    expect(out).toBe('/work/extracted_tracks/poly/ch02_Lav.wav');
    expect(getState().source?.path).toBe('/work/extracted_tracks/poly/ch02_Lav.wav');
    expect(getState().source?.name).toBe('ch02_Lav.wav');
    expect(getState().source?.sourceId).toBe('src-extracted-lav-2');
  });

  it('drops stale async responses in previewRegion, applySpectralBoxRepair, applyUnitOverride, and extractAndLoadTrack when clip/job changes mid-flight', async () => {
    let resolvePreview!: (v: unknown) => void;
    let resolveRepair!: (v: unknown) => void;
    let resolveOverride!: (v: unknown) => void;
    let resolveExtract!: (v: unknown) => void;

    const client = {
      baseUrl: '',
      fileUrl: (p: string) => `/api/audio?path=${encodeURIComponent(p)}`,
      analyze: vi.fn(async () => makeAnalysis('/in/new_clip.wav')),
      renderRegionPreview: vi.fn(
        () =>
          new Promise((r) => {
            resolvePreview = r;
          }),
      ),
      repairSpectralBox: vi.fn(
        () =>
          new Promise((r) => {
            resolveRepair = r;
          }),
      ),
      overrideUnit: vi.fn(
        () =>
          new Promise((r) => {
            resolveOverride = r;
          }),
      ),
      extractTracks: vi.fn(
        () =>
          new Promise((r) => {
            resolveExtract = r;
          }),
      ),
    } as unknown as EngineClient;

    useStore.setState({
      engineStatus: 'ready',
      client,
      source: { path: '/in/clip_a.wav', name: 'clip_a.wav', origin: 'file' },
      original: makeAnalysis('/in/clip_a.wav'),
      cleanedPath: '/out/clip_a_clean.wav',
      report: makeReport('job-a'),
      job: {
        id: 'job-a',
        outputPath: '/out/clip_a_clean.wav',
        reportPath: '/out/clip_a.json',
        status: null,
        streamConnected: false,
      },
      spectralSelection: { startS: 1, endS: 3, fLowHz: 200, fHighHz: 800 },
    });

    const pPrev = previewRegion();
    const pRep = applySpectralBoxRepair();
    const pOv = applyUnitOverride(1, 'force_original');
    const pExt = extractAndLoadTrack('/in/clip_a.wav', 0, 'Boom');

    // User switches to a different clip & job while all 4 requests are in flight
    useStore.setState({
      source: { path: '/in/clip_b.wav', name: 'clip_b.wav', origin: 'file' },
      original: makeAnalysis('/in/clip_b.wav'),
      cleanedPath: '/out/clip_b_clean.wav',
      report: makeReport('job-b'),
      job: {
        id: 'job-b',
        outputPath: '/out/clip_b_clean.wav',
        reportPath: '/out/clip_b.json',
        status: null,
        streamConnected: false,
      },
    });
    vi.clearAllMocks();

    resolvePreview({
      previewPath: '/tmp/stale_prev.wav',
      audioUrl: '/api/audio?path=%2Ftmp%2Fstale_prev.wav',
      renderMs: 20,
      warmHit: true,
      metricsBefore: { lufs: -22, noiseFloorDb: -40 },
      metricsAfter: { lufs: -22, noiseFloorDb: -56 },
      durationS: 6,
    });
    resolveRepair({
      outputPath: '/work/stale_repaired.wav',
      audioUrl: '/api/audio?path=%2Fwork%2Fstale_repaired.wav',
      startS: 1,
      endS: 3,
      fLowHz: 200,
      fHighHz: 800,
      attenuationDb: 24,
      renderMs: 30,
    });
    resolveOverride({
      jobId: 'job-a',
      unitIndex: 1,
      unitId: 1,
      oldDecision: 'enhanced',
      newDecision: 'original_reverted',
      override: 'force_original',
      restitched: true,
      masterSha256: 'stale-sha',
    });
    resolveExtract({
      extractedPaths: ['/work/extracted_tracks/stale.wav'],
      sourceIds: ['src-stale'],
    });

    const [prevRes, repRes, , extRes] = await Promise.all([pPrev, pRep, pOv, pExt]);
    expect(prevRes).toBeNull();
    expect(repRes).toBeNull();
    expect(extRes).toBeNull();
    // Stale responses must NOT touch the new clip's player or state
    expect(mockPlayer.load).not.toHaveBeenCalled();
    expect(getState().source?.path).toBe('/in/clip_b.wav');
    expect(getState().cleanedPath).toBe('/out/clip_b_clean.wav');
    expect(getState().previewActive).toBe(false);
  });

  it('cleanTrackWithMarkersInResolve forwards cleaned path and report to bridge.resolve.batchCleanTrack', async () => {
    const batchCleanTrack = vi.fn(async () => ({
      success: true,
      trackIndex: 3,
      trackName: 'HawaVoClean',
      clipsPlaced: 1,
      markersAdded: 4,
    }));
    mockBridge.current = {
      host: 'resolve',
      engine: { getEndpoint: async () => ({ baseUrl: '' }) },
      resolve: { batchCleanTrack },
    };

    const report = makeReport('j9');
    useStore.setState({
      source: { path: '/in/clip.wav', name: 'clip.wav', origin: 'resolve', mediaId: 'm-123' },
      cleanedPath: '/out/clip_clean.wav',
      report,
      job: {
        id: 'j9',
        outputPath: '/out/clip_clean.wav',
        reportPath: '/out/clip_clean.hawavoclean.json',
        status: null,
        streamConnected: false,
      },
    });

    const res = await cleanTrackWithMarkersInResolve();
    expect(batchCleanTrack).toHaveBeenCalledWith({
      clips: [{ clipId: 'm-123', mediaId: 'm-123', startFrame: 0 }],
      cleanedByClipId: {
        'm-123': {
          filePath: '/out/clip_clean.wav',
          report,
          reportPath: '/out/clip_clean.hawavoclean.json',
        },
      },
      targetTrackName: 'HawaVoClean',
      addGuardMarkers: true,
    });
    expect(res?.markersAdded).toBe(4);
    expect(getState().statusLine).toContain('HawaVoClean');
    expect(getState().statusLine).toContain('4 guard markers');
  });
});
