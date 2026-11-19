// state/proActions.test.ts — Wave 2 StudioRack undo/redo, manual payload wiring, takes, delivery export.

import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { EngineClient } from '../api/client';
import type { AudioAnalysis, HawaVoCleanReport } from '../api/types';
import { DEFAULT_STUDIO_RACK_SETTINGS } from './presets';
import { getState, useStore } from './store';
import {
  exportWithDelivery,
  pinTake,
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
  seek: vi.fn(),
  play: vi.fn(async () => undefined),
  pause: vi.fn(),
  setActive: vi.fn(),
  setLoudnessMatch: vi.fn(),
  hasDeck: vi.fn(() => true),
  gainOffsetDb: 0,
}));

vi.mock('../audio/player', () => ({ getPlayer: () => mockPlayer }));
vi.mock('../bridge', () => ({
  getBridge: () => ({ host: 'web', engine: { getEndpoint: async () => ({ baseUrl: '' }) } }),
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

    // Clear job and modify rack settings -> attaches strength, ambienceFloorDb, deliveryPreset, proRepair
    useStore.setState({ job: null });
    updateRackSettings({
      ...DEFAULT_STUDIO_RACK_SETTINGS,
      strength: 0.85,
      ambienceFloorDb: -38,
      deliveryPreset: 'podcast',
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
