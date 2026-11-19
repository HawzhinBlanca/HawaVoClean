import { describe, expect, it } from 'vitest';
import type { AudioAnalysis, AudioPlan, HawaVoCleanReport, JobStatus } from '../api/types';
import {
  ALLOWED_ASSISTANT_TRANSITIONS,
  assistantReducer,
  deriveEnginePhase,
  deriveProcessingReadiness,
  deriveWorkspacePhase,
  ENGINE_PHASE_TRANSITIONS,
  INITIAL_ASSISTANT_STATE,
  WORKSPACE_PHASE_TRANSITIONS,
} from './machines';
import { useStore } from './store';

const samplePlan: AudioPlan = {
  operation: 'clean',
  target: null,
  cleanup_profile: 'studio',
  model_size: 'large',
  quality: 'high',
  explanation: 'Clean dialogue.',
  provider: 'ollama',
  needs_clarification: false,
};

const baseAnalysis = {
  path: '/clip.wav',
  duration_s: 12,
  sample_rate: 48000,
  channels: 1,
  noise_floor_db: -50,
} as AudioAnalysis;

describe('deriveEnginePhase', () => {
  it('distinguishes connecting, ready, offline_waiting, and offline_probing', () => {
    expect(
      deriveEnginePhase({
        engineStatus: 'connecting',
        engineVersion: null,
        engineOfflineSince: null,
        engineNextProbeAt: null,
        engineProbing: false,
      }),
    ).toEqual({ phase: 'connecting' });

    expect(
      deriveEnginePhase({
        engineStatus: 'ready',
        engineVersion: '2.0.0',
        engineOfflineSince: null,
        engineNextProbeAt: null,
        engineProbing: false,
      }),
    ).toEqual({ phase: 'ready', version: '2.0.0' });

    expect(
      deriveEnginePhase({
        engineStatus: 'offline',
        engineVersion: null,
        engineOfflineSince: 1000,
        engineNextProbeAt: 2000,
        engineProbing: false,
      }),
    ).toEqual({ phase: 'offline_waiting', offlineSince: 1000, nextProbeAt: 2000 });

    expect(
      deriveEnginePhase({
        engineStatus: 'offline',
        engineVersion: null,
        engineOfflineSince: 1000,
        engineNextProbeAt: 2000,
        engineProbing: true,
      }),
    ).toEqual({ phase: 'offline_probing', offlineSince: 1000, nextProbeAt: 2000 });

    // Reconnecting after a prior outage stays in an offline phase so the banner stays mounted.
    expect(
      deriveEnginePhase({
        engineStatus: 'connecting',
        engineVersion: null,
        engineOfflineSince: 1000,
        engineNextProbeAt: 2000,
        engineProbing: false,
      }),
    ).toEqual({ phase: 'offline_waiting', offlineSince: 1000, nextProbeAt: 2000 });

    expect(Object.keys(ENGINE_PHASE_TRANSITIONS)).toEqual([
      'connecting',
      'ready',
      'offline_waiting',
      'offline_probing',
    ]);
  });
});

describe('deriveWorkspacePhase & deriveProcessingReadiness', () => {
  const pristine = useStore.getState();

  it('derives all workspace phases and blocks processing while uploading', () => {
    expect(deriveWorkspacePhase(pristine)).toBe('empty');

    const uploading = {
      ...pristine,
      engineStatus: 'ready' as const,
      source: { path: '/clip.wav', name: 'clip.wav', origin: 'upload' as const },
      original: baseAnalysis,
      upload: { name: 'clip.wav', loaded: 10, total: 100, phase: 'sending' as const },
    };
    expect(deriveWorkspacePhase(uploading)).toBe('uploading');
    expect(deriveProcessingReadiness(uploading).canStart).toBe(false);

    const analyzing = { ...uploading, upload: null, analyzing: true };
    expect(deriveWorkspacePhase(analyzing)).toBe('analyzing');
    expect(deriveProcessingReadiness(analyzing).canStart).toBe(false);

    const ready = { ...analyzing, analyzing: false };
    expect(deriveWorkspacePhase(ready)).toBe('ready');
    expect(deriveProcessingReadiness(ready).canStart).toBe(true);

    const processing = {
      ...ready,
      job: {
        id: 'j1',
        outputPath: '/out.wav',
        reportPath: '/out.json',
        status: null,
        streamConnected: false,
      },
    };
    expect(deriveWorkspacePhase(processing)).toBe('processing');
    expect(deriveProcessingReadiness(processing).canStart).toBe(false);

    const reviewed = {
      ...ready,
      cleanedPath: '/out.wav',
      report: {} as HawaVoCleanReport,
      job: {
        id: 'j1',
        outputPath: '/out.wav',
        reportPath: '/out.json',
        status: { state: 'done' } as JobStatus,
        streamConnected: false,
      },
    };
    expect(deriveWorkspacePhase(reviewed)).toBe('reviewed');

    const failed = {
      ...ready,
      job: {
        id: 'j1',
        outputPath: '/out.wav',
        reportPath: '/out.json',
        status: { state: 'failed' } as JobStatus,
        streamConnected: false,
      },
    };
    expect(deriveWorkspacePhase(failed)).toBe('failed');
    expect(deriveProcessingReadiness(failed).canStart).toBe(true);

    const interrupted = {
      ...ready,
      job: {
        id: 'j1',
        outputPath: '/out.wav',
        reportPath: '/out.json',
        status: { state: 'interrupted' } as JobStatus,
        streamConnected: false,
      },
    };
    expect(deriveWorkspacePhase(interrupted)).toBe('failed');
    expect(deriveProcessingReadiness(interrupted).running).toBe(false);
    expect(deriveProcessingReadiness(interrupted).canStart).toBe(true);
    expect(WORKSPACE_PHASE_TRANSITIONS.processing).toContain('reviewed');
  });
});

describe('assistantReducer', () => {
  it('walks idle -> planning -> planned -> applying -> planned and preserves plan on apply failure', () => {
    let state = INITIAL_ASSISTANT_STATE;
    state = assistantReducer(state, { type: 'START_PLAN' });
    expect(state).toEqual({ phase: 'planning' });

    // Duplicate START_PLAN while planning is a safe no-op.
    expect(assistantReducer(state, { type: 'START_PLAN' })).toBe(state);

    state = assistantReducer(state, { type: 'PLAN_READY', plan: samplePlan });
    expect(state).toEqual({ phase: 'planned', plan: samplePlan, applyError: null });

    state = assistantReducer(state, { type: 'START_APPLY' });
    expect(state).toEqual({ phase: 'applying', plan: samplePlan });

    // Cannot start a new plan while an apply is in flight.
    expect(assistantReducer(state, { type: 'START_PLAN' })).toBe(state);

    state = assistantReducer(state, { type: 'APPLY_FAILED', message: 'Queue full' });
    expect(state).toEqual({ phase: 'planned', plan: samplePlan, applyError: 'Queue full' });

    state = assistantReducer(state, { type: 'START_APPLY' });
    state = assistantReducer(state, { type: 'APPLY_STARTED' });
    expect(state).toEqual({ phase: 'planned', plan: samplePlan, applyError: null });

    state = assistantReducer(state, { type: 'DRAFT_CHANGED' });
    expect(state).toEqual({ phase: 'idle' });
  });

  it('handles clarification, cancellation, planning error, and atomic source-switch RESET', () => {
    const planning = assistantReducer(INITIAL_ASSISTANT_STATE, { type: 'START_PLAN' });
    expect(assistantReducer(planning, { type: 'PLAN_CLARIFIED' })).toEqual({ phase: 'idle' });
    expect(assistantReducer(planning, { type: 'PLAN_CANCELLED' })).toEqual({ phase: 'idle' });

    const err = assistantReducer(planning, { type: 'PLAN_FAILED', message: 'Offline' });
    expect(err).toEqual({ phase: 'error', message: 'Offline' });
    expect(assistantReducer(err, { type: 'DRAFT_CHANGED' })).toEqual({ phase: 'idle' });

    // Stale PLAN_READY arriving after RESET is ignored (impossible to resurrect old plan).
    const reset = assistantReducer(planning, { type: 'RESET' });
    expect(reset).toEqual({ phase: 'idle' });
    expect(assistantReducer(reset, { type: 'PLAN_READY', plan: samplePlan })).toBe(reset);

    // RESET from applying clears applying atomically.
    const applying = assistantReducer(
      assistantReducer(planning, { type: 'PLAN_READY', plan: samplePlan }),
      { type: 'START_APPLY' },
    );
    expect(assistantReducer(applying, { type: 'RESET' })).toEqual({ phase: 'idle' });
    expect(ALLOWED_ASSISTANT_TRANSITIONS.applying).toEqual(['planned', 'idle']);
  });
});
