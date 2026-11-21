import { describe, expect, it } from 'vitest';
import type { AudioAnalysis, AudioPlan, HawaVoCleanReport, JobStatus } from '../api/types';
import {
  ALLOWED_ASSISTANT_TRANSITIONS,
  APP_STATE_MACHINES,
  assistantReducer,
  deriveAuditionMode,
  deriveEnginePhase,
  deriveProcessingReadiness,
  deriveWorkspacePhase,
  ENGINE_PHASE_TRANSITIONS,
  INITIAL_ASSISTANT_STATE,
  INITIAL_LIVE_AUDITION_STATE,
  INITIAL_MIC_TEST_STATE,
  INITIAL_MODEL_MANAGER_STATE,
  INITIAL_TRACK_PICKER_STATE,
  liveAuditionReducer,
  micTestReducer,
  modelManagerReducer,
  shellOverlayReducer,
  trackPickerReducer,
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

describe('shellOverlayReducer & store mutual exclusion', () => {
  it('guarantees at most one drawer/modal is open and keeps legacy booleans in sync', () => {
    expect(shellOverlayReducer('none', { type: 'OPEN', overlay: 'assistant' })).toBe('assistant');
    expect(shellOverlayReducer('assistant', { type: 'OPEN', overlay: 'models' })).toBe('models');
    expect(shellOverlayReducer('models', { type: 'CLOSE_IF', overlay: 'assistant' })).toBe('models');
    expect(shellOverlayReducer('models', { type: 'CLOSE_IF', overlay: 'models' })).toBe('none');
    expect(shellOverlayReducer('none', { type: 'TOGGLE', overlay: 'batch' })).toBe('batch');
    expect(shellOverlayReducer('batch', { type: 'TOGGLE', overlay: 'batch' })).toBe('none');

    const st = useStore.getState();
    st.setAssistantDrawerOpen(true);
    expect(useStore.getState().activeOverlay).toBe('assistant');
    expect(useStore.getState().assistantDrawerOpen).toBe(true);

    // Opening Models automatically closes Assistant (zero stacked slide-over drawers!)
    st.setModelsDrawerOpen(true);
    expect(useStore.getState().activeOverlay).toBe('models');
    expect(useStore.getState().modelsDrawerOpen).toBe(true);
    expect(useStore.getState().assistantDrawerOpen).toBe(false);

    // Stale close of Assistant cannot close Models
    st.setAssistantDrawerOpen(false);
    expect(useStore.getState().activeOverlay).toBe('models');
    st.setActiveOverlay('none');
    expect(useStore.getState().modelsDrawerOpen).toBe(false);
  });
});

describe('deriveAuditionMode & liveAuditionReducer', () => {
  it('derives mutually exclusive audition modes matching audio routing invariants', () => {
    expect(
      deriveAuditionMode({ active: 'original', bothDecksReady: true, deltaMode: false, wetDryMix: 0.5 }),
    ).toBe('original');
    expect(
      deriveAuditionMode({ active: 'cleaned', bothDecksReady: true, deltaMode: false, wetDryMix: 1 }),
    ).toBe('cleaned');
    expect(
      deriveAuditionMode({ active: 'cleaned', bothDecksReady: true, deltaMode: false, wetDryMix: 0.5 }),
    ).toBe('wet_dry_blend');
    expect(
      deriveAuditionMode({ active: 'cleaned', bothDecksReady: true, deltaMode: true, wetDryMix: 0.5 }),
    ).toBe('delta_removed');
    expect(
      deriveAuditionMode({ active: 'cleaned', bothDecksReady: false, deltaMode: true, wetDryMix: 0.5 }),
    ).toBe('cleaned');
  });

  it('walks idle -> debouncing -> rendering -> idle and ignores stale source FIRE', () => {
    let s = liveAuditionReducer(INITIAL_LIVE_AUDITION_STATE, {
      type: 'SCHEDULE',
      sourcePath: '/a.wav',
      now: 100,
    });
    expect(s.phase).toBe('debouncing');
    expect(liveAuditionReducer(s, { type: 'FIRE', sourcePath: '/other.wav' })).toBe(s);
    s = liveAuditionReducer(s, { type: 'FIRE', sourcePath: '/a.wav' });
    expect(s).toEqual({ phase: 'rendering', sourcePath: '/a.wav' });
    s = liveAuditionReducer(s, { type: 'SETTLE' });
    expect(s).toEqual({ phase: 'idle' });
  });
});

describe('trackPickerReducer, modelManagerReducer, micTestReducer & registry', () => {
  it('enforces Poly-WAV track picker transitions and caches loaded tracks across close/reopen', () => {
    const tracks = [
      {
        index: 0,
        name: 'Boom',
        sample_rate: 48000,
        samples: 48000,
        duration_s: 1,
        rms_dbfs: -20,
        peak_dbfs: -6,
        speech_activity_ratio: 0.9,
        likely_dialogue: true,
      },
    ];
    let s = trackPickerReducer(INITIAL_TRACK_PICKER_STATE, { type: 'START_LOAD' });
    expect(s.phase).toBe('loading');
    s = trackPickerReducer(s, { type: 'LOAD_SUCCESS', tracks });
    expect(s).toEqual({ phase: 'ready', tracks });
    s = trackPickerReducer(s, { type: 'CLOSE' });
    expect(s).toEqual({ phase: 'closed', cachedTracks: tracks });
    s = trackPickerReducer(s, { type: 'OPEN_CACHED' });
    expect(s).toEqual({ phase: 'ready', tracks });
    s = trackPickerReducer(s, { type: 'START_EXTRACT', trackIndex: 0 });
    expect(s.phase).toBe('extracting');
    // Cannot extract a second track while extracting
    expect(trackPickerReducer(s, { type: 'START_EXTRACT', trackIndex: 1 })).toBe(s);
    s = trackPickerReducer(s, { type: 'EXTRACT_SUCCESS' });
    expect(s).toEqual(INITIAL_TRACK_PICKER_STATE);
  });

  it('prevents concurrent installs in modelManagerReducer and preserves loaded models on error', () => {
    const models = [{ model_id: 'nisqa_v2', display_name: 'NISQA', installed: false, verified: false, size_bytes: 0, approx_size_mb: 12, used_by: ['smart_safe'] }] as never;
    let s = modelManagerReducer(INITIAL_MODEL_MANAGER_STATE, { type: 'LOAD_START' });
    s = modelManagerReducer(s, { type: 'LOAD_SUCCESS', models });
    expect(s.phase).toBe('ready');
    s = modelManagerReducer(s, { type: 'INSTALL_START', modelId: 'nisqa_v2' });
    expect(s.phase).toBe('installing');
    // Concurrent install or refresh start while installing is blocked
    expect(modelManagerReducer(s, { type: 'INSTALL_START', modelId: 'other' })).toBe(s);
    expect(modelManagerReducer(s, { type: 'LOAD_START' })).toBe(s);
    s = modelManagerReducer(s, { type: 'INSTALL_SUCCESS', modelId: 'nisqa_v2', message: 'Verified' });
    expect(s.phase).toBe('ready');
    expect(s.feedbackById.nisqa_v2).toEqual({ kind: 'ok', text: 'Verified' });
  });

  it('prevents double permission requests and stream leaks in micTestReducer', () => {
    let s = micTestReducer(INITIAL_MIC_TEST_STATE, { type: 'REQUEST_PERMISSION' });
    expect(s).toEqual({ phase: 'requesting_permission' });
    // Second click while permission dialog is open is a no-op
    expect(micTestReducer(s, { type: 'REQUEST_PERMISSION' })).toBe(s);
    s = micTestReducer(s, { type: 'START_RECORDING', durationS: 5 });
    expect(s).toEqual({ phase: 'recording', remainingS: 5 });
    s = micTestReducer(s, { type: 'TICK', remainingS: 3 });
    expect(s).toEqual({ phase: 'recording', remainingS: 3 });
    s = micTestReducer(s, { type: 'START_ENCODING' });
    expect(s).toEqual({ phase: 'encoding' });
    // Clicking while encoding cannot restart recording
    expect(micTestReducer(s, { type: 'REQUEST_PERMISSION' })).toBe(s);
    s = micTestReducer(s, { type: 'COMPLETE' });
    expect(s).toEqual({ phase: 'idle' });
  });

  it('exports all 9 explicit frontend state machines in APP_STATE_MACHINES for AX inspection', () => {
    expect(Object.keys(APP_STATE_MACHINES)).toEqual([
      'engine',
      'workspace',
      'assistant',
      'overlay',
      'audition',
      'liveAudition',
      'trackPicker',
      'modelManager',
      'micTest',
    ]);
  });
});
