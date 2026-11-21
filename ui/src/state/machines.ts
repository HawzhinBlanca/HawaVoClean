import type {
  AudioPlan,
  CapabilityStatusV1,
  JobMode,
  ModelStatusV1,
  Profile,
  TrackInfo,
} from '../api/types';
import {
  getRouteBlockedReason,
  isProfileUnavailable,
  isRouteBlocked,
  jobInFlight,
  type AppState,
} from './store';

/**
 * Explicit finite state machines for the HawaVoClean UI.
 *
 * Each machine here owns a small, total, side-effect-free transition model:
 * - UX: prevents impossible UI combinations (e.g. submitting while planning,
 *   stale plans surviving source switches, or Process arming during an upload).
 * - DX: discriminated unions and single-source-of-truth derivations replace
 *   scattered boolean checks across components.
 * - AX: transition tables and pure reducers are inspectable in one file.
 */

// ---------------------------------------------------------------------------
// 1. Engine connection phase (derived from store connection fields)

export type EnginePhaseKind = 'connecting' | 'ready' | 'offline_waiting' | 'offline_probing';

export type EnginePhase =
  | { readonly phase: 'connecting' }
  | { readonly phase: 'ready'; readonly version: string | null }
  | {
      readonly phase: 'offline_probing';
      readonly offlineSince: number | null;
      readonly nextProbeAt: number | null;
    }
  | {
      readonly phase: 'offline_waiting';
      readonly offlineSince: number | null;
      readonly nextProbeAt: number | null;
    };

export const ENGINE_PHASE_TRANSITIONS: Readonly<
  Record<EnginePhaseKind, readonly EnginePhaseKind[]>
> = Object.freeze({
  connecting: Object.freeze(['ready', 'offline_waiting', 'offline_probing'] as const),
  ready: Object.freeze(['offline_waiting', 'offline_probing'] as const),
  offline_waiting: Object.freeze(['offline_probing', 'ready'] as const),
  offline_probing: Object.freeze(['ready', 'offline_waiting'] as const),
});

export function deriveEnginePhase(
  state: Pick<
    AppState,
    'engineStatus' | 'engineVersion' | 'engineOfflineSince' | 'engineNextProbeAt' | 'engineProbing'
  >,
): EnginePhase {
  if (state.engineStatus === 'ready') {
    return { phase: 'ready', version: state.engineVersion };
  }
  if (state.engineStatus === 'connecting' && state.engineOfflineSince === null) {
    return { phase: 'connecting' };
  }
  if (state.engineProbing) {
    return {
      phase: 'offline_probing',
      offlineSince: state.engineOfflineSince,
      nextProbeAt: state.engineNextProbeAt,
    };
  }
  return {
    phase: 'offline_waiting',
    offlineSince: state.engineOfflineSince,
    nextProbeAt: state.engineNextProbeAt,
  };
}

// ---------------------------------------------------------------------------
// 2. Workspace session phase & processing readiness (derived from store)

export type WorkspacePhase =
  | 'empty'
  | 'uploading'
  | 'analyzing'
  | 'ready'
  | 'processing'
  | 'reviewed'
  | 'failed';

export const WORKSPACE_PHASE_TRANSITIONS: Readonly<
  Record<WorkspacePhase, readonly WorkspacePhase[]>
> = Object.freeze({
  empty: Object.freeze(['uploading', 'analyzing', 'ready'] as const),
  uploading: Object.freeze(['analyzing', 'empty', 'failed'] as const),
  analyzing: Object.freeze(['ready', 'failed', 'empty'] as const),
  ready: Object.freeze(['processing', 'uploading', 'analyzing', 'empty'] as const),
  processing: Object.freeze(['reviewed', 'failed', 'ready'] as const),
  reviewed: Object.freeze(['processing', 'uploading', 'analyzing', 'empty'] as const),
  failed: Object.freeze(['processing', 'uploading', 'analyzing', 'ready', 'empty'] as const),
});

export function deriveWorkspacePhase(
  state: Pick<
    AppState,
    'source' | 'upload' | 'analyzing' | 'original' | 'job' | 'cleanedPath' | 'report'
  >,
): WorkspacePhase {
  if (state.upload !== null) return 'uploading';
  if (jobInFlight(state.job)) return 'processing';
  if (state.analyzing) return 'analyzing';
  if (!state.source) return 'empty';
  const jobState = state.job?.status?.state ?? null;
  if (jobState === 'failed' || jobState === 'interrupted') return 'failed';
  if (jobState === 'done' || (state.cleanedPath !== null && state.report !== null)) {
    return 'reviewed';
  }
  if (!state.original) return 'failed';
  return 'ready';
}

export interface ProcessingReadiness {
  readonly phase: WorkspacePhase;
  readonly running: boolean;
  readonly loading: boolean;
  readonly routeBlocked: boolean;
  readonly blockedReason: string | null;
  readonly restoreMissingConsent: boolean;
  readonly canStart: boolean;
}

export function deriveProcessingReadiness(
  state: Pick<
    AppState,
    | 'engineStatus'
    | 'source'
    | 'upload'
    | 'analyzing'
    | 'original'
    | 'job'
    | 'cleanedPath'
    | 'report'
    | 'mode'
    | 'profile'
    | 'speakerId'
    | 'capabilities'
    | 'reconstructionConsent'
  >,
): ProcessingReadiness {
  const phase = deriveWorkspacePhase(state);
  const running = phase === 'processing';
  const loading = phase === 'uploading' || phase === 'analyzing';
  const engineReady = state.engineStatus === 'ready';

  const { routeBlocked, blockedReason, restoreMissingConsent } = evaluateRouteGate(
    state.mode,
    state.profile,
    state.speakerId,
    state.capabilities,
    state.reconstructionConsent,
  );

  const canStart =
    engineReady &&
    state.source !== null &&
    state.original !== null &&
    !loading &&
    !running &&
    !routeBlocked &&
    !restoreMissingConsent;

  return {
    phase,
    running,
    loading,
    routeBlocked,
    blockedReason,
    restoreMissingConsent,
    canStart,
  };
}

function evaluateRouteGate(
  mode: JobMode,
  profile: Profile,
  speakerId: string | null,
  capabilities: CapabilityStatusV1[] | null,
  reconstructionConsent: boolean,
): {
  routeBlocked: boolean;
  blockedReason: string | null;
  restoreMissingConsent: boolean;
} {
  const restoreCapId = speakerId ? 'restore_enrolled' : 'restore_source';
  const isRestore = mode === 'restore';
  const restoreBlocked = isRestore && isRouteBlocked(capabilities, restoreCapId);
  const restoreBlockedReason = restoreBlocked
    ? getRouteBlockedReason(capabilities, restoreCapId)
    : null;
  const restoreMissingConsent = isRestore && !reconstructionConsent;

  const naturalBlocked = mode === 'natural' && isProfileUnavailable(capabilities, profile);
  const naturalBlockedReason = naturalBlocked
    ? (getRouteBlockedReason(capabilities, profile) ??
      (profile === 'studio_pro'
        ? 'Studio Pro is unavailable on this machine.'
        : profile === 'dereverb'
          ? 'Dereverb is unavailable on this machine.'
          : null))
    : null;

  const smartSafeBlocked = mode === 'smart_safe' && isRouteBlocked(capabilities, 'smart_safe');
  const smartSafeBlockedReason = smartSafeBlocked
    ? getRouteBlockedReason(capabilities, 'smart_safe')
    : null;

  return {
    routeBlocked: restoreBlocked || naturalBlocked || smartSafeBlocked,
    blockedReason: restoreBlockedReason || naturalBlockedReason || smartSafeBlockedReason,
    restoreMissingConsent,
  };
}

// ---------------------------------------------------------------------------
// 3. Audio Assistant chat -> plan -> apply finite state machine

export type AssistantPhase = 'idle' | 'planning' | 'planned' | 'applying' | 'error';

export type AssistantState =
  | { readonly phase: 'idle' }
  | { readonly phase: 'planning' }
  | { readonly phase: 'planned'; readonly plan: AudioPlan; readonly applyError: string | null }
  | { readonly phase: 'applying'; readonly plan: AudioPlan }
  | { readonly phase: 'error'; readonly message: string };

export type AssistantEvent =
  | { readonly type: 'START_PLAN' }
  | { readonly type: 'PLAN_READY'; readonly plan: AudioPlan }
  | { readonly type: 'PLAN_CLARIFIED' }
  | { readonly type: 'PLAN_FAILED'; readonly message: string }
  | { readonly type: 'PLAN_CANCELLED' }
  | { readonly type: 'START_APPLY' }
  | { readonly type: 'APPLY_STARTED' }
  | { readonly type: 'APPLY_FAILED'; readonly message: string }
  | { readonly type: 'DRAFT_CHANGED' }
  | { readonly type: 'RESET' };

export const ALLOWED_ASSISTANT_TRANSITIONS: Readonly<
  Record<AssistantPhase, readonly AssistantPhase[]>
> = Object.freeze({
  idle: Object.freeze(['planning', 'idle'] as const),
  planning: Object.freeze(['planned', 'idle', 'error'] as const),
  planned: Object.freeze(['applying', 'planning', 'idle'] as const),
  applying: Object.freeze(['planned', 'idle'] as const),
  error: Object.freeze(['planning', 'idle'] as const),
});

export const INITIAL_ASSISTANT_STATE: AssistantState = Object.freeze({ phase: 'idle' });

export function assistantReducer(state: AssistantState, event: AssistantEvent): AssistantState {
  switch (event.type) {
    case 'RESET':
      return INITIAL_ASSISTANT_STATE;
    case 'DRAFT_CHANGED':
      return state.phase === 'planned' || state.phase === 'error'
        ? INITIAL_ASSISTANT_STATE
        : state;
    case 'START_PLAN':
      if (state.phase === 'planning' || state.phase === 'applying') return state;
      return { phase: 'planning' };
    case 'PLAN_READY':
      return state.phase === 'planning'
        ? { phase: 'planned', plan: event.plan, applyError: null }
        : state;
    case 'PLAN_CLARIFIED':
    case 'PLAN_CANCELLED':
      return state.phase === 'planning' ? INITIAL_ASSISTANT_STATE : state;
    case 'PLAN_FAILED':
      return state.phase === 'planning' ? { phase: 'error', message: event.message } : state;
    case 'START_APPLY':
      return state.phase === 'planned' ? { phase: 'applying', plan: state.plan } : state;
    case 'APPLY_STARTED':
      return state.phase === 'applying'
        ? { phase: 'planned', plan: state.plan, applyError: null }
        : state;
    case 'APPLY_FAILED':
      return state.phase === 'applying'
        ? { phase: 'planned', plan: state.plan, applyError: event.message }
        : state;
    default: {
      const _exhaustive: never = event;
      void _exhaustive;
      return state;
    }
  }
}

// ---------------------------------------------------------------------------
// 4. Shell overlay / drawer coordinator (mutually exclusive drawers & modal)

export type ShellOverlay = 'none' | 'assistant' | 'batch' | 'models' | 'shortcuts';

export type ShellOverlayEvent =
  | { readonly type: 'OPEN'; readonly overlay: Exclude<ShellOverlay, 'none'> }
  | { readonly type: 'CLOSE' }
  | { readonly type: 'CLOSE_IF'; readonly overlay: Exclude<ShellOverlay, 'none'> }
  | { readonly type: 'TOGGLE'; readonly overlay: Exclude<ShellOverlay, 'none'> };

export const SHELL_OVERLAY_TRANSITIONS: Readonly<
  Record<ShellOverlay, readonly ShellOverlay[]>
> = Object.freeze({
  none: Object.freeze(['assistant', 'batch', 'models', 'shortcuts'] as const),
  assistant: Object.freeze(['none', 'batch', 'models', 'shortcuts'] as const),
  batch: Object.freeze(['none', 'assistant', 'models', 'shortcuts'] as const),
  models: Object.freeze(['none', 'assistant', 'batch', 'shortcuts'] as const),
  shortcuts: Object.freeze(['none', 'assistant', 'batch', 'models'] as const),
});

export function shellOverlayReducer(state: ShellOverlay, event: ShellOverlayEvent): ShellOverlay {
  switch (event.type) {
    case 'OPEN':
      return event.overlay;
    case 'CLOSE':
      return 'none';
    case 'CLOSE_IF':
      return state === event.overlay ? 'none' : state;
    case 'TOGGLE':
      return state === event.overlay ? 'none' : event.overlay;
    default: {
      const _exhaustive: never = event;
      void _exhaustive;
      return state;
    }
  }
}

// ---------------------------------------------------------------------------
// 5. DualPlayer audition mode (single source of truth for audio routing + HPF)

export type AuditionMode = 'original' | 'cleaned' | 'delta_removed' | 'wet_dry_blend';

export const AUDITION_MODE_TRANSITIONS: Readonly<
  Record<AuditionMode, readonly AuditionMode[]>
> = Object.freeze({
  original: Object.freeze(['cleaned', 'delta_removed', 'wet_dry_blend'] as const),
  cleaned: Object.freeze(['original', 'delta_removed', 'wet_dry_blend'] as const),
  delta_removed: Object.freeze(['original', 'cleaned', 'wet_dry_blend'] as const),
  wet_dry_blend: Object.freeze(['original', 'cleaned', 'delta_removed'] as const),
});

export interface AuditionModeInput {
  readonly active: 'original' | 'cleaned';
  readonly bothDecksReady: boolean;
  readonly deltaMode: boolean;
  readonly wetDryMix: number;
}

/**
 * Derive the single active audition mode from player controls.
 * Live HPF audition is ONLY active in `'original'` mode so Cleaned masters,
 * Wet/Dry blending, and phase-cancelled Δ Removed nulling stay 100% pure.
 */
export function deriveAuditionMode(input: AuditionModeInput): AuditionMode {
  if (input.deltaMode && input.bothDecksReady) return 'delta_removed';
  if (input.active === 'cleaned' && input.bothDecksReady && input.wetDryMix > 0 && input.wetDryMix < 1) {
    return 'wet_dry_blend';
  }
  return input.active === 'cleaned' ? 'cleaned' : 'original';
}

// ---------------------------------------------------------------------------
// 6. Live debounced auto-preview scheduler

export type LiveAuditionPhase = 'idle' | 'debouncing' | 'rendering';

export type LiveAuditionState =
  | { readonly phase: 'idle' }
  | { readonly phase: 'debouncing'; readonly sourcePath: string; readonly scheduledAt: number }
  | { readonly phase: 'rendering'; readonly sourcePath: string };

export type LiveAuditionEvent =
  | { readonly type: 'SCHEDULE'; readonly sourcePath: string; readonly now: number }
  | { readonly type: 'FIRE'; readonly sourcePath: string }
  | { readonly type: 'SETTLE' }
  | { readonly type: 'CANCEL' };

export const LIVE_AUDITION_TRANSITIONS: Readonly<
  Record<LiveAuditionPhase, readonly LiveAuditionPhase[]>
> = Object.freeze({
  idle: Object.freeze(['debouncing'] as const),
  debouncing: Object.freeze(['debouncing', 'rendering', 'idle'] as const),
  rendering: Object.freeze(['debouncing', 'idle'] as const),
});

export const INITIAL_LIVE_AUDITION_STATE: LiveAuditionState = Object.freeze({ phase: 'idle' });

export function liveAuditionReducer(
  state: LiveAuditionState,
  event: LiveAuditionEvent,
): LiveAuditionState {
  switch (event.type) {
    case 'SCHEDULE':
      return { phase: 'debouncing', sourcePath: event.sourcePath, scheduledAt: event.now };
    case 'FIRE':
      return state.phase === 'debouncing' && state.sourcePath === event.sourcePath
        ? { phase: 'rendering', sourcePath: event.sourcePath }
        : state;
    case 'SETTLE':
      return state.phase === 'rendering' ? INITIAL_LIVE_AUDITION_STATE : state;
    case 'CANCEL':
      return INITIAL_LIVE_AUDITION_STATE;
    default: {
      const _exhaustive: never = event;
      void _exhaustive;
      return state;
    }
  }
}

// ---------------------------------------------------------------------------
// 7. Poly-WAV track picker (replaces 5 independent useState flags)

export type TrackPickerPhase = 'closed' | 'loading' | 'ready' | 'extracting' | 'error';

export type TrackPickerState =
  | { readonly phase: 'closed'; readonly cachedTracks: readonly TrackInfo[] | null }
  | { readonly phase: 'loading'; readonly cachedTracks: readonly TrackInfo[] | null }
  | { readonly phase: 'ready'; readonly tracks: readonly TrackInfo[] }
  | {
      readonly phase: 'extracting';
      readonly tracks: readonly TrackInfo[];
      readonly trackIndex: number;
    }
  | { readonly phase: 'error'; readonly message: string };

export type TrackPickerEvent =
  | { readonly type: 'OPEN_CACHED' }
  | { readonly type: 'START_LOAD' }
  | { readonly type: 'LOAD_SUCCESS'; readonly tracks: readonly TrackInfo[] }
  | { readonly type: 'LOAD_FAILED'; readonly message: string }
  | { readonly type: 'START_EXTRACT'; readonly trackIndex: number }
  | { readonly type: 'EXTRACT_SUCCESS' }
  | { readonly type: 'EXTRACT_ABORTED' }
  | { readonly type: 'CLOSE' }
  | { readonly type: 'RESET_SOURCE' };

export const TRACK_PICKER_TRANSITIONS: Readonly<
  Record<TrackPickerPhase, readonly TrackPickerPhase[]>
> = Object.freeze({
  closed: Object.freeze(['loading', 'ready', 'error'] as const),
  loading: Object.freeze(['ready', 'error', 'closed'] as const),
  ready: Object.freeze(['extracting', 'closed'] as const),
  extracting: Object.freeze(['closed', 'ready'] as const),
  error: Object.freeze(['loading', 'closed'] as const),
});

export const INITIAL_TRACK_PICKER_STATE: TrackPickerState = Object.freeze({
  phase: 'closed',
  cachedTracks: null,
});

export function trackPickerReducer(
  state: TrackPickerState,
  event: TrackPickerEvent,
): TrackPickerState {
  switch (event.type) {
    case 'RESET_SOURCE':
      return INITIAL_TRACK_PICKER_STATE;
    case 'CLOSE': {
      const cached =
        state.phase === 'ready' || state.phase === 'extracting'
          ? state.tracks
          : state.phase === 'closed' || state.phase === 'loading'
            ? state.cachedTracks
            : null;
      return { phase: 'closed', cachedTracks: cached };
    }
    case 'OPEN_CACHED':
      if (state.phase === 'closed' && state.cachedTracks !== null) {
        return { phase: 'ready', tracks: state.cachedTracks };
      }
      return state;
    case 'START_LOAD':
      return {
        phase: 'loading',
        cachedTracks:
          state.phase === 'closed' || state.phase === 'loading' ? state.cachedTracks : null,
      };
    case 'LOAD_SUCCESS':
      return state.phase === 'loading' ? { phase: 'ready', tracks: event.tracks } : state;
    case 'LOAD_FAILED':
      return state.phase === 'loading' || state.phase === 'closed'
        ? { phase: 'error', message: event.message }
        : state;
    case 'START_EXTRACT':
      return state.phase === 'ready'
        ? { phase: 'extracting', tracks: state.tracks, trackIndex: event.trackIndex }
        : state;
    case 'EXTRACT_SUCCESS':
      return state.phase === 'extracting' ? INITIAL_TRACK_PICKER_STATE : state;
    case 'EXTRACT_ABORTED':
      return state.phase === 'extracting' ? { phase: 'ready', tracks: state.tracks } : state;
    default: {
      const _exhaustive: never = event;
      void _exhaustive;
      return state;
    }
  }
}

// ---------------------------------------------------------------------------
// 8. Neural Model Manager drawer (replaces 5 independent useState hooks)

export type ModelManagerPhase = 'idle' | 'loading' | 'ready' | 'installing' | 'error';

export type ModelFeedbackMap = Readonly<Record<string, { readonly kind: 'ok' | 'err'; readonly text: string }>>;

export type ModelManagerState =
  | { readonly phase: 'idle'; readonly models: readonly ModelStatusV1[]; readonly feedbackById: ModelFeedbackMap }
  | { readonly phase: 'loading'; readonly models: readonly ModelStatusV1[]; readonly feedbackById: ModelFeedbackMap }
  | { readonly phase: 'ready'; readonly models: readonly ModelStatusV1[]; readonly feedbackById: ModelFeedbackMap }
  | {
      readonly phase: 'installing';
      readonly models: readonly ModelStatusV1[];
      readonly modelId: string;
      readonly feedbackById: ModelFeedbackMap;
    }
  | {
      readonly phase: 'error';
      readonly models: readonly ModelStatusV1[];
      readonly message: string;
      readonly feedbackById: ModelFeedbackMap;
    };

export type ModelManagerEvent =
  | { readonly type: 'LOAD_START' }
  | { readonly type: 'LOAD_SUCCESS'; readonly models: readonly ModelStatusV1[] }
  | { readonly type: 'LOAD_FAILED'; readonly message: string }
  | { readonly type: 'INSTALL_START'; readonly modelId: string }
  | {
      readonly type: 'INSTALL_SUCCESS';
      readonly modelId: string;
      readonly message: string;
      readonly models?: readonly ModelStatusV1[];
    }
  | { readonly type: 'INSTALL_FAILED'; readonly modelId: string; readonly message: string };

export const MODEL_MANAGER_TRANSITIONS: Readonly<
  Record<ModelManagerPhase, readonly ModelManagerPhase[]>
> = Object.freeze({
  idle: Object.freeze(['loading'] as const),
  loading: Object.freeze(['ready', 'error'] as const),
  ready: Object.freeze(['loading', 'installing'] as const),
  installing: Object.freeze(['ready', 'loading'] as const),
  error: Object.freeze(['loading', 'installing'] as const),
});

export const INITIAL_MODEL_MANAGER_STATE: ModelManagerState = Object.freeze({
  phase: 'idle',
  models: Object.freeze([]),
  feedbackById: Object.freeze({}),
});

export function modelManagerReducer(
  state: ModelManagerState,
  event: ModelManagerEvent,
): ModelManagerState {
  switch (event.type) {
    case 'LOAD_START':
      if (state.phase === 'installing') return state;
      return { phase: 'loading', models: state.models, feedbackById: state.feedbackById };
    case 'LOAD_SUCCESS':
      if (state.phase === 'installing') {
        return { ...state, models: event.models };
      }
      return { phase: 'ready', models: event.models, feedbackById: state.feedbackById };
    case 'LOAD_FAILED':
      if (state.phase === 'installing') return state;
      return {
        phase: 'error',
        models: state.models,
        message: event.message,
        feedbackById: state.feedbackById,
      };
    case 'INSTALL_START': {
      if (state.phase === 'installing') return state;
      const nextFeedback = { ...state.feedbackById };
      delete nextFeedback[event.modelId];
      return {
        phase: 'installing',
        models: state.models,
        modelId: event.modelId,
        feedbackById: nextFeedback,
      };
    }
    case 'INSTALL_SUCCESS': {
      if (state.phase !== 'installing' || state.modelId !== event.modelId) return state;
      return {
        phase: 'ready',
        models: event.models && event.models.length > 0 ? event.models : state.models,
        feedbackById: {
          ...state.feedbackById,
          [event.modelId]: { kind: 'ok', text: event.message },
        },
      };
    }
    case 'INSTALL_FAILED': {
      if (state.phase !== 'installing' || state.modelId !== event.modelId) return state;
      return {
        phase: 'ready',
        models: state.models,
        feedbackById: {
          ...state.feedbackById,
          [event.modelId]: { kind: 'err', text: event.message },
        },
      };
    }
    default: {
      const _exhaustive: never = event;
      void _exhaustive;
      return state;
    }
  }
}

// ---------------------------------------------------------------------------
// 9. Microphone test recorder (eliminates double-click permission stream leaks)

export type MicTestPhase = 'idle' | 'requesting_permission' | 'recording' | 'encoding' | 'error';

export type MicTestState =
  | { readonly phase: 'idle' }
  | { readonly phase: 'requesting_permission' }
  | { readonly phase: 'recording'; readonly remainingS: number }
  | { readonly phase: 'encoding' }
  | { readonly phase: 'error'; readonly notice: string };

export type MicTestEvent =
  | { readonly type: 'REQUEST_PERMISSION' }
  | { readonly type: 'START_RECORDING'; readonly durationS: number }
  | { readonly type: 'TICK'; readonly remainingS: number }
  | { readonly type: 'START_ENCODING' }
  | { readonly type: 'COMPLETE' }
  | { readonly type: 'FAIL'; readonly notice: string }
  | { readonly type: 'RESET' };

export const MIC_TEST_TRANSITIONS: Readonly<
  Record<MicTestPhase, readonly MicTestPhase[]>
> = Object.freeze({
  idle: Object.freeze(['requesting_permission', 'error'] as const),
  requesting_permission: Object.freeze(['recording', 'error', 'idle'] as const),
  recording: Object.freeze(['recording', 'encoding', 'error', 'idle'] as const),
  encoding: Object.freeze(['idle', 'error'] as const),
  error: Object.freeze(['requesting_permission', 'idle'] as const),
});

export const INITIAL_MIC_TEST_STATE: MicTestState = Object.freeze({ phase: 'idle' });

export function micTestReducer(state: MicTestState, event: MicTestEvent): MicTestState {
  switch (event.type) {
    case 'RESET':
      return INITIAL_MIC_TEST_STATE;
    case 'REQUEST_PERMISSION':
      return state.phase === 'idle' || state.phase === 'error'
        ? { phase: 'requesting_permission' }
        : state;
    case 'START_RECORDING':
      return state.phase === 'requesting_permission'
        ? { phase: 'recording', remainingS: event.durationS }
        : state;
    case 'TICK':
      return state.phase === 'recording'
        ? { phase: 'recording', remainingS: Math.max(1, event.remainingS) }
        : state;
    case 'START_ENCODING':
      return state.phase === 'recording' ? { phase: 'encoding' } : state;
    case 'COMPLETE':
      return state.phase === 'encoding' ? INITIAL_MIC_TEST_STATE : state;
    case 'FAIL':
      return { phase: 'error', notice: event.notice };
    default: {
      const _exhaustive: never = event;
      void _exhaustive;
      return state;
    }
  }
}

// ---------------------------------------------------------------------------
// 10. Unified State Machine Registry (AX / DX single-file inspection catalog)

export const APP_STATE_MACHINES = Object.freeze({
  engine: ENGINE_PHASE_TRANSITIONS,
  workspace: WORKSPACE_PHASE_TRANSITIONS,
  assistant: ALLOWED_ASSISTANT_TRANSITIONS,
  overlay: SHELL_OVERLAY_TRANSITIONS,
  audition: AUDITION_MODE_TRANSITIONS,
  liveAudition: LIVE_AUDITION_TRANSITIONS,
  trackPicker: TRACK_PICKER_TRANSITIONS,
  modelManager: MODEL_MANAGER_TRANSITIONS,
  micTest: MIC_TEST_TRANSITIONS,
});
