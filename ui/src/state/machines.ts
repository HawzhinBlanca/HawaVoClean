import type { AudioPlan, CapabilityStatusV1, JobMode, Profile } from '../api/types';
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
  if (jobState === 'failed') return 'failed';
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
