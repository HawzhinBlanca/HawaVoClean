import { previewRegion } from './actions';
import {
  INITIAL_LIVE_AUDITION_STATE,
  liveAuditionReducer,
  type LiveAuditionState,
} from './machines';
import { getState, jobInFlight } from './store';

const DEFAULT_DEBOUNCE_MS = 380;

let pendingTimer: ReturnType<typeof setTimeout> | null = null;
let auditionState: LiveAuditionState = INITIAL_LIVE_AUDITION_STATE;

/** Inspect the explicit debounced live-preview state machine (`idle | debouncing | rendering`). */
export function getLiveAuditionState(): LiveAuditionState {
  return auditionState;
}

/** Cancel any queued live-knob auto-preview timer. */
export function cancelScheduledLivePreview(): void {
  if (pendingTimer !== null) {
    clearTimeout(pendingTimer);
    pendingTimer = null;
  }
  auditionState = liveAuditionReducer(auditionState, { type: 'CANCEL' });
}

/**
 * Schedule a debounced warm-worker region preview after the user adjusts a
 * knob or goal chip so changes can be auditioned almost immediately without
 * waiting for a full-file render.
 */
export function scheduleLivePreview(debounceMs = DEFAULT_DEBOUNCE_MS): void {
  cancelScheduledLivePreview();
  const st = getState();
  if (!st.livePreviewEnabled) return;
  if (st.engineStatus !== 'ready' || !st.source || !st.original || st.analyzing) return;
  if (jobInFlight(st.job)) return;
  if (!(st.original.duration_s >= 0.5)) return;

  const expectedPath = st.source.path;
  auditionState = liveAuditionReducer(auditionState, {
    type: 'SCHEDULE',
    sourcePath: expectedPath,
    now: Date.now(),
  });
  pendingTimer = setTimeout(() => {
    pendingTimer = null;
    const cur = getState();
    if (
      !cur.livePreviewEnabled ||
      cur.engineStatus !== 'ready' ||
      !cur.source ||
      !cur.original ||
      cur.analyzing ||
      cur.source.path !== expectedPath ||
      jobInFlight(cur.job)
    ) {
      auditionState = liveAuditionReducer(auditionState, { type: 'CANCEL' });
      return;
    }
    auditionState = liveAuditionReducer(auditionState, {
      type: 'FIRE',
      sourcePath: expectedPath,
    });
    void Promise.resolve(previewRegion()).finally(() => {
      auditionState = liveAuditionReducer(auditionState, { type: 'SETTLE' });
    });
  }, debounceMs);
}
