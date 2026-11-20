import { previewRegion } from './actions';
import { getState, jobInFlight } from './store';

const DEFAULT_DEBOUNCE_MS = 380;

let pendingTimer: ReturnType<typeof setTimeout> | null = null;

/** Cancel any queued live-knob auto-preview timer. */
export function cancelScheduledLivePreview(): void {
  if (pendingTimer !== null) {
    clearTimeout(pendingTimer);
    pendingTimer = null;
  }
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
  pendingTimer = setTimeout(() => {
    pendingTimer = null;
    const cur = getState();
    if (!cur.livePreviewEnabled) return;
    if (cur.engineStatus !== 'ready' || !cur.source || !cur.original || cur.analyzing) return;
    if (cur.source.path !== expectedPath || jobInFlight(cur.job)) return;
    void previewRegion();
  }, debounceMs);
}
