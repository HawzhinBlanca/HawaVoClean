// The whole keyboard map (goal box B1) in one place, lifted out of App so the
// bindings can be exercised directly by unit tests as well as by the browser.
// App mounts `useKeyboardMap()`; nothing else installs a global key listener.

import { useEffect } from 'react';
import { isTerminal } from '../api/sse';
import { getPlayer } from '../audio/player';
import {
  cancelAnalysis,
  cancelJob,
  cancelUpload,
  isAnalyzing,
  seekTo,
  setAb,
  startJob,
  togglePlay,
} from './actions';
import { clearSelection, stepUnit } from './selection';
import { getState } from './store';

const SEEK_S = 5;
const SEEK_FINE_S = 1;

/** Typing must never trigger transport keys. */
export function isEditable(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || typeof el.tagName !== 'string') return false;
  const tag = el.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  return el.isContentEditable === true;
}

/** Space on a focused control belongs to that control, not to the transport. */
export function isActivatable(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || typeof el.tagName !== 'string') return false;
  return el.tagName === 'BUTTON' || el.tagName === 'A' || el.getAttribute('role') === 'button';
}

/**
 * The whole keyboard map (goal box B1) in one place, so no two components can
 * claim the same key. Nothing here fires with Cmd/Ctrl/Alt held — those belong
 * to the browser (⌘A still selects) — and nothing fires while a text field has
 * focus or while the shortcut dialog is up.
 */
export function handleKeyDown(e: KeyboardEvent): void {
  if (e.defaultPrevented) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (isEditable(e.target)) return;
  const st = getState();

  if (e.key === '?') {
    e.preventDefault();
    st.setShortcutsOpen(!st.shortcutsOpen);
    return;
  }
  // While the dialog is open it owns the keyboard (it handles Esc/Tab itself).
  if (st.shortcutsOpen) return;

  const hasAudio = Boolean(st.original);
  const job = st.job;
  const running = Boolean(job) && (!job?.status || !isTerminal(job.status.state));

  switch (e.key) {
    case 'Escape': {
      e.preventDefault();
      // Esc means "stop the thing that is happening", innermost first:
      // an upload in flight, then an analysis, then a running job, then the selection.
      if (st.upload) cancelUpload();
      else if (isAnalyzing()) cancelAnalysis();
      else if (running) void cancelJob();
      else if (st.rejection) st.setRejection(null);
      else if (st.selectedUnit) clearSelection();
      return;
    }
    case 'ArrowLeft':
    case 'ArrowRight': {
      if (!hasAudio) return;
      e.preventDefault();
      const step = e.shiftKey ? SEEK_FINE_S : SEEK_S;
      const dir = e.key === 'ArrowRight' ? 1 : -1;
      seekTo(Math.max(0, getPlayer().time + dir * step));
      return;
    }
    case '[':
    case ',': {
      e.preventDefault();
      stepUnit(-1);
      return;
    }
    case ']':
    case '.': {
      e.preventDefault();
      stepUnit(1);
      return;
    }
    default:
      break;
  }

  if (e.code === 'Space' || e.key === ' ') {
    if (isActivatable(e.target)) return;
    e.preventDefault();
    if (hasAudio) togglePlay();
    return;
  }

  switch (e.key.toLowerCase()) {
    case 'a':
      if (hasAudio) setAb('original');
      break;
    case 'b':
      if (st.cleanedPath) setAb('cleaned');
      break;
    case 'p': {
      const canStart =
        st.engineStatus === 'ready' && !!st.source && !!st.original && !st.analyzing && !running;
      if (canStart) void startJob();
      break;
    }
    default:
      break;
  }
}

/** Install the map for the lifetime of the app. */
export function useKeyboardMap(): void {
  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);
}
