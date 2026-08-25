/**
 * `prefers-reduced-motion: reduce`, live.
 *
 * This replaces `useReducedMotion` from `motion/react`, which does not do what
 * its name says. Its implementation is:
 *
 *     const [shouldReduceMotion] = useState(prefersReducedMotion.current);
 *
 * — a `useState` *initialiser*. It samples the preference once when the
 * component mounts and never subscribes to the media query again; the library's
 * own source carries a TODO asking whether anyone misses the update. This app
 * mounts once and stays mounted for a whole editing session, so turning on
 * Reduce Motion in System Settings had no effect on the two things that read it
 * until the page was reloaded: the header's readout cross-fade kept
 * double-printing words in the engine slot, and the PROCESS ring kept easing.
 *
 * The query string is deliberately the exact one `interaction.css:1131` uses,
 * so the CSS and the JS can never disagree about what the user asked for.
 *
 * `useSyncExternalStore` rather than `useState` + `useEffect`: it reads the
 * preference during render, so the first painted frame already honours it —
 * there is no frame of motion before an effect catches up.
 */

import { useSyncExternalStore } from 'react';

const QUERY = '(prefers-reduced-motion: reduce)';

/**
 * Resolved once. Resolve's Electron webview and happy-dom both reach this file,
 * and neither is guaranteed to implement `matchMedia`; where it is missing the
 * honest answer is "the user has not asked for reduced motion", which is also
 * what the CSS does with a media query it cannot evaluate.
 */
const mql: MediaQueryList | null =
  typeof window !== 'undefined' && typeof window.matchMedia === 'function'
    ? window.matchMedia(QUERY)
    : null;

function subscribe(onChange: () => void): () => void {
  if (!mql) return () => {};
  // Safari below 14 has no addEventListener on MediaQueryList. The app targets
  // chrome136, but the same bundle is loaded by whatever webview Resolve ships,
  // so fall back rather than throw during render.
  if (typeof mql.addEventListener === 'function') {
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }
  const legacy = mql as MediaQueryList & {
    addListener?: (cb: () => void) => void;
    removeListener?: (cb: () => void) => void;
  };
  legacy.addListener?.(onChange);
  return () => legacy.removeListener?.(onChange);
}

function getSnapshot(): boolean {
  return mql?.matches === true;
}

/** True while the user has asked the system for reduced motion. Updates live. */
export function useReducedMotion(): boolean {
  // The third argument is the server snapshot; there is no SSR path here, but
  // supplying it keeps the hook usable from a test renderer that has one.
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}
