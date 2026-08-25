// state/reducedMotion.ts — the preference, live.
//
// The point of this file is the one property `motion/react`'s useReducedMotion
// did not have. Its implementation was `useState(prefersReducedMotion.current)`
// — a `useState` *initialiser* — so it sampled the preference at mount and
// never subscribed. This app mounts once and stays mounted for a whole editing
// session, so turning Reduce Motion on in System Settings changed nothing until
// a reload. A test that only asserts "returns true when the query matches"
// would have passed against the broken hook too; the assertion that matters is
// that a change event *after* mount reaches the render.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

/** A `matchMedia` we can drive, standing in for the OS preference. */
function installMatchMedia(initial: boolean): {
  set: (v: boolean) => void;
  listeners: () => number;
} {
  let matches = initial;
  const cbs = new Set<() => void>();
  Object.defineProperty(window, 'matchMedia', {
    configurable: true,
    writable: true,
    value: (query: string) => ({
      media: query,
      get matches() {
        return matches;
      },
      addEventListener: (_: string, cb: () => void) => cbs.add(cb),
      removeEventListener: (_: string, cb: () => void) => cbs.delete(cb),
      addListener: (cb: () => void) => cbs.add(cb),
      removeListener: (cb: () => void) => cbs.delete(cb),
      dispatchEvent: () => true,
      onchange: null,
    }),
  });
  return {
    set(v: boolean) {
      matches = v;
      for (const cb of [...cbs]) cb();
    },
    listeners: () => cbs.size,
  };
}

let host: HTMLElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(async () => {
    root.unmount();
  });
  host.remove();
});

/**
 * The module reads `window.matchMedia` once, at import, so each case needs a
 * fresh module registry with the stub already installed — hence
 * `vi.resetModules()` before every import.
 */
async function mount(initial: boolean): Promise<{
  set: (v: boolean) => void;
  listeners: () => number;
  read: () => string | null;
}> {
  const media = installMatchMedia(initial);
  vi.resetModules();
  const { useReducedMotion } = await import('./reducedMotion');
  function Probe(): ReactElement {
    return createElement('i', { 'data-reduced': String(useReducedMotion()) });
  }
  await act(async () => {
    root.render(createElement(Probe));
  });
  return {
    ...media,
    read: () => host.querySelector('i')?.getAttribute('data-reduced') ?? null,
  };
}

describe('useReducedMotion', () => {
  it('reads the preference on the first painted frame, not one effect later', async () => {
    const m = await mount(true);
    // useSyncExternalStore reads during render, so there is never a frame of
    // motion before an effect catches up.
    expect(m.read()).toBe('true');
  });

  it('follows the preference being turned on while the app stays mounted', async () => {
    const m = await mount(false);
    expect(m.read()).toBe('false');
    await act(async () => {
      m.set(true);
    });
    // This is the assertion motion/react failed: it latched 'false' at mount.
    expect(m.read()).toBe('true');
  });

  it('follows it being turned back off', async () => {
    const m = await mount(true);
    await act(async () => {
      m.set(false);
    });
    expect(m.read()).toBe('false');
  });

  it('unsubscribes on unmount', async () => {
    const m = await mount(false);
    expect(m.listeners()).toBe(1);
    await act(async () => {
      root.unmount();
    });
    expect(m.listeners()).toBe(0);
    // afterEach unmounts again; React tolerates it, and re-creating the root
    // here would leak a second one.
    root = createRoot(document.createElement('div'));
  });

  it('answers "not reduced" where matchMedia does not exist', async () => {
    // Resolve's webview and happy-dom are both reached by this bundle and
    // neither is guaranteed to implement it. Throwing during render would take
    // the whole app down for a preference query.
    Reflect.deleteProperty(window, 'matchMedia');
    vi.resetModules();
    const { useReducedMotion } = await import('./reducedMotion');
    function Probe(): ReactElement {
      return createElement('i', { 'data-reduced': String(useReducedMotion()) });
    }
    await act(async () => {
      root.render(createElement(Probe));
    });
    expect(host.querySelector('i')?.getAttribute('data-reduced')).toBe('false');
  });
});
