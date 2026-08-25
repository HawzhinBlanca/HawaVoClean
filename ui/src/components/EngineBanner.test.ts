// components/EngineBanner.tsx — the entire engine-offline surface.
//
// This is what the screen says when the process behind it dies, and it had no
// test at all. It carries a documented D1 fix that is easy to undo by
// accident: the section used to be `role="alert" aria-live="assertive"` with a
// retry countdown and an outage clock ticking inside it four times a second,
// so a screen-reader user could not hear anything else while the engine was
// down. The announcement is now one short sentence in a live element of its
// own whose text does not tick, and the section is an ordinary named region.
//
// Rendering uses react-dom/client directly (no JSX, so this file rides the
// same `src/**/*.test.{ts,tsx}` include as every other unit test).

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import type { JobStatus } from '../api/types';
import { getState, useStore } from '../state/store';
import { EngineBanner } from './EngineBanner';

vi.mock('../state/actions', () => ({ retryEngineNow: vi.fn() }));

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const pristine = useStore.getState();
let host: HTMLElement;
let root: Root;

beforeEach(() => {
  useStore.setState(pristine, true);
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

async function render(): Promise<void> {
  await act(async () => {
    root.render(createElement(EngineBanner));
  });
}

async function offline(): Promise<void> {
  // `setEngine` stamps `engineOfflineSince` itself on the first non-ready
  // status and keeps it until the engine is ready again — which is exactly the
  // signal the banner gates on.
  await act(async () => {
    getState().setEngine('offline', null, null);
    getState().setEngineProbe(Date.now() + 3000, false);
  });
}

const banner = (): HTMLElement | null => host.querySelector('section.offline-bar');
const text = (): string => banner()?.innerText?.replace(/\s+/g, ' ') ?? banner()?.textContent?.replace(/\s+/g, ' ') ?? '';

describe('EngineBanner appears only when the engine is really gone', () => {
  it('renders nothing while the engine is ready', async () => {
    await act(async () => {
      getState().setEngine('ready', null, '3.3.0');
    });
    await render();
    expect(banner()).toBeNull();
  });

  it('renders nothing on the first connect, before anything has ever worked', async () => {
    // `connecting` with no prior outage is startup, not a failure. A banner
    // here would greet every cold start with an alarm.
    await act(async () => {
      getState().setEngine('connecting', null, null);
    });
    await render();
    expect(banner()).toBeNull();
  });

  it('renders while reconnecting after an outage', async () => {
    await offline();
    await act(async () => {
      getState().setEngine('connecting', null, null);
    });
    await render();
    expect(banner()).not.toBeNull();
  });
});

describe('what the banner says', () => {
  it('is one short assertive sentence, and it does not carry the clock', async () => {
    await offline();
    await render();
    const alert = host.querySelector('[role="alert"]');
    expect(alert).not.toBeNull();
    // The D1 fix: the clock and the countdown live outside the live element.
    // If either leaks in, every tick becomes a fresh interruption.
    expect(alert?.textContent ?? '').not.toMatch(/\d+\.\d\s*s/);
    expect(alert?.textContent ?? '').toContain('Nothing on screen was lost');
    // And the section itself is a plain named region, not a second alert.
    expect(banner()?.getAttribute('role')).not.toBe('alert');
    expect(banner()?.getAttribute('aria-label')).toBe('Engine status');
  });

  it('names what is being held, item by item', async () => {
    // "Your work is safe" is worth nothing; a list is worth something.
    await offline();
    await act(async () => {
      getState().setSource({ path: '/a.wav', name: 'take-01.wav', origin: 'file' });
    });
    await render();
    expect(text()).toContain('take-01.wav');
  });

  it('says something different when a run was in flight', async () => {
    await offline();
    await act(async () => {
      getState().setJob({
        id: 'j1',
        outputPath: '/out/a.wav',
        reportPath: '/out/a.json',
        status: null as JobStatus | null,
        streamConnected: false,
      });
    });
    await render();
    // A run in flight cannot be promised safe — its outcome is unknown until
    // the engine answers — so the sentence must not claim nothing was lost.
    const alert = host.querySelector('[role="alert"]')?.textContent ?? '';
    expect(alert).toContain('in flight');
    expect(text()).toContain('read back');
  });

  it('offers a way out that does not require waiting', async () => {
    await offline();
    await render();
    const retry = [...host.querySelectorAll('button')].find((b) => /retry now/i.test(b.textContent ?? ''));
    expect(retry).toBeDefined();
    expect(retry?.disabled).toBe(false);
  });
});
