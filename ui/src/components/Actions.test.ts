// components/Actions.tsx — the artefact row.
//
// The B6 branch here exists in the component and nowhere else: with the engine
// gone the three files are still on disk and their paths are still on screen,
// but they are not fetchable this second. Handing over a link that downloads a
// connection refusal is worse than saying so, and the sentence that says so is
// the only place that reasoning is expressed. It had no test.
//
// The second half is the pattern the file documents and half of it did not
// follow: a natively `disabled` control takes no pointer events and leaves the
// accessibility tree, so a `title` explaining *why* a file is unavailable can
// never be read — by anyone, sighted or not.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { EngineClient } from '../api/client';
import { getState, useStore } from '../state/store';
import { Actions } from './Actions';

vi.mock('../bridge', () => ({
  getBridge: () => ({ host: 'web', engine: { getEndpoint: async () => ({ baseUrl: '', token: '' }) } }),
}));
// Partial: `artifactsFor` is the pure function that builds the three URLs and
// is exactly what this component's behaviour is made of — mocking it away
// would leave the test asserting against its own stub. Only the three
// side-effecting Resolve/Finder calls are replaced.
vi.mock('../state/actions', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../state/actions')>()),
  importToResolve: vi.fn(),
  replaceInResolve: vi.fn(),
  revealOutput: vi.fn(),
}));

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
    root.render(createElement(Actions));
  });
}

/** A finished run whose three artefacts the engine will serve. */
async function finishedRun(): Promise<void> {
  await act(async () => {
    getState().setEngine('ready', new EngineClient({ baseUrl: 'http://127.0.0.1:8765', token: 't' }), '3.3.0');
    getState().setCleaned(null, '/out/a_studio.wav');
    getState().setJob({
      id: 'j1',
      outputPath: '/out/a_studio.wav',
      reportPath: '/out/a_studio.hawavoclean.json',
      status: null,
      streamConnected: false,
    });
  });
}

const links = (): HTMLAnchorElement[] => [...host.querySelectorAll<HTMLAnchorElement>('a.btn')];
const dead = (): HTMLButtonElement[] => [
  ...host.querySelectorAll<HTMLButtonElement>('button[aria-disabled="true"]'),
];

describe('the artefact row while the engine is there', () => {
  it('hands over all three files as real links', async () => {
    await finishedRun();
    await render();
    expect(links()).toHaveLength(3);
    for (const a of links()) {
      expect(a.getAttribute('href')).toContain('/api/audio');
      // A download, not a navigation: these open the file, not the app.
      expect(a.hasAttribute('download')).toBe(true);
    }
  });
});

describe('the artefact row while the engine is gone', () => {
  it('withdraws the links rather than handing over a connection refusal', async () => {
    await finishedRun();
    await act(async () => {
      getState().setEngine('offline', null, null);
    });
    await render();
    // The paths are still known and still on screen. They are simply not
    // fetchable this second.
    expect(links()).toHaveLength(0);
    expect(dead()).toHaveLength(3);
  });

  it('says the file is coming back, not that it is missing', async () => {
    await finishedRun();
    await act(async () => {
      getState().setEngine('offline', null, null);
    });
    await render();
    const titles = dead().map((b) => b.getAttribute('title') ?? '');
    for (const t of titles) {
      expect(t).toContain('engine is offline');
      expect(t).toContain('still on disk');
      // "Available once a run has finished" would be a lie: one has.
      expect(t).not.toContain('once a run has finished');
    }
  });

  it('keeps the explanation reachable — aria-disabled, never the native attribute', async () => {
    await finishedRun();
    await act(async () => {
      getState().setEngine('offline', null, null);
    });
    await render();
    // A natively disabled button takes no pointer events, so its `title` — the
    // sentence saying why the file is unavailable — is never shown to anyone,
    // and it leaves the accessibility tree entirely. That is the exact failure
    // the sentence exists to prevent.
    for (const b of dead()) {
      expect(b.disabled).toBe(false);
      expect(b.getAttribute('title')).toBeTruthy();
    }
  });
});

describe('the artefact row before there is a run', () => {
  it('says so, and does not pretend the engine is the problem', async () => {
    await act(async () => {
      getState().setEngine('ready', new EngineClient({ baseUrl: 'http://127.0.0.1:8765', token: 't' }), '3.3.0');
    });
    await render();
    expect(links()).toHaveLength(0);
    for (const b of dead()) {
      expect(b.getAttribute('title')).toContain('once a run has finished');
    }
  });
});

describe('a run whose master the engine can no longer serve', () => {
  it('withdraws the master and names the reason, keeping the other two', async () => {
    await finishedRun();
    await act(async () => {
      getState().setArtifacts({
        master: false,
        json: true,
        txt: true,
        reason: 'The master this run wrote is no longer on disk.',
      });
    });
    await render();
    // The report and its sidecar are unaffected by the master being gone.
    expect(links()).toHaveLength(2);
    const gone = dead();
    expect(gone).toHaveLength(1);
    expect(gone[0]?.getAttribute('title')).toContain('no longer on disk');
  });
});
