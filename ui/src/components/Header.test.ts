// components/Header.tsx — the wordmark's version chip.
//
// This file exists because of a bug no pixel test could see. The wordmark
// carried the literal string `v3.2` from commit 086b1c4 — whose own subject
// line is "v3.3.0-dev: first screen" — so it named the wrong product version
// from the moment the file was created, survived seven polish iterations and
// three adversarial audits, and shipped sitting two inches from a lamp reading
// `v3.3.0`. Contrast sweeps and screenshot diffs all pass on a string that is
// simply false.
//
// The chip is now derived from ui/package.json via `define` (vite.config.ts,
// mirrored in vitest.config.ts). `scripts/sync_release_identity.py` already
// gates ui/package.json against src/hawavoclean/release.json, so asserting the
// painted chip equals `__UI_VERSION__` completes the chain from the product's
// single source of truth all the way to the pixel:
//
//   release.json --(python gate)--> ui/package.json --(define)--> the wordmark
//
// Rendering uses react-dom/client directly (no JSX, so this file rides the
// same `src/**/*.test.ts` include as every other unit test).

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { setLocale } from '../state/i18n';
import { useStore, getState } from '../state/store';
import { Header } from './Header';

declare global {
  // React's act() refuses to run outside a declared test environment.
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const pristine = useStore.getState();

let host: HTMLElement;
let root: Root;

beforeEach(() => {
  setLocale('en');
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
  setLocale('en');
});

async function render(): Promise<void> {
  await act(async () => {
    root.render(createElement(Header));
  });
}

describe('Header wordmark version', () => {
  it('paints the UI’s own version, not a string typed into the component', async () => {
    await render();
    const chip = host.querySelector<HTMLElement>('.wordmark .ver');
    expect(chip).not.toBeNull();
    expect(chip?.textContent).toBe(`v${__UI_VERSION__}`);
  });

  it('is a real version, not an empty or placeholder chip', async () => {
    // A `define` that fails to resolve yields the identifier itself or an empty
    // string; either would render a wordmark that says "v" and pass a looser
    // assertion. Pin the shape instead.
    expect(__UI_VERSION__).toMatch(/^\d+\.\d+\.\d+$/);
  });

  it('does not confuse the UI version with the engine’s', async () => {
    // The two chips are different facts — what this bundle is, and what the
    // engine answering it is — and the header must be able to show them
    // disagreeing. That is the whole reason both exist.
    await act(async () => {
      getState().setEngine('ready', null, '9.9.9');
    });
    await render();
    expect(host.querySelector('.wordmark .ver')?.textContent).toBe(`v${__UI_VERSION__}`);
    expect(host.querySelector('.engine .ver')?.textContent).toBe('v9.9.9');
  });
});

describe('Header i18n and language toggle', () => {
  it('renders language toggle button and switches between EN and Kurdish', async () => {
    await render();
    const btn = host.querySelector<HTMLButtonElement>('.lang-toggle');
    expect(btn).not.toBeNull();
    expect(btn?.textContent).toBe('کوردی');
    expect(document.documentElement.dir).toBe('ltr');

    // Click language toggle
    await act(async () => {
      btn?.click();
    });

    expect(btn?.textContent).toBe('English');
    expect(document.documentElement.dir).toBe('rtl');
    expect(document.documentElement.lang).toBe('ckb');

    // HeaderNow should render Kurdish label
    const nowState = host.querySelector<HTMLElement>('.hn-state');
    expect(nowState?.textContent).toBe('هیچ کلیپێک نییە');
  });

  it('renders engine status in Kurdish when locale is ckb', async () => {
    await act(async () => {
      getState().setEngine('ready', null, '3.3.0');
    });
    await render();

    const btn = host.querySelector<HTMLButtonElement>('.lang-toggle');
    await act(async () => {
      btn?.click();
    });

    const engineTxt = host.querySelector<HTMLElement>('.engine .txt .in');
    expect(engineTxt?.textContent).toBe('بزوێنەر ئامادەیە');
  });
});
