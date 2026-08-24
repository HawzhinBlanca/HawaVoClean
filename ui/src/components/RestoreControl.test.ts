// components/RestoreControl.tsx — the restore-mode control (contract
// addendum 2). These are DOM-truth tests, not pixel tests: the control's
// whole contract is *when it exists*, what it offers, and what it writes
// into the store — the store is what the submit reads. Rendering uses
// `react-dom/client` directly (no JSX, so this file rides the same
// `src/**/*.test.ts` include as every other unit test).

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { useStore, getState } from '../state/store';
import { RestoreControl } from './RestoreControl';

declare global {
  // React's act() refuses to run outside a declared test environment.
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
    root.render(createElement(RestoreControl));
  });
}

describe('RestoreControl', () => {
  it('renders nothing until the engine offers restore', async () => {
    await render();
    expect(host.querySelector('.restorectl')).toBeNull();
    // The capability arriving mid-session (a probe landed) brings it up.
    await act(async () => {
      getState().setCapabilities(['character_01'], true);
    });
    expect(host.querySelector('.restorectl')).not.toBeNull();
  });

  it('defaults to Natural with the restore fields folded away', async () => {
    getState().setCapabilities(['character_01'], true);
    await render();
    const on = host.querySelector<HTMLButtonElement>('.restorectl [role="radio"][aria-checked="true"]');
    expect(on?.dataset.value).toBe('natural');
    expect(host.querySelector('.rc-fields')).toBeNull();
  });

  it('restore mode offers exactly the engine’s speakers, first one selected', async () => {
    getState().setCapabilities(['character_01', 'character_02'], true);
    await render();
    await act(async () => {
      getState().setMode('restore');
    });
    const select = host.querySelector<HTMLSelectElement>('.rc-fields select');
    expect(select).not.toBeNull();
    expect(Array.from(select?.options ?? []).map((o) => o.value)).toEqual([
      'character_01',
      'character_02',
    ]);
    expect(select?.value).toBe('character_01');
  });

  it('picking a speaker writes it to the store the submit reads', async () => {
    getState().setCapabilities(['character_01', 'character_02'], true);
    getState().setMode('restore');
    await render();
    const select = host.querySelector('.rc-fields select') as HTMLSelectElement;
    await act(async () => {
      select.value = 'character_02';
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });
    expect(getState().speakerId).toBe('character_02');
  });

  it('the cutoff field parses to a positive number or to auto — never to junk', async () => {
    getState().setCapabilities(['character_01'], true);
    getState().setMode('restore');
    await render();
    const input = host.querySelector('.rc-fields input') as HTMLInputElement;
    // React instruments the element's own `value` property to dedupe events;
    // writing through the prototype's setter is what real typing does.
    const setValue = Object.getOwnPropertyDescriptor(
      Object.getPrototypeOf(input) as object,
      'value',
    )?.set;
    const type = async (text: string): Promise<void> => {
      await act(async () => {
        setValue?.call(input, text);
        input.dispatchEvent(new Event('input', { bubbles: true }));
      });
    };
    await type('7800');
    expect(getState().cutoffHz).toBe(7800);
    // Empty means auto-detect, which the submit expresses by omission.
    await type('');
    expect(getState().cutoffHz).toBeNull();
    // Zero is not a cutoff the engine accepts; it reads as auto, flagged.
    await type('0');
    expect(getState().cutoffHz).toBeNull();
    expect(input.getAttribute('aria-invalid')).toBe('true');
  });
});
