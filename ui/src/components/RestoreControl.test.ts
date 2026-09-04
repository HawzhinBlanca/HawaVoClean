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

  it('does not quantise the cutoff to a step base that excludes round numbers', async () => {
    // `min=1` with `step=100` puts the step *base* at 1, so a browser's valid
    // set is 1, 101, 201 … and every round cutoff anyone types — 4000, 8000,
    // 12000 — is a stepMismatch, while this component accepted all of them and
    // left `aria-invalid` unset. Chrome on the running app: "Please enter a
    // valid value. The two nearest valid values are 7901 and 8001." The spinner
    // from empty stepped 1 → 101 → 201 and could never reach a round number.
    //
    // Asserted as an attribute, deliberately. happy-dom answers
    // `validity.stepMismatch === false` and `checkValidity() === true` for
    // exactly the value Chrome rejects, so a `checkValidity()` assertion here
    // would pass whatever `step` said and prove nothing. `step="any"` is the
    // whole fix: nothing about a cutoff is quantised.
    getState().setCapabilities(['character_01'], true);
    getState().setMode('restore');
    await render();
    const input = host.querySelector('.rc-fields input') as HTMLInputElement;
    expect(input.getAttribute('step')).toBe('any');
  });

  it('refuses a cutoff at or above Nyquist instead of restoring nothing', async () => {
    // The request schema is `gt=0` with no ceiling and the DSP clips to
    // Nyquist, so a cutoff above it ran a full pass that restored nothing and
    // reported nothing. The control now declines it and falls back to auto.
    getState().setCapabilities(['character_01'], true);
    getState().setMode('restore');
    getState().setOriginal({
      path: '/a.wav',
      duration_s: 60,
      sample_rate: 48000,
      channels: 1,
      peaks: { min: [], max: [] },
      rms_db: [],
      spectrum: { freqs_hz: [], db: [] },
      loudness: { integrated_lufs: null, true_peak_dbtp: null },
      noise_floor_db: null,
    });
    await render();
    const input = host.querySelector('.rc-fields input') as HTMLInputElement;
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
    // 24000 is Nyquist at 48 kHz: nothing sits above it.
    expect(input.getAttribute('max')).toBe('23999');
    await type('999999');
    expect(getState().cutoffHz).toBeNull();
    expect(input.getAttribute('aria-invalid')).toBe('true');
    await type('24000');
    expect(getState().cutoffHz).toBeNull();
    // One below Nyquist still leaves a band, so it is accepted.
    await type('23999');
    expect(getState().cutoffHz).toBe(23999);
    expect(input.getAttribute('aria-invalid')).toBeNull();
  });

  it('renders a blocked warning banner when capability is blocked (True-10 D4.11)', async () => {
    getState().setCapabilities(['character_01'], true);
    getState().setCapabilitiesV1([
      {
        capability_id: 'restore_enrolled',
        available: false,
        maturity: 'blocked',
        reason: 'No qualified signed Sorani Restore pack is installed',
      },
    ]);
    getState().setSpeakerId('character_01');
    getState().setMode('restore');
    await render();

    const warning = host.querySelector('.rc-blocked-warning');
    expect(warning).not.toBeNull();
    expect(warning?.textContent).toContain('BLOCKED');
    expect(warning?.textContent).toContain('No qualified signed Sorani Restore pack is installed');
  });

  it('toggles generative reconstruction consent in store (True-10 D4.11)', async () => {
    getState().setCapabilities(['character_01'], true);
    getState().setMode('restore');
    expect(getState().reconstructionConsent).toBe(false);
    await render();

    const checkbox = host.querySelector('.rc-consent input[type="checkbox"]') as HTMLInputElement;
    expect(checkbox).not.toBeNull();
    expect(checkbox.checked).toBe(false);

    await act(async () => {
      checkbox.click();
    });
    expect(getState().reconstructionConsent).toBe(true);
  });
});
