import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { setLocale } from '../state/i18n';
import { getState, useStore } from '../state/store';
import { ProfileControl } from './ProfileControl';

declare global {
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
    root.render(createElement(ProfileControl));
  });
}

function radio(value: string): HTMLButtonElement | null {
  return host.querySelector<HTMLButtonElement>(`button[role="radio"][data-value="${value}"]`);
}

describe('ProfileControl', () => {
  it('enables Studio Pro when capabilities report studio_pro available on MPS', async () => {
    useStore.setState({
      profile: 'production',
      capabilities: [
        { capability_id: 'studio_pro', available: true, maturity: 'qualified', providers: ['mps'] },
        { capability_id: 'studio', available: true, maturity: 'qualified', providers: ['cpu'] },
        { capability_id: 'lowband', available: true, maturity: 'qualified', providers: ['cpu'] },
        { capability_id: 'production', available: true, maturity: 'qualified', providers: ['cpu'] },
      ],
    });
    await render();

    const proBtn = radio('studio_pro');
    expect(proBtn).not.toBeNull();
    expect(proBtn?.textContent).toContain('Studio Pro');
    expect(proBtn?.getAttribute('aria-disabled')).toBeNull();

    await act(async () => {
      proBtn?.click();
    });

    expect(getState().profile).toBe('studio_pro');
    expect(proBtn?.getAttribute('aria-checked')).toBe('true');
    expect(host.querySelector('.pc-sub')?.textContent).toBe('48 kHz neural cleanup (MPS)');
  });

  it('disables Studio Pro cleanly with its capability reason when unavailable or blocked', async () => {
    useStore.setState({
      profile: 'studio',
      capabilities: [
        {
          capability_id: 'studio_pro',
          available: false,
          maturity: 'blocked',
          reason: 'Install MossFormer2_SE_48K weights to enable Studio Pro.',
        },
        { capability_id: 'studio', available: true, maturity: 'qualified', providers: ['cpu'] },
        { capability_id: 'lowband', available: true, maturity: 'qualified', providers: ['cpu'] },
        { capability_id: 'production', available: true, maturity: 'qualified', providers: ['cpu'] },
      ],
    });
    await render();

    const proBtn = radio('studio_pro');
    expect(proBtn?.getAttribute('aria-disabled')).toBe('true');
    expect(proBtn?.getAttribute('title')).toBe(
      'Install MossFormer2_SE_48K weights to enable Studio Pro.',
    );

    const descId = proBtn?.getAttribute('aria-describedby');
    expect(descId).toBeTruthy();
    expect(host.querySelector(`#${descId}`)?.textContent).toContain(
      'Install MossFormer2_SE_48K weights to enable Studio Pro.',
    );

    // Clicking a disabled option must be a no-op (never a broken click).
    await act(async () => {
      proBtn?.click();
    });
    expect(getState().profile).toBe('studio');
  });

  it('skips disabled Studio Pro during keyboard arrow navigation', async () => {
    useStore.setState({
      profile: 'production',
      capabilities: [
        {
          capability_id: 'studio_pro',
          available: false,
          maturity: 'blocked',
          reason: 'Unavailable on CPU-only host.',
        },
        { capability_id: 'studio', available: true, maturity: 'qualified', providers: ['cpu'] },
        { capability_id: 'lowband', available: true, maturity: 'qualified', providers: ['cpu'] },
        { capability_id: 'production', available: true, maturity: 'qualified', providers: ['cpu'] },
      ],
    });
    await render();

    const prodBtn = radio('production');
    await act(async () => {
      prodBtn?.dispatchEvent(
        new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }),
      );
    });
    // Wraps past disabled studio_pro straight to studio.
    expect(getState().profile).toBe('studio');
  });

  it('enables Dereverb when capabilities report dereverb available on MPS and disables it with reason when blocked', async () => {
    useStore.setState({
      profile: 'production',
      capabilities: [
        { capability_id: 'studio_pro', available: true, maturity: 'qualified', providers: ['mps'] },
        { capability_id: 'dereverb', available: true, maturity: 'qualified', providers: ['mps'] },
        { capability_id: 'studio', available: true, maturity: 'qualified', providers: ['cpu'] },
        { capability_id: 'lowband', available: true, maturity: 'qualified', providers: ['cpu'] },
        { capability_id: 'production', available: true, maturity: 'qualified', providers: ['cpu'] },
      ],
    });
    await render();

    const dereverbBtn = radio('dereverb');
    expect(dereverbBtn).not.toBeNull();
    expect(dereverbBtn?.textContent).toContain('Dereverb');
    expect(dereverbBtn?.getAttribute('aria-disabled')).toBeNull();

    await act(async () => {
      dereverbBtn?.click();
    });

    expect(getState().profile).toBe('dereverb');
    expect(dereverbBtn?.getAttribute('aria-checked')).toBe('true');
    expect(host.querySelector('.pc-sub')?.textContent).toBe(
      '48 kHz neural de-reverb + cleanup (MPS)',
    );

    await act(async () => {
      useStore.setState({
        profile: 'studio',
        capabilities: [
          {
            capability_id: 'dereverb',
            available: false,
            maturity: 'blocked',
            reason: 'Install GAP-URGENet weights to enable Dereverb.',
          },
          { capability_id: 'studio', available: true, maturity: 'qualified', providers: ['cpu'] },
        ],
      });
    });

    expect(dereverbBtn?.getAttribute('aria-disabled')).toBe('true');
    expect(dereverbBtn?.getAttribute('title')).toBe(
      'Install GAP-URGENet weights to enable Dereverb.',
    );
    await act(async () => {
      dereverbBtn?.click();
    });
    expect(getState().profile).toBe('studio');
  });

  it('renders localized Sorani Kurdish profile labels when locale is ckb', async () => {
    setLocale('ckb');
    useStore.setState({
      profile: 'studio',
      capabilities: [
        { capability_id: 'studio_pro', available: true, maturity: 'qualified', providers: ['cpu'] },
        { capability_id: 'dereverb', available: true, maturity: 'qualified', providers: ['cpu'] },
      ],
    });
    await render();

    expect(radio('studio_pro')?.textContent).toContain('ستۆدیۆ پرۆ');
    expect(radio('dereverb')?.textContent).toContain('لابردنی دەنگدانەوە');
    expect(radio('studio')?.textContent).toContain('ستۆدیۆ');
    expect(radio('lowband')?.textContent).toContain('باندی نزم');
    expect(radio('production')?.textContent).toContain('بەرهەمهێنان');
  });
});
