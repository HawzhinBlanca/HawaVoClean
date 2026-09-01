// components/SpectrogramDisplay.test.ts — Unit tests for the D3 scrollable energy map spectrogram.

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import type { AudioAnalysis } from '../api/types';
import { useStore } from '../state/store';
import { SpectrogramDisplay } from './SpectrogramDisplay';

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

function mockAnalysis(over: Partial<AudioAnalysis> = {}): AudioAnalysis {
  return {
    path: '/path/to/test.wav',
    sample_rate: 48000,
    channels: 1,
    duration_s: 2.0,
    loudness: { integrated_lufs: -18.0, true_peak_dbtp: -1.0 },
    noise_floor_db: -50.0,
    rms_db: [-20, -18, -19, -21],
    peaks: {
      min: [-0.5, -0.6, -0.4, -0.3],
      max: [0.5, 0.6, 0.4, 0.3],
    },
    spectrum: {
      freqs_hz: [100, 1000, 5000, 10000],
      db: [-30, -20, -40, -50],
    },
    ...over,
  };
}

let host: HTMLElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  useStore.getState().resetForNewSource();
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

describe('SpectrogramDisplay component', () => {
  it('renders nothing when no clip is loaded', () => {
    act(() => {
      root.render(createElement(SpectrogramDisplay));
    });
    expect(host.innerHTML).toBe('');
  });

  it('renders header with SHOW button when clip is loaded', () => {
    act(() => {
      useStore.getState().setSource({ path: '/test.wav', name: 'test.wav', origin: 'file' });
      useStore.getState().setOriginal(mockAnalysis());
      root.render(createElement(SpectrogramDisplay));
    });

    const panel = host.querySelector('.spectrogram-panel');
    expect(panel).not.toBeNull();
    const btn = host.querySelector('button.wave-fit');
    expect(btn?.textContent).toBe('SHOW');
    // Body is hidden initially
    expect(host.querySelector('.spectrogram-body')).toBeNull();
  });

  it('toggles body and canvas when SHOW/HIDE button is clicked', () => {
    act(() => {
      useStore.getState().setSource({ path: '/test.wav', name: 'test.wav', origin: 'file' });
      useStore.getState().setOriginal(mockAnalysis());
      root.render(createElement(SpectrogramDisplay));
    });

    const btn = host.querySelector('button.wave-fit') as HTMLButtonElement;
    expect(btn).not.toBeNull();

    // Click SHOW
    act(() => {
      btn.click();
    });

    expect(btn.textContent).toBe('HIDE');
    const body = host.querySelector('.spectrogram-body');
    expect(body).not.toBeNull();
    const canvas = host.querySelector('canvas.spectrogram-canvas');
    expect(canvas).not.toBeNull();
    expect(host.textContent).toContain('-60 dB');
    expect(host.textContent).toContain('0 dB');

    // Click HIDE
    act(() => {
      btn.click();
    });
    expect(btn.textContent).toBe('SHOW');
    expect(host.querySelector('.spectrogram-body')).toBeNull();
  });

  it('reflects cleaned deck when abMode is cleaned', () => {
    act(() => {
      useStore.getState().setSource({ path: '/test.wav', name: 'test.wav', origin: 'file' });
      useStore.getState().setOriginal(mockAnalysis());
      useStore.getState().setCleaned(mockAnalysis({ path: '/out.wav' }), '/out.wav');
      useStore.getState().setAbMode('cleaned');
      root.render(createElement(SpectrogramDisplay));
    });

    const sub = host.querySelector('.panel-title .sub');
    expect(sub?.textContent).toContain('Cleaned');
  });
});
