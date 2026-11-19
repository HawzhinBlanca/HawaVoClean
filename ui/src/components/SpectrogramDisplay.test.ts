// components/SpectrogramDisplay.test.ts — Unit & accessibility tests for the interactive 2D STFT spectrogram.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import axe from 'axe-core';
import type { AudioAnalysis } from '../api/types';
import { SPECTRAL_SELECTION_EVENT, type SpectralSelection } from '../render/stftSpectrogram';
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

  it('switches between Original, Cleaned, and Removed (Δ) view modes and Log / Mel / Linear frequency scales', () => {
    act(() => {
      useStore.getState().setSource({ path: '/test.wav', name: 'test.wav', origin: 'file' });
      useStore.getState().setOriginal(mockAnalysis());
      useStore.getState().setCleaned(mockAnalysis({ path: '/out.wav' }), '/out.wav');
      root.render(createElement(SpectrogramDisplay));
    });

    // Expand spectrogram
    const showBtn = host.querySelector<HTMLButtonElement>('button.wave-fit');
    act(() => showBtn?.click());

    // Frequency axis ticks 100 Hz .. 16 kHz rendered
    const tickLabels = [...host.querySelectorAll('.spectrogram-axis-tick')].map(
      (el) => el.textContent,
    );
    expect(tickLabels).toEqual(['100 Hz', '500 Hz', '1 kHz', '4 kHz', '8 kHz', '16 kHz']);

    // Switch to Removed (Δ)
    const buttons = [...host.querySelectorAll<HTMLButtonElement>('.spectrogram-seg-btn')];
    const deltaBtn = buttons.find((b) => b.textContent === 'Removed (Δ)');
    expect(deltaBtn).not.toBeUndefined();
    act(() => deltaBtn?.click());
    expect(deltaBtn?.getAttribute('aria-pressed')).toBe('true');
    expect(host.querySelector('.panel-title .sub')?.textContent).toContain('Removed (Δ)');
    expect(host.querySelector('.spectrogram-gradient.delta')).not.toBeNull();

    // Switch scale from Log to Linear and check tick positions change
    const logTop1k = host.querySelectorAll<HTMLElement>('.spectrogram-axis-tick')[2]?.style.top;
    const linearBtn = buttons.find((b) => b.textContent === 'Linear');
    expect(linearBtn).not.toBeUndefined();
    act(() => linearBtn?.click());
    expect(linearBtn?.getAttribute('aria-pressed')).toBe('true');
    const linTop1k = host.querySelectorAll<HTMLElement>('.spectrogram-axis-tick')[2]?.style.top;
    expect(linTop1k).not.toBe(logTop1k);
  });

  it('supports interactive 2D Time x Frequency box selection, dispatches hawa:spectral-selection, and clears via button', () => {
    const received: Array<SpectralSelection | null> = [];
    const onEvent = (e: Event): void => {
      received.push((e as CustomEvent<SpectralSelection | null>).detail);
    };
    window.addEventListener(SPECTRAL_SELECTION_EVENT, onEvent);
    const onProp = vi.fn();

    try {
      act(() => {
        useStore.getState().setSource({ path: '/test.wav', name: 'test.wav', origin: 'file' });
        useStore.getState().setOriginal(mockAnalysis({ duration_s: 10.0, sample_rate: 48000 }));
        root.render(createElement(SpectrogramDisplay, { onSpectralSelectionChange: onProp }));
      });

      act(() => host.querySelector<HTMLButtonElement>('button.wave-fit')?.click());

      const stage = host.querySelector<HTMLDivElement>('.spectrogram-stage');
      expect(stage).not.toBeNull();
      vi.spyOn(stage!, 'getBoundingClientRect').mockReturnValue({
        left: 0,
        top: 0,
        width: 400,
        height: 160,
        right: 400,
        bottom: 160,
        x: 0,
        y: 0,
        toJSON: () => ({}),
      });

      // Drag from (x=100 -> 2.5s, y=32 -> high Hz) to (x=300 -> 7.5s, y=128 -> low Hz)
      act(() => {
        stage!.dispatchEvent(
          new PointerEvent('pointerdown', { bubbles: true, button: 0, clientX: 100, clientY: 32, pointerId: 1 }),
        );
      });
      act(() => {
        stage!.dispatchEvent(
          new PointerEvent('pointermove', { bubbles: true, button: 0, clientX: 300, clientY: 128, pointerId: 1 }),
        );
      });
      act(() => {
        stage!.dispatchEvent(
          new PointerEvent('pointerup', { bubbles: true, button: 0, clientX: 300, clientY: 128, pointerId: 1 }),
        );
      });

      expect(received).toHaveLength(1);
      const sel = received[0]!;
      expect(sel.startS).toBeCloseTo(2.5, 2);
      expect(sel.endS).toBeCloseTo(7.5, 2);
      expect(sel.fLowHz).toBeGreaterThanOrEqual(20);
      expect(sel.fHighHz).toBeGreaterThan(sel.fLowHz);
      expect(sel.fHighHz).toBeLessThanOrEqual(24000);
      expect(onProp).toHaveBeenLastCalledWith(sel);
      expect(useStore.getState().spectralSelection).toEqual(sel);

      // Selection box + Hz/timecode chip + Clear button rendered
      expect(host.querySelector('.spectrogram-selection-box')).not.toBeNull();
      const chip = host.querySelector('.spectrogram-selection-chip');
      expect(chip).not.toBeNull();
      expect(chip?.textContent).toContain('00:02.50');
      expect(chip?.textContent).toContain('00:07.50');

      const clearBtn = host.querySelector<HTMLButtonElement>('button.spectrogram-clear-btn');
      expect(clearBtn).not.toBeNull();
      act(() => clearBtn?.click());

      expect(received).toHaveLength(2);
      expect(received[1]).toBeNull();
      expect(host.querySelector('.spectrogram-selection-chip')).toBeNull();
      expect(host.querySelector('.spectrogram-selection-box')).toBeNull();
    } finally {
      window.removeEventListener(SPECTRAL_SELECTION_EVENT, onEvent);
    }
  });

  it('passes axe-core WCAG 2.2 AA accessibility checks when expanded with active 2D selection', async () => {
    act(() => {
      useStore.getState().setSource({ path: '/test.wav', name: 'test.wav', origin: 'file' });
      useStore.getState().setOriginal(mockAnalysis({ duration_s: 4.0 }));
      useStore.getState().setCleaned(mockAnalysis({ path: '/out.wav', duration_s: 4.0 }), '/out.wav');
      root.render(createElement(SpectrogramDisplay));
    });

    act(() => host.querySelector<HTMLButtonElement>('button.wave-fit')?.click());

    const results = await axe.run(host, {
      rules: {
        'color-contrast': { enabled: false },
        region: { enabled: false },
      },
    });
    expect(results.violations).toEqual([]);
  });

  it('fetches live 2D STFT tiles from POST /api/v1/spectrogram when ready and caches by key', async () => {
    vi.useFakeTimers();
    try {
      const fetchSpectrogramTile = vi.fn(async () => ({
        start_s: 0,
        end_s: 2,
        sample_rate: 48000,
        f_min_hz: 20,
        f_max_hz: 24000,
        n_bins: 2,
        width_cols: 2,
        scale: 'linear' as const,
        db_floor: -72,
        db_ceil: -2,
        data_b64: 'AQIDBA==',
      }));
      act(() => {
        useStore.setState({
          engineStatus: 'ready',
          client: { fetchSpectrogramTile } as unknown as NonNullable<
            ReturnType<typeof useStore.getState>['client']
          >,
        });
        useStore.getState().setSource({ path: '/path/to/test.wav', name: 'test.wav', origin: 'file' });
        useStore.getState().setOriginal(mockAnalysis({ duration_s: 2.0 }));
        root.render(createElement(SpectrogramDisplay));
      });

      act(() => host.querySelector<HTMLButtonElement>('button.wave-fit')?.click());
      await act(async () => {
        await vi.advanceTimersByTimeAsync(160);
      });

      expect(fetchSpectrogramTile).toHaveBeenCalledTimes(1);
      expect(fetchSpectrogramTile).toHaveBeenCalledWith(
        expect.objectContaining({
          path: '/path/to/test.wav',
          mode: 'original',
          startS: 0,
          endS: 2,
          scale: 'linear',
        }),
        expect.any(AbortSignal),
      );
      expect(host.textContent).toContain('-72 dB');
      expect(host.textContent).toContain('-2 dB');
    } finally {
      vi.useRealTimers();
    }
  });
});
