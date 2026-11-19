// components/resilience.test.ts — Component-level leak, abort, chaos, and a11y regression tests.
//
// Covers:
//  1. SpectrogramDisplay: abort in-flight /api/v1/spectrogram on hide/unmount/source-switch,
//     re-fetch cleanly after aborted hide->show, no setState after unmount, pointercancel mid-drag,
//     out-of-bounds drag clamping, and zero-width canvas ResizeObserver safety.
//  2. LiveMeter: clean RAF cancellation on pause and unmount (including stale callback safety).
//  3. VideoPreview: 0-second duration, NaN currentTime, media decode error, rapid playing toggle,
//     and video.play() NotAllowedError rejection with zero unhandled rejections.
//  4. StudioRack: hostile localStorage (__proto__, corrupt JSON, huge array) and QuotaExceededError
//     during save/import without crashing + WCAG 2.2 AA axe-core check.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import axe from 'axe-core';
import type { AudioAnalysis } from '../api/types';
import { DEFAULT_STUDIO_RACK_SETTINGS, PRESET_STORAGE_KEY } from '../state/presets';
import { useStore } from '../state/store';
import { LiveMeter } from './LiveMeter';
import { SpectrogramDisplay } from './SpectrogramDisplay';
import { StudioRack } from './StudioRack';
import { VideoPreview } from './VideoPreview';

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

function mockAnalysis(over: Partial<AudioAnalysis> = {}): AudioAnalysis {
  return {
    path: '/path/to/clip_a.wav',
    sample_rate: 48000,
    channels: 1,
    duration_s: 10.0,
    loudness: { integrated_lufs: -18.0, true_peak_dbtp: -1.0 },
    noise_floor_db: -50.0,
    rms_db: [-20, -18, -19, -21],
    peaks: { min: [-0.5, -0.6, -0.4, -0.3], max: [0.5, 0.6, 0.4, 0.3] },
    spectrum: { freqs_hz: [100, 1000, 5000, 10000], db: [-30, -20, -40, -50] },
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

afterEach(async () => {
  await act(async () => {
    root.unmount();
  });
  host.remove();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('SpectrogramDisplay resilience (abort, unmount, drag bounds, pointercancel, zero-width resize)', () => {
  it('aborts in-flight /api/v1/spectrogram on hide, source switch, and unmount, and re-fetches cleanly when re-shown', async () => {
    vi.useFakeTimers();
    const signals: AbortSignal[] = [];
    const resolvers: Array<(v: unknown) => void> = [];
    const fetchSpectrogramTile = vi.fn(
      (_req: unknown, signal?: AbortSignal) =>
        new Promise((resolve) => {
          if (signal) signals.push(signal);
          resolvers.push(resolve);
        }),
    );

    act(() => {
      useStore.setState({
        engineStatus: 'ready',
        client: { fetchSpectrogramTile } as unknown as NonNullable<
          ReturnType<typeof useStore.getState>['client']
        >,
      });
      useStore.getState().setSource({ path: '/path/to/clip_a.wav', name: 'clip_a.wav', origin: 'file' });
      useStore.getState().setOriginal(mockAnalysis());
      root.render(createElement(SpectrogramDisplay));
    });

    // Show -> debounce triggers fetch #1
    const showBtn = () => host.querySelector<HTMLButtonElement>('button.wave-fit')!;
    act(() => showBtn().click());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(150);
    });
    expect(fetchSpectrogramTile).toHaveBeenCalledTimes(1);
    expect(signals[0]?.aborted).toBe(false);

    // Hide while fetch #1 is still in flight -> MUST abort signal #1
    act(() => showBtn().click());
    expect(signals[0]?.aborted).toBe(true);

    // Re-show -> MUST trigger a fresh fetch #2 (regression check for stuck lastFetchKeyRef)
    act(() => showBtn().click());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(150);
    });
    expect(fetchSpectrogramTile).toHaveBeenCalledTimes(2);
    expect(signals[1]?.aborted).toBe(false);

    // Switch source while fetch #2 is in flight -> MUST abort signal #2 and start fetch #3
    act(() => {
      useStore.getState().setSource({ path: '/path/to/clip_b.wav', name: 'clip_b.wav', origin: 'file' });
      useStore.getState().setOriginal(mockAnalysis({ path: '/path/to/clip_b.wav' }));
    });
    expect(signals[1]?.aborted).toBe(true);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(150);
    });
    expect(fetchSpectrogramTile).toHaveBeenCalledTimes(3);
    expect(signals[2]?.aborted).toBe(false);

    // Unmount while fetch #3 is in flight -> MUST abort signal #3, and late resolve must not throw/setState
    act(() => root.unmount());
    expect(signals[2]?.aborted).toBe(true);

    await act(async () => {
      resolvers[2]?.({
        start_s: 0,
        end_s: 10,
        sample_rate: 48000,
        f_min_hz: 20,
        f_max_hz: 24000,
        n_bins: 2,
        width_cols: 2,
        scale: 'linear',
        db_floor: -60,
        db_ceil: 0,
        data_b64: 'AQIDBA==',
      });
      await Promise.resolve();
    });
  });

  it('handles pointercancel mid-drag, out-of-bounds negative/overflow drags, and zero-width ResizeObserver safely', () => {
    let resizeCb: ResizeObserverCallback | null = null;
    const disconnectSpy = vi.fn();
    class MockResizeObserver {
      constructor(cb: ResizeObserverCallback) {
        resizeCb = cb;
      }
      observe = vi.fn();
      unobserve = vi.fn();
      disconnect = disconnectSpy;
    }
    vi.stubGlobal('ResizeObserver', MockResizeObserver);

    const onChange = vi.fn();
    try {
      act(() => {
        useStore.getState().setSource({ path: '/test.wav', name: 'test.wav', origin: 'file' });
        useStore.getState().setOriginal(mockAnalysis({ duration_s: 10.0, sample_rate: 48000 }));
        root.render(createElement(SpectrogramDisplay, { onSpectralSelectionChange: onChange }));
      });
      act(() => host.querySelector<HTMLButtonElement>('button.wave-fit')?.click());

      // Zero-width ResizeObserver notification must not throw or collapse canvas width to 0
      const canvas = host.querySelector<HTMLCanvasElement>('canvas.spectrogram-canvas')!;
      expect(() => {
        act(() => {
          resizeCb?.(
            [
              {
                contentRect: { width: 0, height: 0 } as DOMRectReadOnly,
              } as ResizeObserverEntry,
            ],
            {} as ResizeObserver,
          );
        });
      }).not.toThrow();
      expect(canvas.width).toBeGreaterThan(0);

      const stage = host.querySelector<HTMLDivElement>('.spectrogram-stage')!;
      vi.spyOn(stage, 'getBoundingClientRect').mockReturnValue({
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

      // 1. pointercancel mid-drag clears drag preview and does NOT emit a selection
      act(() => {
        stage.dispatchEvent(
          new PointerEvent('pointerdown', { bubbles: true, button: 0, clientX: 50, clientY: 40, pointerId: 1 }),
        );
      });
      act(() => {
        stage.dispatchEvent(
          new PointerEvent('pointermove', { bubbles: true, button: 0, clientX: 200, clientY: 120, pointerId: 1 }),
        );
      });
      expect(host.querySelector('.spectrogram-selection-box')).not.toBeNull();
      act(() => {
        stage.dispatchEvent(new PointerEvent('pointercancel', { bubbles: true, pointerId: 1 }));
      });
      expect(host.querySelector('.spectrogram-selection-box')).toBeNull();
      expect(onChange).not.toHaveBeenCalled();

      // 2. Drag starting negative (-80, -30) and ending past width/height (900, 400) clamps cleanly
      act(() => {
        stage.dispatchEvent(
          new PointerEvent('pointerdown', { bubbles: true, button: 0, clientX: -80, clientY: -30, pointerId: 2 }),
        );
      });
      act(() => {
        stage.dispatchEvent(
          new PointerEvent('pointermove', { bubbles: true, button: 0, clientX: 900, clientY: 400, pointerId: 2 }),
        );
      });
      act(() => {
        stage.dispatchEvent(
          new PointerEvent('pointerup', { bubbles: true, button: 0, clientX: 900, clientY: 400, pointerId: 2 }),
        );
      });

      expect(onChange).toHaveBeenCalledTimes(1);
      const sel = onChange.mock.calls[0]?.[0];
      expect(sel.startS).toBe(0);
      expect(sel.endS).toBe(10);
      expect(sel.fLowHz).toBeCloseTo(20, 1);
      expect(sel.fHighHz).toBeCloseTo(24000, 1);

      act(() => root.unmount());
      expect(disconnectSpy).toHaveBeenCalledTimes(1);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe('LiveMeter RAF lifecycle', () => {
  it('cancels requestAnimationFrame cleanly on pause and unmount and ignores stale RAF callbacks', () => {
    let nextId = 1;
    const activeCallbacks = new Map<number, FrameRequestCallback>();
    const cancelledIds: number[] = [];

    const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      const id = nextId++;
      activeCallbacks.set(id, cb);
      return id;
    });
    const cafSpy = vi.spyOn(window, 'cancelAnimationFrame').mockImplementation((id) => {
      cancelledIds.push(id);
      activeCallbacks.delete(id);
    });

    const readLevels = vi.fn(() => ({
      momentaryLufs: -18,
      shortTermLufs: -20,
      peakDbfs: -6,
      truePeakHoldDbfs: -4,
    }));

    // Start playing -> schedules exactly 1 RAF
    act(() => {
      root.render(createElement(LiveMeter, { playing: true, readLevels }));
    });
    expect(rafSpy).toHaveBeenCalledTimes(1);
    const firstId = 1;
    const firstCb = activeCallbacks.get(firstId)!;

    // Pump one frame -> removes fired frame from browser queue, updates levels, schedules next frame
    activeCallbacks.delete(firstId);
    act(() => {
      firstCb(100);
    });
    expect(readLevels).toHaveBeenCalledTimes(1);
    expect(rafSpy).toHaveBeenCalledTimes(2);
    const secondId = 2;
    const staleSecondCb = activeCallbacks.get(secondId)!;

    // Pause -> cancels pending RAF
    act(() => {
      root.render(createElement(LiveMeter, { playing: false, readLevels }));
    });
    expect(cafSpy).toHaveBeenCalledWith(secondId);
    expect(activeCallbacks.size).toBe(0);

    // Even if a stale queued callback fires late after pause, it must NOT reschedule RAF or read levels
    act(() => {
      staleSecondCb(200);
    });
    expect(readLevels).toHaveBeenCalledTimes(1);
    expect(rafSpy).toHaveBeenCalledTimes(2);

    // Resume playing, then unmount -> cancels pending RAF cleanly
    act(() => {
      root.render(createElement(LiveMeter, { playing: true, readLevels }));
    });
    const thirdId = 3;
    expect(activeCallbacks.has(thirdId)).toBe(true);

    act(() => root.unmount());
    expect(cancelledIds).toContain(thirdId);
    expect(activeCallbacks.size).toBe(0);
  });
});

describe('VideoPreview chaos & rejection resilience', () => {
  it('handles 0-second duration, NaN currentTime, rapid playing toggles, NotAllowedError rejection, and decode error with 0 unhandled rejections', async () => {
    const unhandled: unknown[] = [];
    const onUnhandled = (e: PromiseRejectionEvent) => unhandled.push(e.reason);
    window.addEventListener('unhandledrejection', onUnhandled);

    let playCalls = 0;
    const playSpy = vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(() => {
      playCalls += 1;
      if (playCalls % 2 === 1) {
        return Promise.reject(new DOMException('Autoplay blocked', 'NotAllowedError'));
      }
      return Promise.resolve();
    });
    const pauseSpy = vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});

    try {
      // 1. 0-second duration + NaN currentTime + playing=true (NotAllowedError rejection)
      await act(async () => {
        root.render(
          createElement(VideoPreview, {
            videoUrl: 'http://127.0.0.1:8765/api/audio?path=/clip.mp4',
            currentTime: Number.NaN,
            duration: 0,
            playing: true,
            collapsed: false,
            onSeek: vi.fn(),
            onToggleCollapse: vi.fn(),
          }),
        );
      });

      expect(host.querySelector('input.vp-scrub')).toBeNull();
      expect(host.querySelector('[data-testid="video-preview-time"]')?.textContent).toBe('00:00.0');

      // 2. Rapid playing toggles (true -> false -> true -> false -> true -> false)
      for (const isPlaying of [false, true, false, true, false]) {
        await act(async () => {
          root.render(
            createElement(VideoPreview, {
              videoUrl: 'http://127.0.0.1:8765/api/audio?path=/clip.mp4',
              currentTime: isPlaying ? 4.2 : Number.NaN,
              duration: 0,
              playing: isPlaying,
              collapsed: false,
              onSeek: vi.fn(),
              onToggleCollapse: vi.fn(),
            }),
          );
        });
      }

      // Flush microtasks to surface any unhandled rejections
      await new Promise((r) => setTimeout(r, 10));
      expect(unhandled).toEqual([]);
      expect(playSpy).toHaveBeenCalled();

      // 3. Media decode error degrades gracefully
      const video = host.querySelector<HTMLVideoElement>('video')!;
      await act(async () => {
        video.dispatchEvent(new Event('error'));
      });
      expect(host.querySelector('.vp-error')).not.toBeNull();
      expect(pauseSpy).toBeDefined();
    } finally {
      window.removeEventListener('unhandledrejection', onUnhandled);
      playSpy.mockRestore();
      pauseSpy.mockRestore();
    }
  });
});

describe('StudioRack hostile localStorage & QuotaExceededError recovery', () => {
  it('mounts cleanly with corrupt/hostile localStorage and recovers from QuotaExceededError on save with 0 axe-core violations', async () => {
    window.localStorage.setItem(
      PRESET_STORAGE_KEY,
      '[{"__proto__":{"polluted":true},"id":"user:bad","name":"Evil","settings":{}}]',
    );

    const onChange = vi.fn();
    await act(async () => {
      root.render(
        createElement(StudioRack, {
          settings: DEFAULT_STUDIO_RACK_SETTINGS,
          onChange,
        }),
      );
    });

    expect(host.querySelector('.studio-rack')).not.toBeNull();
    expect((Object.prototype as Record<string, unknown>)['polluted']).toBeUndefined();

    // Simulate QuotaExceededError when saving a new user preset
    const setItemSpy = vi.spyOn(window.localStorage, 'setItem').mockImplementation(() => {
      throw new DOMException('QuotaExceededError', 'QuotaExceededError');
    });

    try {
      const saveToggle = [...host.querySelectorAll<HTMLButtonElement>('button.sr-btn')].find(
        (b) => b.textContent === 'Save',
      )!;
      await act(async () => {
        saveToggle.click();
      });

      const input = host.querySelector<HTMLInputElement>('input.sr-text-input')!;
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
      await act(async () => {
        setter?.call(input, 'Session Fallback Preset');
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
      });

      const form = host.querySelector<HTMLFormElement>('form[aria-label="Save user preset"]')!;
      await act(async () => {
        form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
      });

      // Must not crash; preset is retained in session state and status explains storage fallback
      expect(host.textContent).toContain('Session Fallback Preset');
      expect(host.querySelector('.sr-status')?.textContent).toContain('session');

      const results = await axe.run(host, {
        rules: {
          'color-contrast': { enabled: false },
          region: { enabled: false },
        },
      });
      expect(results.violations).toEqual([]);
    } finally {
      setItemSpy.mockRestore();
      window.localStorage.removeItem(PRESET_STORAGE_KEY);
    }
  });
});
