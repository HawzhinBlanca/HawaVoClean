// components/Transport.test.ts — Unit & accessibility tests for Transport + LiveMeter pro audition controls.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import axe from 'axe-core';
import type { AudioAnalysis } from '../api/types';
import { getPlayer, type LiveMeterLevels } from '../audio/player';
import { useStore } from '../state/store';
import { LiveMeter } from './LiveMeter';
import { Transport } from './Transport';
import { usePlayerSync } from '../App';

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const ORIG_URL = 'http://127.0.0.1:8765/api/audio?path=/orig.wav';
const CLEAN_URL = 'http://127.0.0.1:8765/api/audio?path=/clean.wav';

function mockAnalysis(path = '/orig.wav'): AudioAnalysis {
  return {
    path,
    sample_rate: 48000,
    channels: 1,
    duration_s: 12.0,
    loudness: { integrated_lufs: -18.0, true_peak_dbtp: -1.5 },
    noise_floor_db: -54.0,
    rms_db: [-20, -18, -19, -21],
    peaks: { min: [-0.5, -0.6, -0.4, -0.3], max: [0.5, 0.6, 0.4, 0.3] },
    spectrum: { freqs_hz: [100, 1000, 5000, 10000], db: [-30, -20, -38, -50] },
  };
}

function stubAudioFetch(): void {
  vi.stubGlobal('URL', {
    ...URL,
    createObjectURL: vi.fn(() => `blob:transport-test-${Math.random().toString(16).slice(2)}`),
    revokeObjectURL: vi.fn(),
  });
  vi.stubGlobal(
    'fetch',
    vi.fn(async (_url: string, init?: RequestInit) => {
      const headers = new Headers();
      if (init?.headers) {
        headers.set('Content-Range', 'bytes 0-0/16');
        headers.set('Content-Length', '1');
      }
      return {
        ok: true,
        status: init?.headers ? 206 : 200,
        headers,
        arrayBuffer: async () => new ArrayBuffer(1),
        blob: async () => new Blob([new Uint8Array(16)]),
      } as unknown as Response;
    }),
  );
}

async function settle(): Promise<void> {
  for (let i = 0; i < 8; i++) await Promise.resolve();
}

let host: HTMLElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  useStore.getState().resetForNewSource();
  const p = getPlayer();
  p.setDeltaMode(false);
  p.setLoopEnabled(false);
  p.setLoopRange(null);
  p.setWetDryMix(1);
  p.resetPeakHold();
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.unstubAllGlobals();
});

describe('Transport pro audition controls', () => {
  it('renders Removed, Loop, and LiveMeter, and hides Wet/Dry Mix when only Original is loaded', () => {
    act(() => {
      useStore.setState({
        original: mockAnalysis('/orig.wav'),
        cleanedPath: null,
        abMode: 'original',
      });
      root.render(createElement(Transport));
    });

    const deltaBtn = host.querySelector<HTMLButtonElement>('button.btn-delta');
    const loopBtn = host.querySelector<HTMLButtonElement>('button.btn-loop');
    const mixSlider = host.querySelector<HTMLInputElement>('input.mix-slider');
    const meter = host.querySelector('.live-meter');

    expect(deltaBtn).not.toBeNull();
    expect(deltaBtn?.disabled).toBe(true);
    expect(deltaBtn?.title).toBe('Listen to removed audio only (D)');
    expect(deltaBtn?.getAttribute('aria-pressed')).toBe('false');

    expect(loopBtn).not.toBeNull();
    expect(loopBtn?.disabled).toBe(false);
    expect(loopBtn?.title).toBe('Loop selection or clip (L)');
    expect(loopBtn?.getAttribute('aria-pressed')).toBe('false');

    expect(mixSlider).toBeNull();
    expect(meter).not.toBeNull();

    // Row 1 holds only Play, A/B switch, and Timecode; Level Match lives in `.transport-pro` (row 2)
    // so `.timecode` never overlaps `.btn-level-match` at 326–366 px right-column widths.
    const transportEl = host.querySelector('.transport')!;
    const row1DirectChildren = [...transportEl.children].filter(
      (el) => !el.classList.contains('transport-pro') && !el.classList.contains('deckfault'),
    );
    expect(row1DirectChildren).toHaveLength(3);
    expect(host.querySelector('.transport-pro > button.btn-level-match')).not.toBeNull();
  });

  it('enables Removed, Loop, and Wet/Dry Mix when both decks are served and wires them to DualPlayer', async () => {
    stubAudioFetch();
    const p = getPlayer();
    p.load('original', ORIG_URL);
    p.load('cleaned', CLEAN_URL);
    await settle();

    act(() => {
      useStore.setState({
        original: mockAnalysis('/orig.wav'),
        cleaned: mockAnalysis('/clean.wav'),
        cleanedPath: '/clean.wav',
        artifacts: { master: true, json: true, txt: true, reason: '' },
        abMode: 'cleaned',
      });
      root.render(createElement(Transport));
    });

    const deltaBtn = host.querySelector<HTMLButtonElement>('button.btn-delta');
    const loopBtn = host.querySelector<HTMLButtonElement>('button.btn-loop');
    const mixSlider = host.querySelector<HTMLInputElement>('input.mix-slider');

    expect(deltaBtn?.disabled).toBe(false);
    expect(loopBtn?.disabled).toBe(false);
    expect(mixSlider).not.toBeNull();
    expect(mixSlider?.getAttribute('aria-label')).toBe('Wet dry mix');
    expect(mixSlider?.value).toBe('100');

    // Toggle Removed (Delta)
    act(() => {
      deltaBtn?.click();
    });
    expect(p.snapshot().deltaMode).toBe(true);
    expect(deltaBtn?.getAttribute('aria-pressed')).toBe('true');

    // Toggle Loop
    act(() => {
      loopBtn?.click();
    });
    expect(p.snapshot().loopEnabled).toBe(true);
    expect(loopBtn?.getAttribute('aria-pressed')).toBe('true');

    // Adjust Wet/Dry Mix via native value setter + input event
    act(() => {
      if (mixSlider) {
        const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
        setter?.call(mixSlider, '40');
        mixSlider.dispatchEvent(new Event('input', { bubbles: true }));
        mixSlider.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
    expect(p.snapshot().wetDryMix).toBeCloseTo(0.4, 2);
    expect(host.querySelector('.mix-pct')?.textContent).toBe('40%');
  });

  it('syncs store selectionRange to player loopRange', () => {
    function SyncedTransport() {
      usePlayerSync();
      return createElement(Transport);
    }
    act(() => {
      useStore.setState({
        original: mockAnalysis('/orig.wav'),
        selectionRange: { start: 2.5, end: 7.5 },
      });
      root.render(createElement(SyncedTransport));
    });
    expect(getPlayer().snapshot().loopRange).toEqual([2.5, 7.5]);

    act(() => {
      useStore.setState({ selectionRange: null });
    });
    expect(getPlayer().snapshot().loopRange).toBeNull();
  });

  it('passes axe-core WCAG 2.2 AA accessibility checks with all pro audition controls active', async () => {
    act(() => {
      useStore.setState({
        original: mockAnalysis('/orig.wav'),
        cleaned: mockAnalysis('/clean.wav'),
        cleanedPath: '/clean.wav',
        artifacts: { master: true, json: true, txt: true, reason: '' },
        abMode: 'cleaned',
        loudnessMatch: true,
        gainOffsetDb: 1.5,
      });
      root.render(createElement(Transport));
    });

    const results = await axe.run(host, {
      rules: {
        'color-contrast': { enabled: false },
        region: { enabled: false },
      },
    });
    expect(results.violations).toEqual([]);
  });
});

describe('LiveMeter component', () => {
  it('only schedules requestAnimationFrame while playing, updates readings, and resets peak hold on click', () => {
    const rafQueue: FrameRequestCallback[] = [];
    const rafSpy = vi.fn((cb: FrameRequestCallback) => {
      rafQueue.push(cb);
      return rafQueue.length;
    });
    const cancelSpy = vi.fn();
    vi.stubGlobal('requestAnimationFrame', rafSpy);
    vi.stubGlobal('cancelAnimationFrame', cancelSpy);

    let current: LiveMeterLevels = {
      momentaryLufs: -16.4,
      shortTermLufs: -18.2,
      peakDbfs: -0.2,
      truePeakHoldDbfs: -0.2,
    };
    const resetSpy = vi.fn(() => {
      current = { ...current, truePeakHoldDbfs: -70 };
    });

    // Stopped -> zero RAF requests
    act(() => {
      root.render(
        createElement(LiveMeter, {
          playing: false,
          readLevels: () => current,
          onResetPeak: resetSpy,
        }),
      );
    });
    expect(rafSpy).not.toHaveBeenCalled();

    // Playing -> schedules RAF and shows live LUFS + CLIP indicator
    act(() => {
      root.render(
        createElement(LiveMeter, {
          playing: true,
          readLevels: () => current,
          onResetPeak: resetSpy,
        }),
      );
    });
    expect(rafSpy).toHaveBeenCalledTimes(1);

    act(() => {
      const cb = rafQueue.shift();
      cb?.(100);
    });
    expect(host.textContent).toContain('-16.4');
    expect(host.textContent).toContain('-18.2');
    const peakBtn = host.querySelector<HTMLButtonElement>('button.live-meter-peak');
    expect(peakBtn?.textContent).toBe('CLIP');

    // Clicking peak hold resets latch
    act(() => {
      peakBtn?.click();
    });
    expect(resetSpy).toHaveBeenCalledTimes(1);
    expect(peakBtn?.textContent).toBe('--.- dB');

    // Stopping cancels RAF
    act(() => {
      root.render(
        createElement(LiveMeter, {
          playing: false,
          readLevels: () => current,
          onResetPeak: resetSpy,
        }),
      );
    });
    expect(cancelSpy).toHaveBeenCalled();
  });
});
