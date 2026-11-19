import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import axe from 'axe-core';
import type { AudioAnalysis } from '../api/types';
import { waveView } from '../render/viewWindow';
import {
  clearSelectionRange,
  markSelectionEnd,
  markSelectionStart,
} from '../state/actions';
import { getState, useStore } from '../state/store';
import { WaveformDisplay } from './WaveformDisplay';

const mockPlayer = vi.hoisted(() => ({
  time: 0,
  activeDeck: 'original' as 'original' | 'cleaned',
  hasDeck: vi.fn((_deck: 'original' | 'cleaned') => true),
  seek: vi.fn((t: number) => {
    mockPlayer.time = t;
  }),
  subscribe: vi.fn(() => () => undefined),
}));

vi.mock('../audio/player', () => ({
  getPlayer: () => mockPlayer,
}));

vi.mock('../render/WaveformHost', () => ({
  WaveformHost: class {
    setView = vi.fn();
    setData = vi.fn();
    clear = vi.fn();
    setFocus = vi.fn();
    setHighlight = vi.fn();
    setUnits = vi.fn();
    setPlayhead = vi.fn();
    setHover = vi.fn();
    requestStats = vi.fn();
    dispose = vi.fn();
  },
}));

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const pristine = useStore.getState();

let host: HTMLElement;
let root: Root;

function fakeAnalysis(duration = 60): AudioAnalysis {
  return {
    path: '/take.wav',
    duration_s: duration,
    sample_rate: 48000,
    channels: 1,
    peaks: { min: [-0.5, -0.4], max: [0.5, 0.4] },
    rms_db: [-20, -22],
    spectrum: { freqs_hz: [100, 1000], db: [-30, -25] },
    loudness: { integrated_lufs: -20, true_peak_dbtp: -1 },
    noise_floor_db: -50,
  };
}

beforeEach(() => {
  useStore.setState(pristine, true);
  waveView.clear();
  mockPlayer.time = 0;
  mockPlayer.hasDeck.mockReturnValue(true);
  mockPlayer.seek.mockClear();
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
    root.render(createElement(WaveformDisplay));
  });
}

describe('waveform region picker helpers', () => {
  it('marks start and end from live playhead time and preserves ordering', () => {
    useStore.setState({
      source: { path: '/take.wav', name: 'take.wav', origin: 'file' },
      original: fakeAnalysis(60),
      duration: 60,
    });

    mockPlayer.time = 15.25;
    markSelectionStart();
    expect(getState().selectionRange).toEqual({ start: 15.25, end: 60 });

    mockPlayer.time = 28.5;
    markSelectionEnd();
    expect(getState().selectionRange).toEqual({ start: 15.25, end: 28.5 });

    // Marking end before existing start resets start cleanly to 0 (never inverted).
    mockPlayer.time = 10;
    markSelectionEnd();
    expect(getState().selectionRange).toEqual({ start: 0, end: 10 });

    // Marking start after existing end spans to clip end.
    mockPlayer.time = 40;
    markSelectionStart();
    expect(getState().selectionRange).toEqual({ start: 40, end: 60 });

    clearSelectionRange();
    expect(getState().selectionRange).toBeNull();
  });

  it('is a safe no-op when no audio is loaded', () => {
    mockPlayer.time = 12;
    markSelectionStart();
    markSelectionEnd();
    expect(getState().selectionRange).toBeNull();
  });
});

describe('WaveformDisplay interactive region picker', () => {
  it('exposes In / Out / Clear buttons and updates selection overlay without breaking seek', async () => {
    useStore.setState({
      source: { path: '/take.wav', name: 'take.wav', origin: 'file' },
      original: fakeAnalysis(100),
      duration: 100,
    });
    await render();

    const inBtn = host.querySelector<HTMLButtonElement>(
      'button[aria-label="In — mark region start at playhead"]',
    );
    const outBtn = host.querySelector<HTMLButtonElement>(
      'button[aria-label="Out — mark region end at playhead"]',
    );
    expect(inBtn?.disabled).toBe(false);
    expect(outBtn?.disabled).toBe(false);
    expect(host.querySelector('button[aria-label="Clear selected region"]')).toBeNull();

    mockPlayer.time = 15;
    await act(async () => {
      inBtn?.click();
    });
    mockPlayer.time = 28;
    await act(async () => {
      outBtn?.click();
    });

    expect(getState().selectionRange).toEqual({ start: 15, end: 28 });

    const overlay = host.querySelector<HTMLDivElement>('.wave-sel-overlay');
    expect(overlay?.style.display).toBe('');
    expect(Number.parseFloat(overlay?.style.left ?? '')).toBeCloseTo(15, 4);
    expect(Number.parseFloat(overlay?.style.width ?? '')).toBeCloseTo(13, 4);

    const clearBtn = host.querySelector<HTMLButtonElement>(
      'button[aria-label="Clear selected region"]',
    );
    expect(clearBtn).not.toBeNull();
    await act(async () => {
      clearBtn?.click();
    });
    expect(getState().selectionRange).toBeNull();
    expect(overlay?.style.display).toBe('none');
  });

  it('supports Shift+drag range selection while preserving plain click-to-seek', async () => {
    useStore.setState({
      source: { path: '/take.wav', name: 'take.wav', origin: 'file' },
      original: fakeAnalysis(100),
      duration: 100,
    });
    await render();

    const wrap = host.querySelector<HTMLDivElement>('#wave-canvas-wrap')!;
    vi.spyOn(wrap, 'getBoundingClientRect').mockReturnValue({
      left: 0,
      top: 0,
      right: 1000,
      bottom: 100,
      width: 1000,
      height: 100,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });

    // Plain click seeks without creating a selection.
    await act(async () => {
      wrap.dispatchEvent(
        new PointerEvent('pointerdown', { button: 0, clientX: 250, pointerId: 1, bubbles: true }),
      );
      wrap.dispatchEvent(
        new PointerEvent('pointerup', { button: 0, clientX: 250, pointerId: 1, bubbles: true }),
      );
    });
    expect(mockPlayer.seek).toHaveBeenCalledWith(25);
    expect(getState().selectionRange).toBeNull();

    // Shift+drag creates an ordered selection range without seeking.
    mockPlayer.seek.mockClear();
    await act(async () => {
      wrap.dispatchEvent(
        new PointerEvent('pointerdown', {
          button: 0,
          clientX: 400,
          shiftKey: true,
          pointerId: 2,
          bubbles: true,
        }),
      );
      wrap.dispatchEvent(
        new PointerEvent('pointermove', {
          button: 0,
          clientX: 150,
          shiftKey: true,
          pointerId: 2,
          bubbles: true,
        }),
      );
      wrap.dispatchEvent(
        new PointerEvent('pointerup', {
          button: 0,
          clientX: 150,
          shiftKey: true,
          pointerId: 2,
          bubbles: true,
        }),
      );
    });
    expect(mockPlayer.seek).not.toHaveBeenCalled();
    expect(getState().selectionRange).toEqual({ start: 15, end: 40 });

    // Degenerate zero-width Shift+click clears selection cleanly.
    await act(async () => {
      wrap.dispatchEvent(
        new PointerEvent('pointerdown', {
          button: 0,
          clientX: 300,
          shiftKey: true,
          pointerId: 3,
          bubbles: true,
        }),
      );
      wrap.dispatchEvent(
        new PointerEvent('pointerup', {
          button: 0,
          clientX: 300,
          shiftKey: true,
          pointerId: 3,
          bubbles: true,
        }),
      );
    });
    expect(getState().selectionRange).toBeNull();
  });

  it('handles I, O, and Escape keyboard shortcuts on the waveform display', async () => {
    useStore.setState({
      source: { path: '/take.wav', name: 'take.wav', origin: 'file' },
      original: fakeAnalysis(80),
      duration: 80,
    });
    await render();

    const wrap = host.querySelector<HTMLDivElement>('#wave-canvas-wrap')!;

    mockPlayer.time = 12;
    await act(async () => {
      wrap.dispatchEvent(new KeyboardEvent('keydown', { key: 'i', bubbles: true, cancelable: true }));
    });
    expect(getState().selectionRange).toEqual({ start: 12, end: 80 });

    mockPlayer.time = 30;
    await act(async () => {
      wrap.dispatchEvent(new KeyboardEvent('keydown', { key: 'o', bubbles: true, cancelable: true }));
    });
    expect(getState().selectionRange).toEqual({ start: 12, end: 30 });

    await act(async () => {
      wrap.dispatchEvent(
        new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }),
      );
    });
    expect(getState().selectionRange).toBeNull();
  });

  it('passes axe-core accessibility checks with active region picker controls', async () => {
    useStore.setState({
      source: { path: '/take.wav', name: 'take.wav', origin: 'file' },
      original: fakeAnalysis(80),
      duration: 80,
      selectionRange: { start: 10, end: 25 },
    });
    await render();

    const results = await axe.run(host, {
      runOnly: {
        type: 'tag',
        values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'],
      },
      rules: {
        'color-contrast': { enabled: false },
        region: { enabled: false },
      },
    });
    const seriousOrCritical = results.violations.filter(
      (v) => v.impact === 'serious' || v.impact === 'critical',
    );
    expect(seriousOrCritical).toEqual([]);
  });
});
